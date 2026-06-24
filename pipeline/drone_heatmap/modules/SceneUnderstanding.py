import cv2
import numpy as np
import base64
import json
import sys
import os

# 1. Dynamically find the 'drone_heatmap' root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 2. Force Python to look at the root directory for imports
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Centralized architecture configuration imports
from config.prompts import REASONING_PROMPT, VLM_PROMPT
from config.lm_config import VLM_CONFIG, LLM_CONFIG  # ← LOAD FROM CONFIG SUBFOLDER
from modules.llama_request_helper import LlamaVlmClient

class SceneUnderstanding:
    def __init__(self):
        self.model = None
        self.vocabulary = {}
        self.vocabulary_alpha = 0.25

        # Sourced dynamically from ml_config.json via the loader
        self.vlm = LlamaVlmClient(host=VLM_CONFIG["host"], port=VLM_CONFIG["port"])
        self.llm = LlamaVlmClient(host=LLM_CONFIG["host"], port=LLM_CONFIG["port"])

    def _vocabulary_labels(self, max_size=12):
        # Sort by score descending and cap the length to prevent prompt/context bloat
        sorted_vocab = sorted(self.vocabulary.items(), key=lambda item: item[1], reverse=True)
        top_labels = [label for label, score in sorted_vocab[:max_size]]
        return sorted(top_labels)

    def _update_vocabulary(self, labels):
        for label_info in labels.values():
            if "label" not in label_info or "score" not in label_info:
                continue
            label = label_info["label"]
            score = float(label_info["score"])

            if label not in self.vocabulary:
                self.vocabulary[label] = score
            else:
                previous_score = self.vocabulary[label]
                self.vocabulary[label] = (
                    self.vocabulary_alpha * score
                    + (1 - self.vocabulary_alpha) * previous_score
                )

    def _loads_json_object(self, text):
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start == -1 or end == -1 or end <= start:
                raise
            return json.loads(text[start:end + 1])

    def _normalize_labels(self, labels):
        normalized = {}
        if not isinstance(labels, dict):
            return normalized

        for prompt, label_info in labels.items():
            if not isinstance(label_info, dict):
                continue
            if "label" not in label_info or "score" not in label_info:
                continue

            normalized[prompt] = {
                "label": str(label_info["label"]),
                "score": float(label_info["score"]),
            }
        return normalized

    def get_labels(self, image: np.ndarray, task: str):
        image = cv2.resize(
            image,
            (384, 384),
            interpolation=cv2.INTER_AREA
        )

        _, buffer = cv2.imencode(".jpg", image)
        image_b64 = base64.b64encode(buffer.tobytes()).decode("utf-8")
        
        # Stage 1: VLM Perception utilizing runtime config max_tokens
        vlm_response = self.vlm.request(
            prompt=VLM_PROMPT, 
            image_base64=image_b64, 
            max_tokens=VLM_CONFIG["max_tokens"]
        )   

        try:
            observations = self._loads_json_object(vlm_response)["observations"]
        except:
            print(f"❌ Critical: VLM responded with an unparseable schema.")
            print(f"Raw response was:\n{vlm_response}")
            observations = []

        vlm_conf_map = {}
        if isinstance(observations, list):
            for obs in observations:
                if isinstance(obs, dict) and "feature" in obs and "confidence" in obs:
                    vlm_conf_map[obs["feature"]] = float(obs["confidence"])

        # Stage 2: Instruction / Reasoning Model
        reasoning_prompt = REASONING_PROMPT.format(
            task=task,
            observations=json.dumps(observations, indent=2),
            vocabulary=json.dumps(self._vocabulary_labels(), indent=2),
        )

        # Targets specified context generation bounds dynamically
        llm_response = self.llm.request(prompt=reasoning_prompt, max_tokens=LLM_CONFIG["max_tokens"])
        labels = self._normalize_labels(self._loads_json_object(llm_response))

        # --- FUZZY TOKEN MATCHING BLOCK ---
        for feature_key, label_info in labels.items():
            llm_relevance = label_info["score"]
            
            key_words = set(feature_key.lower().replace(",", " ").split())
            vlm_confidence = None
            best_overlap = 0
            
            for vlm_feat, vlm_conf in vlm_conf_map.items():
                vlm_words = set(vlm_feat.lower().replace(",", " ").split())
                overlap = len(key_words.intersection(vlm_words))
                if overlap > best_overlap:
                    best_overlap = overlap
                    vlm_confidence = vlm_conf
            
            if vlm_confidence is None:
                vlm_confidence = 0.40

            combined_score = (llm_relevance / 100.0) * vlm_confidence
            label_info["combined_score"] = combined_score

        self._update_vocabulary(labels)

        print("OBSERVATIONS:")
        print(observations)

        print("LABELS (Fuzzy Matching Fixed):")
        print(labels)

        return labels