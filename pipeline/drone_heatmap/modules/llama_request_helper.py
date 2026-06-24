import base64
import json
import os
from typing import Any, Dict, List, Optional
import requests
import time
import cv2
import numpy as np

class BaseRemoteClient:
    """Shared base class to handle connection configurations and image encoding."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8080):
        self.host = host
        self.port = port

    def _encode_image_to_base64(self, image_path: str) -> str:
        """Reads a local image and encodes it to a base64 string."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Target image not found at: {image_path}")

        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")


class LlamaVlmClient(BaseRemoteClient):
    """Client for text and vision requests targeting a Llama VLM server."""

    @property
    def endpoint(self) -> str:
        return f"http://{self.host}:{self.port}/v1/chat/completions"

    def request(
        self,
        prompt: str,
        image_path: Optional[str] = None,
        image_base64: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 512,
        **extra_settings,
    ) -> str:
        content_payload = [{"type": "text", "text": prompt}]

        if image_path:
            base64_image = self._encode_image_to_base64(image_path)
            content_payload.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                }
            )
        elif image_base64:
            content_payload.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                }
            )

        payload = {
            "model": "local-vlm",
            "messages": [{"role": "user", "content": content_payload}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        payload.update(extra_settings)

        try:
            response = requests.post(self.endpoint, json=payload, timeout=60)
            if response.status_code == 200:
                response_data = response.json()
                return response_data["choices"][0]["message"]["content"]
            else:
                # RAISE an error instead of returning a string string
                raise RuntimeError(f"VLM Server Error ({response.status_code}): {response.text}")
                
        except requests.exceptions.ConnectionError as e:
            raise ConnectionError(f"Could not connect to VLM server at {self.endpoint}.") from e


class Sam3Client(BaseRemoteClient):
    """Client for promptable object segmentation using base64 encoded images."""

    @property
    def endpoint(self) -> str:
        return f"http://{self.host}:{self.port}/v1/predict"

    def request(
        self,
        image_base64: Any,
        prompts: List[str],
        confidence: float = 0.25,
    ) -> Dict[str, Any]:
        """Sends an image (base64 string OR raw NumPy array) and metadata to the SAM 3 server."""
        try:
            # Smart check: if raw NumPy frame is passed from live loop, auto-encode to string text
            if isinstance(image_base64, np.ndarray):
                _, buffer = cv2.imencode('.jpg', image_base64)
                image_base64 = base64.b64encode(buffer).decode('utf-8')
            
            elif isinstance(image_base64, str) and "," in image_base64:
                image_base64 = image_base64.split(",", 1)[1]

            payload = {
                "image": image_base64,
                "prompts": prompts,
                "conf": confidence
            }

            network_start = time.perf_counter()
            response = requests.post(self.endpoint, json=payload, timeout=60)
            network_time = time.perf_counter() - network_start

            if response.status_code == 200:
                data = response.json()
                data["client_network_time"] = network_time
                return data
            else:
                return {"error": f"API Error ({response.status_code}): {response.text}"}
                
        except requests.exceptions.ConnectionError:
            return {"error": "Connection Error: Could not connect to SAM3 server."}
        except Exception as e:
            return {"error": f"Unexpected Error: {str(e)}"}