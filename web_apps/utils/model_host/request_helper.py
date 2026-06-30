import base64
import json
import io
import os
from typing import Any, Dict, List, Optional
import requests
import time


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
        """Dynamically builds the OpenAI-compliant chat completions URL."""
        return f"http://{self.host}:{self.port}/v1/chat/completions"

    def request(
        self,
        prompt: str,
        image_path: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 512,
        **extra_settings,
    ) -> str:
        """Sends a text prompt and an optional image to the local VLM server."""
        content_payload = [{"type": "text", "text": prompt}]

        if image_path:
            base64_image = self._encode_image_to_base64(image_path)
            content_payload.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
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
                return f"API Error ({response.status_code}): {response.text}"
        except requests.exceptions.ConnectionError:
            return f"Connection Error: Could not connect to VLM server at {self.endpoint}."
        except Exception as e:
            return f"Unexpected Error processing VLM request: {str(e)}"


class Sam3Client(BaseRemoteClient):
    """Client for promptable object segmentation using base64 encoded images."""

    @property
    def endpoint(self) -> str:
        return f"http://{self.host}:{self.port}/v1/predict"

    def request(
    self,
    image_base64: str,
    prompts: List[str],
    confidence: float = 0.25,
    ) -> Dict[str, Any]:
        """Sends a clean JSON payload matching the SAM3 Pydantic validation schema."""
        
        # Clean up the string if a data URI prefix accidentally got passed
        if "," in image_base64:
            image_base64 = image_base64.split(",", 1)[1]

        # Construct payload exactly like SAM3InferenceRequest schema
        payload = {
            "image": image_base64,
            "prompts": prompts,
            "conf": confidence
        }

        try:
            network_start = time.perf_counter()
            
            # Change from files/data to a clean json= argument
            response = requests.post(
                self.endpoint, 
                json=payload, 
                headers={"Content-Type": "application/json"},
                timeout=60
            )
            
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
