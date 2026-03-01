import time
from pathlib import Path
from typing import Optional

import requests
import json

from .base import (
    AnalysisResult,
    BaseMLLMProvider,
    DebugInfo,
)


VISION_MODEL_KEYWORDS = [
    "llava", "bakllava", "moondream", "minicpm", "gemma3",
    "llama3.2-vision", "llama4", "internvl", "cogvlm",
    "qwen2-vl", "qwen-vl", "qwen3-vl",
]


class OllamaProvider(BaseMLLMProvider):
    name = "LLaVA (Ollama)"
    
    def __init__(
        self,
        model: str = "llava:7b",
        base_url: str = "http://localhost:11434",
        api_key: Optional[str] = None,
        display_name: Optional[str] = None,
        max_image_dimension: int = 0,
        jpeg_quality: int = 0
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.max_image_dimension = max_image_dimension
        self.jpeg_quality = jpeg_quality
        if display_name:
            self.name = display_name
    
    def _get_headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers
    
    def is_available(self) -> bool:
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                headers=self._get_headers(),
                timeout=10
            )
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m.get("name", "") for m in models]
                return any(self.model in name or name in self.model for name in model_names)
            return False
        except requests.exceptions.RequestException:
            return False
    
    def analyze(
        self,
        image_a_path: Path,
        image_b_path: Path,
        prompt: str,
        pair_id: str,
        element: str
    ) -> AnalysisResult:
        start_time = time.time()
        
        debug = DebugInfo(
            api_endpoint=f"{self.base_url}/api/chat",
            model_used=self.model
        )
        
        try:
            orig_a_kb, orig_a_res = self.get_image_info(image_a_path)
            orig_b_kb, orig_b_res = self.get_image_info(image_b_path)
            debug.image_a_size_original_kb = orig_a_kb
            debug.image_b_size_original_kb = orig_b_kb
            debug.image_a_resolution_original = orig_a_res
            debug.image_b_resolution_original = orig_b_res
            
            image_a_b64 = self.encode_image(image_a_path, self.max_image_dimension, self.jpeg_quality)
            image_b_b64 = self.encode_image(image_b_path, self.max_image_dimension, self.jpeg_quality)
            
            debug.image_a_size_sent_kb = self.get_b64_size_kb(image_a_b64)
            debug.image_b_size_sent_kb = self.get_b64_size_kb(image_b_b64)
            
            if self.max_image_dimension > 0:
                debug.image_a_resolution_sent = f"max {self.max_image_dimension}px"
                debug.image_b_resolution_sent = f"max {self.max_image_dimension}px"
            else:
                debug.image_a_resolution_sent = orig_a_res
                debug.image_b_resolution_sent = orig_b_res
            
            full_prompt = f"""Du erhältst zwei Screenshots einer Webseite.
Bild 1 ist das Referenzbild (Screenshot A).
Bild 2 ist das Vergleichsbild (Screenshot B).

{prompt}"""
            
            debug.prompt_sent = full_prompt
            
            schema = self.get_presence_check_schema()

            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": full_prompt,
                        "images": [image_a_b64, image_b_b64],
                    }
                ],
                "stream": False,
                "format": schema,
                "options": {
                    "temperature": 0,
                    "num_predict": 1024,
                },
            }
            
            payload_json = json.dumps(payload)
            debug.request_payload_size_kb = len(payload_json.encode()) / 1024
            
            response = requests.post(
                f"{self.base_url}/api/chat",
                headers=self._get_headers(),
                data=payload_json,
                timeout=240
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
            if response.status_code != 200:
                return AnalysisResult(
                    pair_id=pair_id,
                    mllm=self.name,
                    element=element,
                    latency_ms=latency_ms,
                    error=f"Ollama API error: {response.status_code} - {response.text}",
                    debug=debug
                )
            
            result = response.json()
            raw_response = ""
            if isinstance(result, dict) and "message" in result and isinstance(result["message"], dict):
                raw_response = result["message"].get("content", "") or ""
            
            tokens_used = None
            if "eval_count" in result and "prompt_eval_count" in result:
                tokens_used = result["eval_count"] + result["prompt_eval_count"]
            
            presence_a, presence_b, reasoning, description_a, description_b = self._parse_response(raw_response)
            
            return AnalysisResult(
                pair_id=pair_id,
                mllm=self.name,
                element=element,
                description_a=description_a,
                description_b=description_b,
                presence_a=presence_a,
                presence_b=presence_b,
                reasoning=reasoning,
                tokens_used=tokens_used,
                latency_ms=latency_ms,
                raw_response=raw_response,
                debug=debug
            )
            
        except requests.exceptions.Timeout:
            latency_ms = (time.time() - start_time) * 1000
            return AnalysisResult(
                pair_id=pair_id,
                mllm=self.name,
                element=element,
                latency_ms=latency_ms,
                error="Timeout: Ollama response took too long",
                debug=debug
            )
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return AnalysisResult(
                pair_id=pair_id,
                mllm=self.name,
                element=element,
                latency_ms=latency_ms,
                error=str(e),
                debug=debug
            )


def check_ollama_status(
    base_url: str = "http://localhost:11434",
    api_key: Optional[str] = None
) -> dict:
    """Check Ollama status and available models."""
    try:
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        
        response = requests.get(f"{base_url}/api/tags", headers=headers, timeout=10)
        if response.status_code == 200:
            models = response.json().get("models", [])
            vision_models = [
                m["name"] for m in models
                if any(v in m["name"].lower() for v in VISION_MODEL_KEYWORDS)
            ]
            return {
                "running": True,
                "models": [m["name"] for m in models],
                "vision_models": vision_models
            }
        return {"running": False, "error": f"Status {response.status_code}"}
    except requests.exceptions.RequestException as e:
        return {"running": False, "error": str(e)}
