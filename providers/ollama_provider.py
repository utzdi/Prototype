import time
from pathlib import Path
from typing import Optional

import requests

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from .base import (
    AnalysisResult,
    BaseMLLMProvider,
    DebugInfo,
    PresenceCheckResult,
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
        jpeg_quality: int = 0,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.max_image_dimension = max_image_dimension
        self.jpeg_quality = jpeg_quality
        if display_name:
            self.name = display_name

        if api_key:
            # Remote server with authentication: use OpenAI-compatible endpoint
            llm = ChatOpenAI(
                api_key=api_key,
                base_url=f"{self.base_url}/v1",
                model=model,
                max_tokens=1024,
                temperature=0,
            )
            self._endpoint_label = f"{self.base_url}/v1/chat/completions (OpenAI-compat)"
        else:
            # Local Ollama: use native ChatOllama with json_schema structured output
            llm = ChatOllama(
                base_url=self.base_url,
                model=model,
                temperature=0,
                num_predict=1024,
            )
            self._endpoint_label = f"{self.base_url}/api/chat (ChatOllama)"

        self.chain = llm.with_structured_output(
            PresenceCheckResult,
            method="json_schema",
            include_raw=True,
        )

    def analyze(
        self,
        image_a_path: Path,
        image_b_path: Path,
        prompt: str,
        pair_id: str,
        element: str,
    ) -> AnalysisResult:
        start_time = time.time()
        debug_info = DebugInfo(
            model_used=self.model,
            api_endpoint=self._endpoint_label,
        )

        try:
            orig_a_kb, orig_a_res = self.get_image_info(image_a_path)
            orig_b_kb, orig_b_res = self.get_image_info(image_b_path)
            debug_info.image_a_size_original_kb = orig_a_kb
            debug_info.image_b_size_original_kb = orig_b_kb
            debug_info.image_a_resolution_original = orig_a_res
            debug_info.image_b_resolution_original = orig_b_res

            image_a_b64 = self.encode_image(
                image_a_path, self.max_image_dimension, self.jpeg_quality
            )
            image_b_b64 = self.encode_image(
                image_b_path, self.max_image_dimension, self.jpeg_quality
            )

            debug_info.image_a_size_sent_kb = self.get_b64_size_kb(image_a_b64)
            debug_info.image_b_size_sent_kb = self.get_b64_size_kb(image_b_b64)

            mime_a = "image/jpeg" if self.max_image_dimension > 0 else self.get_mime_type(image_a_path)
            mime_b = "image/jpeg" if self.max_image_dimension > 0 else self.get_mime_type(image_b_path)

            # Prepend Ollama-specific context hint so the model knows image order
            full_prompt = (
                "Du erhältst zwei Screenshots einer Webseite.\n"
                "Bild 1 ist das Referenzbild (Screenshot A).\n"
                "Bild 2 ist das Vergleichsbild (Screenshot B).\n\n"
                + prompt
            )
            debug_info.prompt_sent = full_prompt

            message = HumanMessage(content=[
                {"type": "text", "text": full_prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_a};base64,{image_a_b64}"},
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_b};base64,{image_b_b64}"},
                },
            ])

            result = self.chain.invoke([message])
            parsed: Optional[PresenceCheckResult] = result.get("parsed")
            raw_msg = result.get("raw")

            latency_ms = (time.time() - start_time) * 1000

            raw_response = ""
            tokens_used = None
            if raw_msg is not None:
                raw_response = raw_msg.content if isinstance(raw_msg.content, str) else str(raw_msg.content)
                if raw_msg.usage_metadata:
                    tokens_used = raw_msg.usage_metadata.get("total_tokens")

            if parsed is None:
                raise ValueError(f"Structured output parsing failed: {result.get('parsing_error')}")

            return AnalysisResult(
                pair_id=pair_id,
                mllm=self.name,
                element=element,
                description_a=parsed.description_a,
                description_b=parsed.description_b,
                presence_a=parsed.presence_a,
                presence_b=parsed.presence_b,
                reasoning=parsed.reasoning,
                tokens_used=tokens_used,
                latency_ms=latency_ms,
                raw_response=raw_response,
                debug=debug_info,
            )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return AnalysisResult(
                pair_id=pair_id,
                mllm=self.name,
                element=element,
                latency_ms=latency_ms,
                error=str(e),
                debug=debug_info,
            )


def check_ollama_status(
    base_url: str = "http://localhost:11434",
    api_key: Optional[str] = None,
) -> dict:
    """Check Ollama status and available models via direct API call."""
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
                "vision_models": vision_models,
            }
        return {"running": False, "error": f"Status {response.status_code}"}
    except requests.exceptions.RequestException as e:
        return {"running": False, "error": str(e)}
