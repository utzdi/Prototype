import time
from pathlib import Path
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from .base import AnalysisResult, BaseMLLMProvider, DebugInfo, PresenceCheckResult


OPENAI_VISION_MODELS = [
    "gpt-5.2",
    "gpt-5",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4o",
    "gpt-4o-mini",
]


class OpenAIProvider(BaseMLLMProvider):
    name = "GPT-5.2"

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-5.2",
        max_image_dimension: int = 0,
        jpeg_quality: int = 0,
        image_detail: str = "auto",
    ):
        self.model = model
        self.max_image_dimension = max_image_dimension
        self.jpeg_quality = jpeg_quality
        self.image_detail = image_detail
        self.name = model

        llm = ChatOpenAI(api_key=api_key, model=model)
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
            prompt_sent=prompt,
            model_used=self.model,
            api_endpoint="ChatOpenAI / chat.completions (json_schema)",
        )

        try:
            size_a_orig, res_a_orig = self.get_image_info(image_a_path)
            size_b_orig, res_b_orig = self.get_image_info(image_b_path)
            debug_info.image_a_size_original_kb = size_a_orig
            debug_info.image_b_size_original_kb = size_b_orig
            debug_info.image_a_resolution_original = res_a_orig
            debug_info.image_b_resolution_original = res_b_orig

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

            message = HumanMessage(content=[
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_a};base64,{image_a_b64}",
                        "detail": self.image_detail,
                    },
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_b};base64,{image_b_b64}",
                        "detail": self.image_detail,
                    },
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
