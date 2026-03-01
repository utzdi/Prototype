import time
from pathlib import Path
from typing import Optional

from openai import OpenAI

from .base import AnalysisResult, BaseMLLMProvider


class OpenAIProvider(BaseMLLMProvider):
    name = "GPT-4V"
    
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
    
    def analyze(
        self,
        image_a_path: Path,
        image_b_path: Path,
        prompt: str,
        pair_id: str,
        element: str
    ) -> AnalysisResult:
        start_time = time.time()
        
        try:
            image_a_b64 = self.encode_image(image_a_path)
            image_b_b64 = self.encode_image(image_b_path)
            
            mime_a = self.get_mime_type(image_a_path)
            mime_b = self.get_mime_type(image_b_path)

            schema = self.get_presence_check_schema()

            raw_response = ""
            tokens_used = None

            # Prefer structured outputs via the Responses API. Fallback to chat.completions.
            try:
                response = self.client.responses.create(
                    model=self.model,
                    input=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": prompt},
                                {"type": "input_image", "image_url": f"data:{mime_a};base64,{image_a_b64}"},
                                {"type": "input_image", "image_url": f"data:{mime_b};base64,{image_b_b64}"},
                            ],
                        }
                    ],
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "presence_check",
                            "schema": schema,
                            "strict": True,
                        },
                    },
                    max_output_tokens=1024,
                )

                raw_response = getattr(response, "output_text", "") or ""
                if getattr(response, "usage", None):
                    tokens_used = getattr(response.usage, "total_tokens", None)
            except Exception:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:{mime_a};base64,{image_a_b64}",
                                        "detail": "high",
                                    },
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:{mime_b};base64,{image_b_b64}",
                                        "detail": "high",
                                    },
                                },
                            ],
                        }
                    ],
                    max_tokens=1024,
                )
                raw_response = response.choices[0].message.content or ""
                tokens_used = response.usage.total_tokens if response.usage else None

            latency_ms = (time.time() - start_time) * 1000

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
                raw_response=raw_response
            )
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return AnalysisResult(
                pair_id=pair_id,
                mllm=self.name,
                element=element,
                latency_ms=latency_ms,
                error=str(e)
            )
