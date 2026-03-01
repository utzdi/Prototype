import time
import json
from pathlib import Path

import anthropic

from .base import AnalysisResult, BaseMLLMProvider


class AnthropicProvider(BaseMLLMProvider):
    name = "Claude Vision"
    
    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514"):
        self.client = anthropic.Anthropic(api_key=api_key)
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

            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                temperature=0,
                tools=[
                    {
                        "name": "presence_check",
                        "description": "Return presence check as structured JSON.",
                        "input_schema": schema,
                    }
                ],
                tool_choice={"type": "tool", "name": "presence_check"},
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": mime_a,
                                    "data": image_a_b64,
                                },
                            },
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": mime_b,
                                    "data": image_b_b64,
                                },
                            },
                        ],
                    }
                ],
            )
            
            latency_ms = (time.time() - start_time) * 1000
            raw_response = ""
            tokens_used = (response.usage.input_tokens + response.usage.output_tokens) if response.usage else None

            tool_input = None
            if response.content:
                for block in response.content:
                    # anthropic SDK returns objects with .type or dicts depending on version
                    block_type = getattr(block, "type", None) or (block.get("type") if isinstance(block, dict) else None)
                    if block_type == "tool_use":
                        name = getattr(block, "name", None) or (block.get("name") if isinstance(block, dict) else None)
                        if name == "presence_check":
                            tool_input = getattr(block, "input", None) or (block.get("input") if isinstance(block, dict) else None)
                            break

            if tool_input is None:
                # Fallback: if model returned text, try parse it
                raw_response = response.content[0].text if response.content and hasattr(response.content[0], "text") else ""
            else:
                raw_response = json.dumps(tool_input, ensure_ascii=False)

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
