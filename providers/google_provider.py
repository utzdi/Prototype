import time
from pathlib import Path

import google.generativeai as genai
from PIL import Image

from .base import AnalysisResult, BaseMLLMProvider


class GoogleProvider(BaseMLLMProvider):
    name = "Gemini Pro"
    
    def __init__(self, api_key: str, model: str = "gemini-1.5-pro"):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)
    
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
            image_a = Image.open(image_a_path)
            image_b = Image.open(image_b_path)
            
            response = self.model.generate_content(
                [prompt, image_a, image_b],
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=1024,
                    temperature=0,
                    response_mime_type="application/json",
                )
            )
            
            latency_ms = (time.time() - start_time) * 1000
            raw_response = response.text if response.text else ""
            
            tokens_used = None
            if hasattr(response, "usage_metadata"):
                tokens_used = (
                    response.usage_metadata.prompt_token_count +
                    response.usage_metadata.candidates_token_count
                )
            
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
