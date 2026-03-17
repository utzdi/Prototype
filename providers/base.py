from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import base64
import io
import logging
import json
from pathlib import Path

from PIL import Image
from pydantic import BaseModel

logger = logging.getLogger(__name__)


@dataclass
class DebugInfo:
    """Stores debug information for a single API call."""
    prompt_sent: str = ""
    image_a_size_original_kb: float = 0
    image_b_size_original_kb: float = 0
    image_a_size_sent_kb: float = 0
    image_b_size_sent_kb: float = 0
    image_a_resolution_original: str = ""
    image_b_resolution_original: str = ""
    image_a_resolution_sent: str = ""
    image_b_resolution_sent: str = ""
    request_payload_size_kb: float = 0
    api_endpoint: str = ""
    model_used: str = ""


@dataclass
class AnalysisResult:
    pair_id: str
    mllm: str
    element: str
    description_a: Optional[str] = None
    description_b: Optional[str] = None
    presence_a: Optional[bool] = None
    presence_b: Optional[bool] = None
    match: Optional[bool] = None
    reasoning: Optional[str] = None
    tokens_used: Optional[int] = None
    latency_ms: Optional[float] = None
    raw_response: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    error: Optional[str] = None
    debug: Optional[DebugInfo] = field(default=None, repr=False)
    ground_truth: Optional[str] = field(default=None)

    def __post_init__(self):
        if self.presence_a is not None and self.presence_b is not None:
            self.match = self.presence_a == self.presence_b

    @property
    def model_classification(self) -> Optional[str]:
        """Derive the 4-class label from presence_a / presence_b."""
        if self.presence_a is None or self.presence_b is None:
            return None
        if self.presence_a and self.presence_b:
            return "both"
        if self.presence_a and not self.presence_b:
            return "only_a"
        if not self.presence_a and self.presence_b:
            return "only_b"
        return "neither"

    @property
    def correct(self) -> Optional[bool]:
        """True if model_classification matches ground_truth, None if either is unknown."""
        if self.ground_truth is None or self.model_classification is None:
            return None
        return self.model_classification == self.ground_truth

    def to_dict(self, columns: list[str]) -> dict:
        result = {}
        for col in columns:
            if col == "timestamp":
                result[col] = self.timestamp.isoformat()
            elif col == "model_classification":
                result[col] = self.model_classification
            elif col == "correct":
                result[col] = self.correct
            elif hasattr(self, col):
                result[col] = getattr(self, col)
        return result


DEFAULT_MAX_DIMENSION = 1024
DEFAULT_JPEG_QUALITY = 80


class PresenceCheckResult(BaseModel):
    """Pydantic schema for structured MLLM output. Used with LangChain's with_structured_output()."""
    description_a: str
    description_b: str
    presence_a: bool
    presence_b: bool
    reasoning: str


PRESENCE_CHECK_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "description_a": {"type": "string"},
        "description_b": {"type": "string"},
        "presence_a": {"type": "boolean"},
        "presence_b": {"type": "boolean"},
        "reasoning": {"type": "string"},
    },
    "required": ["description_a", "description_b", "presence_a", "presence_b", "reasoning"],
    "additionalProperties": False,
}


class BaseMLLMProvider(ABC):
    name: str = "Base"
    
    @abstractmethod
    def analyze(
        self,
        image_a_path: Path,
        image_b_path: Path,
        prompt: str,
        pair_id: str,
        element: str
    ) -> AnalysisResult:
        pass
    
    @staticmethod
    def encode_image(image_path: Path, max_dimension: int = 0, quality: int = 0) -> str:
        """Encode image to base64, optionally resizing and compressing."""
        if max_dimension > 0:
            return BaseMLLMProvider._encode_resized(image_path, max_dimension, quality or DEFAULT_JPEG_QUALITY)
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    
    @staticmethod
    def _encode_resized(image_path: Path, max_dimension: int, quality: int) -> str:
        """Resize image to fit within max_dimension and encode as JPEG."""
        img = Image.open(image_path)
        if img.mode == "RGBA":
            bg = Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[3])
            img = bg
        elif img.mode != "RGB":
            img = img.convert("RGB")
        
        w, h = img.size
        if w > max_dimension or h > max_dimension:
            ratio = min(max_dimension / w, max_dimension / h)
            new_w, new_h = int(w * ratio), int(h * ratio)
            img = img.resize((new_w, new_h), Image.LANCZOS)
            logger.debug(f"Resized {image_path.name}: {w}x{h} -> {new_w}x{new_h}")
        
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality, optimize=True)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    @staticmethod
    def get_image_info(image_path: Path) -> tuple[float, str]:
        """Return (file_size_kb, 'WxH' resolution)."""
        size_kb = image_path.stat().st_size / 1024
        try:
            img = Image.open(image_path)
            resolution = f"{img.size[0]}x{img.size[1]}"
        except Exception:
            resolution = "unknown"
        return size_kb, resolution
    
    @staticmethod
    def get_b64_size_kb(b64_string: str) -> float:
        return len(b64_string) * 3 / 4 / 1024
    
    @staticmethod
    def get_mime_type(image_path: Path) -> str:
        suffix = image_path.suffix.lower()
        mime_types = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
            ".gif": "image/gif"
        }
        return mime_types.get(suffix, "image/png")
    
    @staticmethod
    def get_presence_check_schema() -> dict:
        return PRESENCE_CHECK_SCHEMA
    
    def _parse_response(self, response_text: str) -> tuple[Optional[bool], Optional[bool], Optional[str], Optional[str], Optional[str]]:
        """
        Parse the MLLM response to extract JSON fields.
        Expected JSON with keys: description_a, description_b, presence_a, presence_b, reasoning.
        """
        obj = self._safe_json_loads(response_text)
        if not isinstance(obj, dict):
            return None, None, None, None, None
        
        presence_a = obj.get("presence_a")
        presence_b = obj.get("presence_b")
        reasoning = obj.get("reasoning")
        description_a = obj.get("description_a")
        description_b = obj.get("description_b")
        
        if not isinstance(presence_a, bool):
            presence_a = None
        if not isinstance(presence_b, bool):
            presence_b = None
        if not isinstance(reasoning, str):
            reasoning = None
        if not isinstance(description_a, str):
            description_a = None
        if not isinstance(description_b, str):
            description_b = None
        
        return presence_a, presence_b, reasoning, description_a, description_b
    
    @staticmethod
    def _safe_json_loads(text: str):
        """
        Best-effort JSON parsing: strips code fences and extracts the first JSON object if needed.
        Returns parsed object or None.
        """
        if not text:
            return None
        
        stripped = text.strip()
        if stripped.startswith("```"):
            # Remove markdown code fences
            lines = stripped.splitlines()
            if len(lines) >= 3 and lines[0].startswith("```") and lines[-1].startswith("```"):
                stripped = "\n".join(lines[1:-1]).strip()
        
        try:
            return json.loads(stripped)
        except Exception:
            pass
        
        # Try to extract a JSON object substring
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = stripped[start : end + 1]
            try:
                return json.loads(candidate)
            except Exception:
                return None
        
        return None
