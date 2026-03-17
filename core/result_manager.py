import csv
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from providers.base import AnalysisResult


@dataclass
class ExportConfig:
    include_descriptions: bool = True
    include_reasoning: bool = True
    include_tokens: bool = False
    include_latency: bool = False
    include_raw_response: bool = False
    include_timestamp: bool = True
    
    def get_columns(self) -> list[str]:
        columns = [
            "pair_id", "mllm", "element",
            "presence_a", "presence_b", "match",
            "ground_truth", "model_classification", "correct",
        ]
        if self.include_descriptions:
            columns.extend(["description_a", "description_b"])
        if self.include_reasoning:
            columns.append("reasoning")
        if self.include_tokens:
            columns.append("tokens_used")
        if self.include_latency:
            columns.append("latency_ms")
        if self.include_raw_response:
            columns.append("raw_response")
        if self.include_timestamp:
            columns.append("timestamp")
        columns.append("error")
        return columns


class ResultManager:
    def __init__(self):
        self.results: list[AnalysisResult] = []
        self.export_config = ExportConfig()
    
    def add_result(self, result: AnalysisResult):
        """Add a single result."""
        self.results.append(result)
    
    def add_results(self, results: list[AnalysisResult]):
        """Add multiple results."""
        self.results.extend(results)
    
    def clear(self):
        """Clear all results."""
        self.results = []
    
    def get_results(self) -> list[AnalysisResult]:
        """Get all results."""
        return self.results
    
    def get_results_by_pair(self, pair_id: str) -> list[AnalysisResult]:
        """Get results for a specific pair."""
        return [r for r in self.results if r.pair_id == pair_id]
    
    def get_results_by_mllm(self, mllm: str) -> list[AnalysisResult]:
        """Get results for a specific MLLM."""
        return [r for r in self.results if r.mllm == mllm]
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to a pandas DataFrame."""
        columns = self.export_config.get_columns()
        data = [r.to_dict(columns) for r in self.results]
        return pd.DataFrame(data, columns=columns)
    
    def export_csv(self, filepath: Path) -> Path:
        """Export results to CSV file."""
        df = self.to_dataframe()
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(filepath, index=False, encoding="utf-8")
        return filepath
    
    def get_summary(self) -> dict:
        """Get a summary of the results."""
        if not self.results:
            return {
                "total": 0,
                "by_mllm": {},
                "matches": 0,
                "mismatches": 0,
                "errors": 0
            }
        
        by_mllm = {}
        matches = 0
        mismatches = 0
        errors = 0

        for result in self.results:
            if result.mllm not in by_mllm:
                by_mllm[result.mllm] = {
                    "total": 0, "matches": 0, "errors": 0,
                    "correct": 0, "incorrect": 0, "evaluated": 0,
                }

            by_mllm[result.mllm]["total"] += 1

            if result.error:
                errors += 1
                by_mllm[result.mllm]["errors"] += 1
            elif result.match is True:
                matches += 1
                by_mllm[result.mllm]["matches"] += 1
            elif result.match is False:
                mismatches += 1

            c = result.correct
            if c is not None:
                by_mllm[result.mllm]["evaluated"] += 1
                if c:
                    by_mllm[result.mllm]["correct"] += 1
                else:
                    by_mllm[result.mllm]["incorrect"] += 1

        for stats in by_mllm.values():
            ev = stats["evaluated"]
            stats["accuracy"] = round(stats["correct"] / ev, 4) if ev > 0 else None

        return {
            "total": len(self.results),
            "by_mllm": by_mllm,
            "matches": matches,
            "mismatches": mismatches,
            "errors": errors
        }
    
    def set_export_config(self, **kwargs):
        """Update export configuration."""
        for key, value in kwargs.items():
            if hasattr(self.export_config, key):
                setattr(self.export_config, key, value)


def generate_export_filename(prefix: str = "results") -> str:
    """Generate a timestamped filename for export."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}.csv"
