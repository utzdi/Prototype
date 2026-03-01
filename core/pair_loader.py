from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import re


SUPPORTED_FORMATS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}


@dataclass
class ScreenshotPair:
    pair_id: str
    reference_path: Path
    comparison_path: Path
    folder_path: Path
    
    def is_valid(self) -> bool:
        return (
            self.reference_path.exists() and 
            self.comparison_path.exists() and
            self.reference_path.suffix.lower() in SUPPORTED_FORMATS and
            self.comparison_path.suffix.lower() in SUPPORTED_FORMATS
        )


class PairLoader:
    def __init__(
        self,
        reference_pattern: str = "reference",
        comparison_pattern: str = "comparison"
    ):
        self.reference_pattern = reference_pattern.lower()
        self.comparison_pattern = comparison_pattern.lower()
    
    def scan_folder(self, folder_path: Path) -> list[ScreenshotPair]:
        """
        Scan a folder for screenshot pairs.
        Expects structure like:
            folder/
                paar_001/
                    reference.png
                    comparison.png
                paar_002/
                    ...
        """
        pairs = []
        folder_path = Path(folder_path)
        
        if not folder_path.exists():
            return pairs
        
        for subfolder in sorted(folder_path.iterdir()):
            if not subfolder.is_dir():
                continue
            
            pair = self._find_pair_in_folder(subfolder)
            if pair:
                pairs.append(pair)
        
        return pairs
    
    def _find_pair_in_folder(self, folder: Path) -> Optional[ScreenshotPair]:
        """Find a screenshot pair within a single folder."""
        reference_path = None
        comparison_path = None
        
        for file in folder.iterdir():
            if not file.is_file():
                continue
            if file.suffix.lower() not in SUPPORTED_FORMATS:
                continue
            
            filename_lower = file.stem.lower()
            
            if self._matches_pattern(filename_lower, self.reference_pattern):
                reference_path = file
            elif self._matches_pattern(filename_lower, self.comparison_pattern):
                comparison_path = file
        
        if reference_path and comparison_path:
            return ScreenshotPair(
                pair_id=folder.name,
                reference_path=reference_path,
                comparison_path=comparison_path,
                folder_path=folder
            )
        
        return None
    
    def _matches_pattern(self, filename: str, pattern: str) -> bool:
        """Check if filename matches the pattern (contains or equals)."""
        return pattern in filename or filename == pattern
    
    def set_patterns(self, reference: str, comparison: str):
        """Update the naming patterns."""
        self.reference_pattern = reference.lower()
        self.comparison_pattern = comparison.lower()


def load_pairs(
    folder_path: Path,
    reference_pattern: str = "reference",
    comparison_pattern: str = "comparison"
) -> list[ScreenshotPair]:
    """Convenience function to load pairs from a folder."""
    loader = PairLoader(reference_pattern, comparison_pattern)
    return loader.scan_folder(folder_path)
