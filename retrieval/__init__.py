"""
Retrieval module for VideoQA evidence mining pipeline.
"""

from .frame_retrieval import retrieve_relevant_frames, FrameRetriever
from .region_localization import localize_target_regions, RegionLocalizer
from .ocr_visibility import score_crop_visibility, OCRVisibilityScorer

__all__ = [
    "retrieve_relevant_frames",
    "FrameRetriever",
    "localize_target_regions",
    "RegionLocalizer",
    "score_crop_visibility",
    "OCRVisibilityScorer",
]
