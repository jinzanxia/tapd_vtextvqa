"""
Pipeline module for VideoQA evidence mining.
"""

from .evidence_pipeline import (
    EvidenceMiningPipeline,
    run_pipeline,
)

__all__ = [
    "EvidenceMiningPipeline",
    "run_pipeline",
]
