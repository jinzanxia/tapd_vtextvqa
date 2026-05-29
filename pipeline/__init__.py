"""
Pipeline module for VideoQA evidence mining.
"""

from .evidence_pipeline import (
    EvidenceMiningPipeline,
    route_question,
    run_pipeline,
)

__all__ = [
    "EvidenceMiningPipeline",
    "route_question",
    "run_pipeline",
]
