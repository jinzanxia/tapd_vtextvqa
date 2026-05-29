"""
Parsing module for VideoQA evidence mining pipeline.
"""

from .question_parser import parse_question, QuestionParser

__all__ = [
    "parse_question",
    "QuestionParser",
]
