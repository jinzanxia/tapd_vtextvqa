"""
Prompt builder utilities for constructing structured prompts for evidence mining.
"""
from typing import Dict


def build_question_parsing_prompt(question: str) -> str:
    """Build prompt for question structural parsing."""
    return f"""Extract structured information from the following question:

1. target object: What is the main object being asked about?
2. attribute: What descriptive attributes does it have? (e.g., color, size)
3. spatial relations: Where is it located relative to other objects?
4. task type: What type of task is this? (e.g., ocr, counting, detection, spatial_reasoning)

Question: {question}

Return ONLY valid JSON with these exact keys: target, attribute, relation, task
Example format:
{{"target": "sign", "attribute": "blue", "relation": "above road", "task": "ocr"}}

JSON:"""


def build_frame_retrieval_prompt(parsed_question: Dict[str, str]) -> str:
    """Build retrieval-oriented prompt for frame relevance scoring."""
    target = parsed_question.get("target", "object")
    attribute = parsed_question.get("attribute", "")
    relation = parsed_question.get("relation", "")

    if attribute and relation:
        return f"Does this frame contain a {attribute} {target} {relation}?"
    if attribute:
        return f"Does this frame contain a {attribute} {target}?"
    if relation:
        return f"Does this frame contain a {target} {relation}?"
    return f"Does this frame contain a {target}?"


def build_frame_relevance_scoring_prompt(retrieval_prompt: str) -> str:
    """Build prompt for frame relevance scoring."""
    return f"""Given the image/frame, answer the following question with a confidence score:

Question: {retrieval_prompt}

Answer with JSON format:
{{"answer": "yes" or "no", "confidence": <float 0-1>}}

Example:
{{"answer": "yes", "confidence": 0.95}}

JSON:"""


def build_region_localization_prompt(parsed_question: Dict[str, str]) -> str:
    """Build prompt for target region localization."""
    target = parsed_question.get("target", "object")
    attribute = parsed_question.get("attribute", "")
    target_description = f"{attribute} {target}" if attribute else target

    return f"""Locate the {target_description} in this image.

Return ONLY valid JSON with one bounding box in pixel coordinates:
{{"box": [x1, y1, x2, y2], "confidence": <float 0-1>}}

If the target is text or appears in a row/table, include nearby text or numbers needed to answer the question.

JSON:"""


def build_raw_question_localization_prompt(question: str) -> str:
    """Build localization prompt directly from the original question."""
    return f"""Locate the visual evidence region needed to answer this question:

Question: {question}

Return ONLY valid JSON with one bounding box in pixel coordinates:
{{"box": [x1, y1, x2, y2], "confidence": <float 0-1>}}

Include the object, text, number, sign, label, or nearby context that is most likely needed to answer.

JSON:"""


def build_ocr_focused_localization_prompt(question: str) -> str:
    """Build localization prompt focused on readable text/number evidence."""
    return f"""Locate the clearest readable text or number region that could answer this question:

Question: {question}

Prefer signs, labels, logos, captions, printed words, prices, IDs, scores, or numbers. Include nearby context if it helps identify the correct text.

Return ONLY valid JSON with one bounding box in pixel coordinates:
{{"box": [x1, y1, x2, y2], "confidence": <float 0-1>}}

JSON:"""


def build_ocr_visibility_prompt(parsed_question: Dict[str, str]) -> str:
    """Build prompt for OCR readability scoring."""
    target = parsed_question.get("target", "target")

    return f"""Assess the readability of text on the {target} in this image:

Rate the text readability on a scale of 0-1, considering:
- Sharpness and focus
- Contrast and visibility
- Angle and distortion
- Occlusion

Return JSON:
{{"readability": <float 0-1>, "confidence": <float 0-1>}}

JSON:"""


def build_crop_localization_scoring_prompt(parsed_question: Dict[str, str]) -> str:
    """Build prompt for verifying whether a candidate crop contains the target."""
    target = parsed_question.get("target", "target")
    attribute = parsed_question.get("attribute", "")
    target_description = f"{attribute} {target}" if attribute else target

    return f"""Does this cropped region contain the {target_description}?

Return ONLY valid JSON:
{{"answer": "yes" or "no", "confidence": <float 0-1>}}

JSON:"""


def build_final_reasoning_prompt(original_question: str, context: str = "") -> str:
    """Build final reasoning prompt that combines global and local context."""
    if context:
        return f"""{context}

Image 1: Full scene context (global view)
Image 2: Zoomed-in crop of the relevant target region (local evidence)

Please answer the following question based on both images:

{original_question}

Provide a concise, accurate answer."""

    return f"""Image 1: Full scene context (global view)
Image 2: Zoomed-in crop of the relevant target region (local evidence)

Please answer the following question based on both images:

{original_question}

Provide a concise, accurate answer."""


def build_simple_reasoning_prompt(question: str) -> str:
    """Build simple reasoning prompt for single image."""
    return f"""Please answer the following question based on the image:

{question}

Provide a concise, accurate answer."""
