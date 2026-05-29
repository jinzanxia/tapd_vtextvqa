"""
Prompt builder utilities for constructing structured prompts for various stages.
"""
from typing import Dict, Any


def build_question_parsing_prompt(question: str) -> str:
    """
    Build prompt for question structural parsing.
    
    Args:
        question: The original QA question
        
    Returns:
        Formatted prompt string for Qwen to parse question structure
    """
    prompt = f"""Extract structured information from the following question:

1. **target object**: What is the main object being asked about?
2. **attribute**: What descriptive attributes does it have? (e.g., color, size)
3. **spatial relations**: Where is it located relative to other objects?
4. **task type**: What type of task is this? (e.g., ocr, counting, detection, spatial_reasoning)

Question: {question}

Return ONLY valid JSON with these exact keys: target, attribute, relation, task
Example format:
{{"target": "sign", "attribute": "blue", "relation": "above road", "task": "ocr"}}

JSON:"""
    return prompt


def build_frame_retrieval_prompt(parsed_question: Dict[str, str]) -> str:
    """
    Build retrieval-oriented prompt for frame relevance scoring.
    
    Args:
        parsed_question: Structured question with keys: target, attribute, relation, task
        
    Returns:
        Retrieval prompt focusing on object existence and spatial relation
    """
    target = parsed_question.get("target", "object")
    attribute = parsed_question.get("attribute", "")
    relation = parsed_question.get("relation", "")
    
    # Build natural retrieval prompt
    if attribute and relation:
        retrieval_prompt = f"Does this frame contain a {attribute} {target} {relation}?"
    elif attribute:
        retrieval_prompt = f"Does this frame contain a {attribute} {target}?"
    elif relation:
        retrieval_prompt = f"Does this frame contain a {target} {relation}?"
    else:
        retrieval_prompt = f"Does this frame contain a {target}?"
    
    return retrieval_prompt


def build_frame_relevance_scoring_prompt(retrieval_prompt: str) -> str:
    """
    Build prompt for Qwen to score frame relevance.
    
    Args:
        retrieval_prompt: The retrieval question
        
    Returns:
        Prompt for relevance scoring
    """
    prompt = f"""Given the image/frame, answer the following question with a confidence score:

Question: {retrieval_prompt}

Answer with JSON format:
{{"answer": "yes" or "no", "confidence": <float 0-1>}}

Example:
{{"answer": "yes", "confidence": 0.95}}

JSON:"""
    return prompt


def build_region_localization_prompt(parsed_question: Dict[str, str]) -> str:
    """
    Build prompt for target region localization.
    
    Args:
        parsed_question: Structured question with keys: target, attribute, relation, task
        
    Returns:
        Object-centric grounding prompt
    """
    target = parsed_question.get("target", "object")
    attribute = parsed_question.get("attribute", "")
    
    # Build localization prompt
    if attribute:
        localization_prompt = f"Locate the {attribute} {target}."
    else:
        localization_prompt = f"Locate the {target}."
    
    return localization_prompt


def build_ocr_visibility_prompt(parsed_question: Dict[str, str]) -> str:
    """
    Build prompt for OCR visibility scoring.
    
    Args:
        parsed_question: Structured question with keys: target, attribute, relation, task
        
    Returns:
        OCR readability assessment prompt
    """
    target = parsed_question.get("target", "target")
    
    prompt = f"""Assess the readability of text on the {target} in this image:

Rate the text readability on a scale of 0-1, considering:
- Sharpness and focus
- Contrast and visibility
- Angle and distortion
- Occlusion

Return JSON:
{{"readability": <float 0-1>, "confidence": <float 0-1>}}

JSON:"""
    return prompt


def build_crop_localization_scoring_prompt(parsed_question: Dict[str, str]) -> str:
    """
    Build prompt for verifying whether a candidate crop contains the target.

    Args:
        parsed_question: Structured question with keys: target, attribute, relation, task

    Returns:
        Prompt that asks for yes/no plus confidence.
    """
    target = parsed_question.get("target", "target")
    attribute = parsed_question.get("attribute", "")

    if attribute:
        target_description = f"{attribute} {target}"
    else:
        target_description = target

    prompt = f"""Does this cropped region contain the {target_description}?

Return ONLY valid JSON:
{{"answer": "yes" or "no", "confidence": <float 0-1>}}

JSON:"""
    return prompt


def build_final_reasoning_prompt(original_question: str, context: str = "") -> str:
    """
    Build final reasoning prompt that combines global and local context.
    
    Args:
        original_question: The original QA question
        context: Optional context about the images
        
    Returns:
        Final reasoning prompt for Qwen2.5-VL
    """
    if context:
        prompt = f"""{context}

Image 1: Full scene context (global view)
Image 2: Zoomed-in crop of the relevant target region (local evidence)

Please answer the following question based on both images:

{original_question}

Provide a concise, accurate answer."""
    else:
        prompt = f"""Image 1: Full scene context (global view)
Image 2: Zoomed-in crop of the relevant target region (local evidence)

Please answer the following question based on both images:

{original_question}

Provide a concise, accurate answer."""
    
    return prompt


def build_simple_reasoning_prompt(question: str) -> str:
    """
    Build simple reasoning prompt for single image.
    
    Args:
        question: The QA question
        
    Returns:
        Simple reasoning prompt
    """
    prompt = f"""Please answer the following question based on the image:

{question}

Provide a concise, accurate answer."""
    return prompt
