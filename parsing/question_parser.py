"""
Question Structural Parser using Qwen LLM.

Stage 1 of the hierarchical evidence mining pipeline.
Converts original QA question into structured representation for retrieval and localization.
"""

import json
import logging
from typing import Dict, Any, Optional
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

logger = logging.getLogger(__name__)


class QuestionParser:
    """Parse questions into structured representations."""
    
    def __init__(self, 
                 model: Optional[Qwen2_5_VLForConditionalGeneration] = None,
                 processor: Optional[AutoProcessor] = None,
                 device: str = "cuda:0"):
        """
        Initialize question parser.
        
        Args:
            model: Qwen2.5-VL model instance. If None, will load from default path.
            processor: AutoProcessor instance. If None, will load from default path.
            device: Device to run model on
        """
        self.model = model
        self.processor = processor
        self.device = device
        
        # Load model and processor if not provided
        if self.model is None or self.processor is None:
            self._load_model()
    
    def _load_model(self):
        """Load Qwen2.5-VL model and processor."""
        try:
            model_path = "Qwen/Qwen2.5-VL-7B-Instruct"
            logger.info(f"Loading model from {model_path}")
            
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                device_map=self.device,
                torch_dtype=torch.bfloat16,
                attn_implementation="sdpa",
            )
            self.processor = AutoProcessor.from_pretrained(model_path)
            logger.info("Model and processor loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def parse(self, question: str) -> Dict[str, str]:
        """
        Parse question into structured representation.
        
        Args:
            question: Original QA question
            
        Returns:
            Dict with keys: target, attribute, relation, task
            
        Example:
            Input: "What does the sign in blue on top of the road say?"
            Output: {
                "target": "sign",
                "attribute": "blue",
                "relation": "above road",
                "task": "ocr"
            }
        """
        from ..utils.prompt_builder import build_question_parsing_prompt
        
        prompt = build_question_parsing_prompt(question)
        
        try:
            # Prepare conversation
            conversation = [
                {"role": "system", "content": "You are a helpful assistant for understanding questions."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                    ]
                },
            ]
            
            # Apply chat template
            text = self.processor.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Prepare inputs
            inputs = self.processor(
                text=[text],
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.model.device)
            
            # Generate response
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                    temperature=0,
                    num_beams=1,
                )
            
            # Decode response
            generated_ids_trimmed = [
                out_ids[len(in_ids):] 
                for in_ids, out_ids in zip(inputs.input_ids, output_ids)
            ]
            response = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0].strip()
            
            # Parse JSON response
            parsed = self._extract_json(response)
            
            # Validate and fill defaults
            result = self._validate_parsed_question(parsed)
            
            logger.info(f"Parsed question: {question}")
            logger.info(f"Result: {result}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error parsing question: {e}")
            # Return default structure on error
            return {
                "target": "object",
                "attribute": "",
                "relation": "",
                "task": "general"
            }
    
    @staticmethod
    def _extract_json(text: str) -> Dict[str, Any]:
        """
        Extract JSON from model response.
        
        Args:
            text: Model response text
            
        Returns:
            Parsed JSON dictionary
        """
        try:
            # Try to find JSON in the text
            start_idx = text.find('{')
            end_idx = text.rfind('}') + 1
            
            if start_idx >= 0 and end_idx > start_idx:
                json_str = text[start_idx:end_idx]
                return json.loads(json_str)
            else:
                logger.warning(f"No JSON found in response: {text}")
                return {}
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            return {}
    
    @staticmethod
    def _validate_parsed_question(parsed: Dict[str, Any]) -> Dict[str, str]:
        """
        Validate and fill defaults for parsed question.
        
        Args:
            parsed: Parsed question dictionary
            
        Returns:
            Validated dictionary with all required keys
        """
        # Define defaults
        defaults = {
            "target": "object",
            "attribute": "",
            "relation": "",
            "task": "general"
        }
        
        # Fill in missing keys with defaults
        result = {**defaults}
        for key in defaults.keys():
            if key in parsed and isinstance(parsed[key], str):
                result[key] = parsed[key].lower().strip()
        
        return result


def parse_question(question: str,
                   model: Optional[Qwen2_5_VLForConditionalGeneration] = None,
                   processor: Optional[AutoProcessor] = None,
                   device: str = "cuda:0") -> Dict[str, str]:
    """
    Convenience function to parse a question.
    
    Args:
        question: Question to parse
        model: Optional Qwen model
        processor: Optional processor
        device: Device to run on
        
    Returns:
        Parsed question structure
    """
    parser = QuestionParser(model=model, processor=processor, device=device)
    return parser.parse(question)
