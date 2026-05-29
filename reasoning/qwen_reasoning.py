"""
VLM-based reasoning module for final answer generation.

Stage 6 of the hierarchical evidence mining pipeline.
Generates final answer using global context and local OCR evidence.
"""

import logging
from typing import Dict, Any, Optional, List, Union
import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

logger = logging.getLogger(__name__)


class QwenReasoner:
    """Generate answers using Qwen2.5-VL with global and local evidence."""
    
    def __init__(self,
                 model: Optional[Qwen2_5_VLForConditionalGeneration] = None,
                 processor: Optional[AutoProcessor] = None,
                 device: str = "cuda:0"):
        """
        Initialize Qwen reasoner.
        
        Args:
            model: Qwen2.5-VL model instance
            processor: AutoProcessor instance
            device: Device to run on
        """
        self.model = model
        self.processor = processor
        self.device = device
    
    def reason(self,
               question: str,
               global_frame: Optional[Image.Image] = None,
               local_crop: Optional[Image.Image] = None,
               context: str = "") -> str:
        """
        Generate answer using global and/or local evidence.
        
        Args:
            question: The original QA question
            global_frame: Full scene context (optional)
            local_crop: Zoomed-in target crop (optional)
            context: Additional context string (optional)
            
        Returns:
            Generated answer string
        """
        from utils.prompt_builder import (
            build_final_reasoning_prompt,
            build_simple_reasoning_prompt
        )
        
        # Determine which evidence we have
        has_global = global_frame is not None
        has_local = local_crop is not None
        
        if has_global and has_local:
            # Use both global and local evidence
            return self._reason_with_both_evidence(
                question, global_frame, local_crop, context
            )
        elif has_global:
            # Use only global frame
            return self._reason_with_single_image(
                question, global_frame, context="Global frame context"
            )
        elif has_local:
            # Use only local crop
            return self._reason_with_single_image(
                question, local_crop, context="Zoomed-in crop of target region"
            )
        else:
            logger.warning("No evidence provided for reasoning")
            return "Unable to process - no evidence provided."
    
    def _reason_with_both_evidence(self,
                                   question: str,
                                   global_frame: Image.Image,
                                   local_crop: Image.Image,
                                   context: str = "") -> str:
        """
        Generate answer using both global and local evidence.
        
        Args:
            question: The QA question
            global_frame: Full scene image
            local_crop: Target region crop
            context: Additional context
            
        Returns:
            Generated answer
        """
        from utils.prompt_builder import build_final_reasoning_prompt
        
        prompt = build_final_reasoning_prompt(question, context)
        
        try:
            # Build conversation with two images
            conversation = [
                {"role": "system", "content": "You are a helpful assistant for question answering."},
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ]
                },
            ]
            
            text = self.processor.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True
            )

            global_frame = self._ensure_min_vlm_size(global_frame)
            local_crop = self._ensure_min_vlm_size(local_crop)
            
            inputs = self.processor(
                text=[text],
                images=[global_frame, local_crop],
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.model.device)
            
            # Generate response
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=128,
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
            
            # Clean up response
            response = self._clean_response(response)
            
            return response
            
        except Exception as e:
            logger.error(f"Error reasoning with both evidence: {e}")
            return f"Unable to process: {str(e)}"
    
    def _reason_with_single_image(self,
                                  question: str,
                                  image: Image.Image,
                                  context: str = "") -> str:
        """
        Generate answer using a single image.
        
        Args:
            question: The QA question
            image: Single image for reasoning
            context: Context about the image
            
        Returns:
            Generated answer
        """
        from utils.prompt_builder import build_simple_reasoning_prompt
        
        prompt = build_simple_reasoning_prompt(question)
        if context:
            prompt = f"{context}\n\n{prompt}"
        
        try:
            # Build conversation
            conversation = [
                {"role": "system", "content": "You are a helpful assistant for question answering."},
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ]
                },
            ]
            
            text = self.processor.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True
            )

            image = self._ensure_min_vlm_size(image)
            
            inputs = self.processor(
                text=[text],
                images=[image],
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.model.device)
            
            # Generate response
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=128,
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
            
            # Clean up response
            response = self._clean_response(response)
            
            return response
            
        except Exception as e:
            logger.error(f"Error reasoning with single image: {e}")
            return f"Unable to process: {str(e)}"
    
    @staticmethod
    def _ensure_min_vlm_size(image: Image.Image, min_size: int = 336) -> Image.Image:
        """Resize images that are too small for Qwen-VL's patch processor."""
        if image.width >= min_size and image.height >= min_size:
            return image

        scale = max(min_size / max(image.width, 1), min_size / max(image.height, 1))
        new_size = (
            max(min_size, int(round(image.width * scale))),
            max(min_size, int(round(image.height * scale))),
        )
        return image.resize(new_size, Image.Resampling.BICUBIC)

    @staticmethod
    def _clean_response(response: str) -> str:
        """
        Clean up model response.
        
        Args:
            response: Raw model response
            
        Returns:
            Cleaned response
        """
        # Remove common prefixes
        response = response.replace("Answer:", "").strip()
        response = response.replace("Response:", "").strip()
        response = response.replace("The answer is:", "").strip()
        response = response.replace("The answer:", "").strip()
        
        # Remove trailing period if present
        if response.endswith('.'):
            response = response[:-1]
        
        return response.strip()


def run_vlm_reasoning(question: str,
                     global_frame: Optional[Image.Image] = None,
                     local_crop: Optional[Image.Image] = None,
                     model: Optional[Qwen2_5_VLForConditionalGeneration] = None,
                     processor: Optional[AutoProcessor] = None,
                     device: str = "cuda:0",
                     context: str = "") -> str:
    """
    Generate final answer using Qwen2.5-VL with evidence.
    
    Args:
        question: The original QA question
        global_frame: Full scene context (optional)
        local_crop: Zoomed-in target crop (optional)
        model: Optional Qwen model
        processor: Optional processor
        device: Device to run on
        context: Additional context
        
    Returns:
        Generated answer string
    """
    reasoner = QwenReasoner(model=model, processor=processor, device=device)
    return reasoner.reason(question, global_frame, local_crop, context)
