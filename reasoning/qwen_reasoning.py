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
               global_frame: Union[Image.Image, List[Image.Image], None] = None,
               local_crop: Union[Image.Image, List[Image.Image], None] = None,
               context: str = "") -> str:
        """
        Generate answer using global and/or local evidence.
        
        Args:
            question: The original QA question
            global_frame: Full scene context image(s) (optional)
            local_crop: Zoomed-in target crop image(s) (optional)
            context: Additional context string (optional)
            
        Returns:
            Generated answer string
        """
        from utils.prompt_builder import (
            build_final_reasoning_prompt,
            build_simple_reasoning_prompt
        )
        
        global_frames = self._as_image_list(global_frame)
        local_crops = self._as_image_list(local_crop)

        # Determine which evidence we have
        has_global = bool(global_frames)
        has_local = bool(local_crops)
        
        if has_global and has_local:
            return self._reason_with_multiple_images(
                question,
                global_frames,
                local_crops,
                context,
            )
        elif has_global:
            return self._reason_with_multiple_images(
                question,
                global_frames,
                [],
                context,
            )
        elif has_local:
            return self._reason_with_multiple_images(
                question,
                [],
                local_crops,
                context or "Zoomed-in crop of target region",
            )
        else:
            logger.warning("No evidence provided for reasoning")
            return "Unable to process - no evidence provided."

    def _reason_with_multiple_images(self,
                                     question: str,
                                     global_frames: List[Image.Image],
                                     local_crops: List[Image.Image],
                                     context: str = "") -> str:
        """
        Generate answer using any number of global frames and local crops.

        Images are sent to Qwen in this order: global frame 1, local crop 1,
        global frame 2, local crop 2, and so on.
        """
        all_images = self._interleave_evidence_images(global_frames, local_crops)
        if not all_images:
            logger.warning("No images provided for multi-image reasoning")
            return "Unable to process - no evidence provided."

        try:
            if global_frames and not local_crops and not context:
                prompt = (
                    "Please provide a brief answer based on the sampled video frames, "
                    "using as few words as possible. Question: " + question
                )
            else:
                evidence_note = self._build_evidence_note(len(global_frames), len(local_crops), context)
                prompt = (
                    f"{evidence_note}\n\n"
                    "Please provide a brief answer based on the sampled video frames/images, "
                    f"using as few words as possible. Question: {question}"
                )

            conversation = [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": [{"type": "image"} for _ in all_images] + [
                        {"type": "text", "text": prompt},
                    ],
                },
            ]

            text = self.processor.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True,
            )

            images = all_images

            inputs = self.processor(
                text=[text],
                images=images,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.model.device)

            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False,
                    temperature=0,
                    num_beams=1,
                )

            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, output_ids)
            ]
            return self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()

        except Exception as e:
            logger.error(f"Error reasoning with multiple images: {e}")
            return f"Unable to process: {str(e)}"

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
        return self._reason_with_multiple_images(question, [global_frame], [local_crop], context)

    def _reason_with_both_evidence_bak(self,
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
            #response = self._clean_response(response)
            
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
    def _as_image_list(images: Union[Image.Image, List[Image.Image], None]) -> List[Image.Image]:
        """Normalize optional single/list image inputs into a clean list."""
        if images is None:
            return []
        if isinstance(images, Image.Image):
            return [images]
        return [image for image in images if isinstance(image, Image.Image)]

    @staticmethod
    def _interleave_evidence_images(global_frames: List[Image.Image],
                                    local_crops: List[Image.Image]) -> List[Image.Image]:
        """Order evidence as G1, L1, G2, L2, appending any unpaired extras at the end."""
        images = []
        max_count = max(len(global_frames), len(local_crops))
        for idx in range(max_count):
            if idx < len(global_frames):
                images.append(global_frames[idx])
            if idx < len(local_crops):
                images.append(local_crops[idx])
        return images

    @staticmethod
    def _build_evidence_note(global_count: int, local_count: int, context: str = "") -> str:
        """Describe image ordering for multi-image prompts."""
        parts = []
        if global_count and local_count:
            paired_count = min(global_count, local_count)
            parts.append(
                f"Images are interleaved as global frame then its corresponding local crop for "
                f"{paired_count} pair(s)."
            )
            if global_count > local_count:
                parts.append("Any remaining images after the pairs are additional global frames.")
            elif local_count > global_count:
                parts.append("Any remaining images after the pairs are additional local crops.")
        elif global_count:
            parts.append(f"The {global_count} image(s) are full video frames in relevance order.")
        elif local_count:
            parts.append(f"The {local_count} image(s) are zoomed-in crops of candidate target regions in score order.")
        if context:
            parts.append(context)
        return " ".join(parts)

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
                     global_frame: Union[Image.Image, List[Image.Image], None] = None,
                     local_crop: Union[Image.Image, List[Image.Image], None] = None,
                     model: Optional[Qwen2_5_VLForConditionalGeneration] = None,
                     processor: Optional[AutoProcessor] = None,
                     device: str = "cuda:0",
                     context: str = "") -> str:
    """
    Generate final answer using Qwen2.5-VL with evidence.
    
    Args:
        question: The original QA question
        global_frame: Full scene context image(s) (optional)
        local_crop: Zoomed-in target crop image(s) (optional)
        model: Optional Qwen model
        processor: Optional processor
        device: Device to run on
        context: Additional context
        
    Returns:
        Generated answer string
    """
    reasoner = QwenReasoner(model=model, processor=processor, device=device)
    return reasoner.reason(question, global_frame, local_crop, context)
