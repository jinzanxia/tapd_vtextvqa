"""
Target Region Localization.

Stage 3 of the hierarchical evidence mining pipeline.
Finds the target object region inside retrieved frames using object-centric grounding.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

logger = logging.getLogger(__name__)


class RegionLocalizer:
    """Localize target regions in frames using VLM grounding capabilities."""
    
    def __init__(self,
                 model: Optional[Qwen2_5_VLForConditionalGeneration] = None,
                 processor: Optional[AutoProcessor] = None,
                 device: str = "cuda:0"):
        """
        Initialize region localizer.
        
        Args:
            model: Qwen2.5-VL model instance
            processor: AutoProcessor instance
            device: Device to run on
        """
        self.model = model
        self.processor = processor
        self.device = device
    
    def localize(self,
                 frames: List[Dict[str, Any]],
                 region_prompt: str) -> List[Dict[str, Any]]:
        """
        Localize target regions in retrieved frames.
        
        Args:
            frames: List of frame dicts with keys: frame_id, frame, score
                    (output from retrieve_relevant_frames)
            region_prompt: Object-centric grounding prompt (e.g., "Locate the blue sign")
            
        Returns:
            List of candidate regions with keys: frame_id, frame, bbox, score
            bbox format: [x1, y1, x2, y2] normalized to [0, 1] or pixel coordinates
        """
        if not frames:
            logger.warning("Empty frames list")
            return []
        
        candidate_regions = []
        
        for frame_dict in frames:
            frame_id = frame_dict["frame_id"]
            frame = frame_dict["frame"]
            frame_score = frame_dict["score"]
            
            # Localize regions in this frame
            bboxes = self._localize_in_frame(frame, region_prompt)
            
            # Create candidate region entries
            for i, bbox in enumerate(bboxes):
                candidate_regions.append({
                    "frame_id": frame_id,
                    "region_id": i,
                    "frame": frame,
                    "bbox": bbox,
                    "frame_score": frame_score,
                    "region_score": bbox.get("confidence", 0.5),
                    "combined_score": frame_score * bbox.get("confidence", 0.5),
                })
        
        # Sort by combined score
        candidate_regions.sort(key=lambda x: x["combined_score"], reverse=True)
        
        logger.info(f"Localized {len(candidate_regions)} candidate regions across {len(frames)} frames")
        
        return candidate_regions
    
    def _localize_in_frame(self,
                          frame: Image.Image,
                          region_prompt: str) -> List[Dict[str, Any]]:
        """
        Localize regions in a single frame.
        
        Args:
            frame: PIL Image
            region_prompt: Localization prompt
            
        Returns:
            List of bboxes with keys: x1, y1, x2, y2, confidence
        """
        from ..utils.prompt_builder import build_region_localization_prompt
        
        try:
            # Build conversation for grounding
            conversation = [
                {"role": "system", "content": "You are a helpful assistant for object localization."},
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": region_prompt},
                    ]
                },
            ]
            
            text = self.processor.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True
            )
            
            inputs = self.processor(
                text=[text],
                images=[frame],
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
            
            # Parse bounding boxes from response
            bboxes = self._parse_bboxes(response, frame.size)
            
            if not bboxes:
                logger.debug(f"No bboxes found in response: {response[:100]}")
                # Fallback: use center crop if grounding fails
                bboxes = [self._get_center_crop_bbox(frame.size)]
            
            return bboxes
            
        except Exception as e:
            logger.error(f"Error localizing region: {e}")
            # Fallback: return center crop
            return [self._get_center_crop_bbox(frame.size if isinstance(frame, Image.Image) else (frame.shape[1], frame.shape[0]))]
    
    @staticmethod
    def _parse_bboxes(response: str, frame_size: Tuple[int, int]) -> List[Dict[str, Any]]:
        """
        Parse bounding boxes from model response.
        
        Args:
            response: Model response text
            frame_size: (width, height) of the frame
            
        Returns:
            List of bbox dicts with keys: x1, y1, x2, y2, confidence
        """
        import json
        import re
        
        bboxes = []
        
        try:
            # Try to extract JSON
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx >= 0 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                data = json.loads(json_str)
                
                if "boxes" in data:
                    for box in data["boxes"]:
                        if isinstance(box, (list, tuple)) and len(box) >= 4:
                            bbox = {
                                "x1": float(box[0]),
                                "y1": float(box[1]),
                                "x2": float(box[2]),
                                "y2": float(box[3]),
                                "confidence": float(box[4]) if len(box) > 4 else 0.8,
                            }
                            bboxes.append(bbox)
                elif "box" in data:
                    box = data["box"]
                    if isinstance(box, (list, tuple)) and len(box) >= 4:
                        bbox = {
                            "x1": float(box[0]),
                            "y1": float(box[1]),
                            "x2": float(box[2]),
                            "y2": float(box[3]),
                            "confidence": 0.8,
                        }
                        bboxes.append(bbox)
        except Exception as e:
            logger.debug(f"Error parsing JSON from response: {e}")
        
        # Try to extract coordinates using regex if JSON parsing failed
        if not bboxes:
            # Pattern: [x1, y1, x2, y2] or (x1, y1, x2, y2)
            pattern = r'\[?\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]?'
            matches = re.findall(pattern, response)
            
            for match in matches:
                bbox = {
                    "x1": float(match[0]),
                    "y1": float(match[1]),
                    "x2": float(match[2]),
                    "y2": float(match[3]),
                    "confidence": 0.6,
                }
                bboxes.append(bbox)
        
        return bboxes
    
    @staticmethod
    def _get_center_crop_bbox(frame_size: Tuple[int, int],
                             crop_ratio: float = 0.5) -> Dict[str, float]:
        """
        Get bounding box for center crop (fallback).
        
        Args:
            frame_size: (width, height)
            crop_ratio: Ratio of crop size to frame size
            
        Returns:
            Bbox dict
        """
        w, h = frame_size
        crop_w = int(w * crop_ratio)
        crop_h = int(h * crop_ratio)
        x1 = (w - crop_w) // 2
        y1 = (h - crop_h) // 2
        x2 = x1 + crop_w
        y2 = y1 + crop_h
        
        return {
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "confidence": 0.5,
        }


def localize_target_regions(frames: List[Dict[str, Any]],
                            region_prompt: str,
                            model: Optional[Qwen2_5_VLForConditionalGeneration] = None,
                            processor: Optional[AutoProcessor] = None,
                            device: str = "cuda:0") -> List[Dict[str, Any]]:
    """
    Localize target regions in retrieved frames.
    
    Args:
        frames: Output from retrieve_relevant_frames
        region_prompt: Object-centric grounding prompt
        model: Optional Qwen model
        processor: Optional processor
        device: Device to run on
        
    Returns:
        List of candidate regions with localization information
    """
    localizer = RegionLocalizer(model=model, processor=processor, device=device)
    return localizer.localize(frames, region_prompt)
