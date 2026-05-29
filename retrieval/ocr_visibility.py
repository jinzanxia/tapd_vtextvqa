"""
OCR Visibility Scoring and Evidence Crop Selection.

Stage 4 of the hierarchical evidence mining pipeline.
Among multiple candidate regions, selects the crop with the clearest readable text.
Uses OCR confidence, sharpness, and VLM visibility scoring.
"""

import logging
from typing import List, Dict, Any, Optional
import cv2
import numpy as np
from PIL import Image
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

logger = logging.getLogger(__name__)


class OCRVisibilityScorer:
    """Score OCR visibility and readability of candidate crops."""
    
    def __init__(self,
                 model: Optional[Qwen2_5_VLForConditionalGeneration] = None,
                 processor: Optional[AutoProcessor] = None,
                 device: str = "cuda:0",
                 alpha: float = 0.4,
                 beta: float = 0.3,
                 gamma: float = 0.3):
        """
        Initialize OCR visibility scorer.
        
        Args:
            model: Qwen2.5-VL model instance
            processor: AutoProcessor instance
            device: Device to run on
            alpha: Weight for OCR confidence (PaddleOCR)
            beta: Weight for sharpness (Laplacian variance)
            gamma: Weight for VLM visibility score
        """
        self.model = model
        self.processor = processor
        self.device = device
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        # Try to import PaddleOCR for OCR confidence scoring
        self.ocr_model = None
        try:
            from paddleocr import PaddleOCR
            self.ocr_model = PaddleOCR(use_angle_cls=True, lang='en')
            logger.info("PaddleOCR loaded successfully")
        except ImportError:
            logger.warning("PaddleOCR not available. Will skip OCR confidence scoring.")
    
    def score_crops(self,
                    candidate_regions: List[Dict[str, Any]],
                    ocr_prompt: str) -> Dict[str, Any]:
        """
        Score and select best OCR crop from candidate regions.
        
        Args:
            candidate_regions: List of candidate region dicts from localize_target_regions
                              Each should have: frame, bbox, frame_score, region_score
            ocr_prompt: OCR readability assessment prompt
            
        Returns:
            Dict with best crop and scoring details
        """
        if not candidate_regions:
            logger.warning("No candidate regions provided")
            return {
                "success": False,
                "message": "No candidate regions",
                "best_crop": None,
                "best_region": None,
            }
        
        # Extract crops from regions
        crops_data = []
        for region in candidate_regions:
            crop = self._extract_crop(region["frame"], region["bbox"])
            if crop is not None:
                crops_data.append({
                    "region": region,
                    "crop": crop,
                })
        
        if not crops_data:
            logger.warning("Failed to extract any crops")
            return {
                "success": False,
                "message": "Failed to extract crops",
                "best_crop": None,
                "best_region": None,
            }
        
        # Score each crop
        scores = []
        for crop_data in crops_data:
            crop = crop_data["crop"]
            region = crop_data["region"]
            
            # Compute individual scores
            ocr_conf = self._compute_ocr_confidence(crop)
            sharpness = self._compute_sharpness(crop)
            vlm_score = self._compute_vlm_visibility(crop, ocr_prompt)
            
            # Normalize scores to [0, 1]
            ocr_conf_norm = min(1.0, ocr_conf)
            sharpness_norm = min(1.0, sharpness / 100.0)  # Normalize Laplacian variance
            vlm_score_norm = vlm_score
            
            # Combine scores
            combined_score = (
                self.alpha * ocr_conf_norm +
                self.beta * sharpness_norm +
                self.gamma * vlm_score_norm
            )
            
            scores.append({
                "region": region,
                "crop": crop,
                "ocr_confidence": ocr_conf_norm,
                "sharpness": sharpness_norm,
                "vlm_visibility": vlm_score_norm,
                "combined_score": combined_score,
            })
        
        # Select best crop
        best_score_dict = max(scores, key=lambda x: x["combined_score"])
        
        logger.info(f"Selected best crop from {len(scores)} candidates")
        logger.info(f"Score breakdown: OCR={best_score_dict['ocr_confidence']:.3f}, "
                   f"Sharpness={best_score_dict['sharpness']:.3f}, "
                   f"VLM={best_score_dict['vlm_visibility']:.3f}, "
                   f"Combined={best_score_dict['combined_score']:.3f}")
        
        return {
            "success": True,
            "best_crop": best_score_dict["crop"],
            "best_region": best_score_dict["region"],
            "scores": scores,
            "best_scores": {
                "ocr_confidence": best_score_dict["ocr_confidence"],
                "sharpness": best_score_dict["sharpness"],
                "vlm_visibility": best_score_dict["vlm_visibility"],
                "combined_score": best_score_dict["combined_score"],
            },
        }
    
    def _extract_crop(self, frame: Image.Image, bbox: Dict[str, float]) -> Optional[Image.Image]:
        """
        Extract crop from frame given bounding box.
        
        Args:
            frame: PIL Image
            bbox: Dict with keys x1, y1, x2, y2
            
        Returns:
            Cropped PIL Image or None if extraction fails
        """
        try:
            # Ensure bbox coordinates are integers
            x1 = max(0, int(bbox["x1"]))
            y1 = max(0, int(bbox["y1"]))
            x2 = min(frame.width, int(bbox["x2"]))
            y2 = min(frame.height, int(bbox["y2"]))
            
            # Validate crop
            if x1 >= x2 or y1 >= y2:
                logger.warning(f"Invalid crop coordinates: ({x1}, {y1}, {x2}, {y2})")
                return None
            
            # Extract crop
            crop = frame.crop((x1, y1, x2, y2))
            return crop
            
        except Exception as e:
            logger.error(f"Error extracting crop: {e}")
            return None
    
    def _compute_ocr_confidence(self, crop: Image.Image) -> float:
        """
        Compute OCR confidence using PaddleOCR.
        
        Args:
            crop: Cropped PIL Image
            
        Returns:
            Average OCR confidence in [0, 1]
        """
        if self.ocr_model is None:
            logger.debug("PaddleOCR not available, using default OCR confidence")
            return 0.5
        
        try:
            # Convert PIL to numpy
            crop_np = np.array(crop)
            
            # Run OCR
            results = self.ocr_model.ocr(crop_np, cls=True)
            
            if not results or not results[0]:
                logger.debug("No text detected in crop")
                return 0.0
            
            # Compute average confidence
            confidences = [box[-1] for box in results[0]]
            avg_confidence = np.mean(confidences) if confidences else 0.0
            
            return float(avg_confidence)
            
        except Exception as e:
            logger.error(f"Error computing OCR confidence: {e}")
            return 0.5
    
    @staticmethod
    def _compute_sharpness(crop: Image.Image) -> float:
        """
        Compute sharpness using Laplacian variance.
        
        Args:
            crop: Cropped PIL Image
            
        Returns:
            Laplacian variance (higher = sharper)
        """
        try:
            # Convert to numpy and grayscale
            crop_np = np.array(crop)
            if len(crop_np.shape) == 3:
                crop_gray = cv2.cvtColor(crop_np, cv2.COLOR_RGB2GRAY)
            else:
                crop_gray = crop_np
            
            # Compute Laplacian variance
            laplacian = cv2.Laplacian(crop_gray, cv2.CV_64F)
            variance = laplacian.var()
            
            return float(variance)
            
        except Exception as e:
            logger.error(f"Error computing sharpness: {e}")
            return 0.0
    
    def _compute_vlm_visibility(self, crop: Image.Image, ocr_prompt: str) -> float:
        """
        Compute VLM-based OCR visibility score.
        
        Args:
            crop: Cropped PIL Image
            ocr_prompt: OCR visibility assessment prompt
            
        Returns:
            Visibility score in [0, 1]
        """
        from utils.prompt_builder import build_ocr_visibility_prompt
        
        try:
            visibility_prompt = build_ocr_visibility_prompt({"target": "text"})
            
            # Build conversation
            conversation = [
                {"role": "system", "content": "You are an expert in OCR text visibility assessment."},
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": visibility_prompt},
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
                images=[crop],
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
            
            # Extract score
            return self._extract_visibility_score(response)
            
        except Exception as e:
            logger.error(f"Error computing VLM visibility: {e}")
            return 0.5
    
    @staticmethod
    def _extract_visibility_score(response: str) -> float:
        """Extract visibility score from VLM response."""
        import json
        
        try:
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx >= 0 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                data = json.loads(json_str)
                
                if "readability" in data:
                    return float(data["readability"])
                elif "visibility" in data:
                    return float(data["visibility"])
                elif "score" in data:
                    return float(data["score"])
            
            # Fallback
            if "high" in response.lower():
                return 0.8
            elif "medium" in response.lower():
                return 0.5
            elif "low" in response.lower():
                return 0.2
            else:
                return 0.5
        except Exception as e:
            logger.debug(f"Error extracting visibility score: {e}")
            return 0.5


def score_crop_visibility(candidate_regions: List[Dict[str, Any]],
                         ocr_prompt: str,
                         model: Optional[Qwen2_5_VLForConditionalGeneration] = None,
                         processor: Optional[AutoProcessor] = None,
                         device: str = "cuda:0",
                         alpha: float = 0.4,
                         beta: float = 0.3,
                         gamma: float = 0.3) -> Dict[str, Any]:
    """
    Score crop visibility and select best OCR crop.
    
    Args:
        candidate_regions: Output from localize_target_regions
        ocr_prompt: OCR visibility assessment prompt
        model: Optional Qwen model
        processor: Optional processor
        device: Device to run on
        alpha: Weight for OCR confidence
        beta: Weight for sharpness
        gamma: Weight for VLM visibility
        
    Returns:
        Dict with best_crop and scoring details
    """
    scorer = OCRVisibilityScorer(
        model=model,
        processor=processor,
        device=device,
        alpha=alpha,
        beta=beta,
        gamma=gamma
    )
    return scorer.score_crops(candidate_regions, ocr_prompt)
