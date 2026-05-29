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

    _shared_ocr_model = None
    _ocr_load_attempted = False
    
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
        
        # PaddleOCR is expensive to initialize; keep one process-local model.
        self.ocr_model = self._get_shared_ocr_model()

    @classmethod
    def _get_shared_ocr_model(cls):
        """Return a shared PaddleOCR instance, loading it at most once."""
        if cls._ocr_load_attempted:
            return cls._shared_ocr_model

        cls._ocr_load_attempted = True
        try:
            from paddleocr import PaddleOCR
            cls._shared_ocr_model = PaddleOCR(use_angle_cls=True, lang='en')
            logger.info("PaddleOCR loaded successfully")
        except ImportError:
            logger.warning("PaddleOCR not available. Will skip OCR confidence scoring.")
        except Exception as e:
            logger.warning(f"Failed to load PaddleOCR. Will skip OCR confidence scoring: {e}")

        return cls._shared_ocr_model
    
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
            raw_x1 = float(bbox["x1"])
            raw_y1 = float(bbox["y1"])
            raw_x2 = float(bbox["x2"])
            raw_y2 = float(bbox["y2"])

            # Some VLMs return normalized boxes. Convert them before rounding.
            if max(raw_x1, raw_y1, raw_x2, raw_y2) <= 1.5:
                raw_x1 *= frame.width
                raw_x2 *= frame.width
                raw_y1 *= frame.height
                raw_y2 *= frame.height

            x1 = max(0, int(raw_x1))
            y1 = max(0, int(raw_y1))
            x2 = min(frame.width, int(raw_x2))
            y2 = min(frame.height, int(raw_y2))
            
            # Validate crop
            if x1 >= x2 or y1 >= y2:
                logger.warning(f"Invalid crop coordinates: ({x1}, {y1}, {x2}, {y2})")
                return None

            x1, y1, x2, y2 = self._expand_bbox_adaptively(
                x1, y1, x2, y2, frame.width, frame.height
            )
            
            # Extract crop
            crop = frame.crop((x1, y1, x2, y2))
            crop = self._ensure_min_vlm_size(crop)
            return crop
            
        except Exception as e:
            logger.error(f"Error extracting crop: {e}")
            return None

    @staticmethod
    def _expand_bbox_adaptively(x1: int,
                                y1: int,
                                x2: int,
                                y2: int,
                                frame_w: int,
                                frame_h: int) -> tuple:
        """Expand bbox with more context for small text regions."""
        box_w = x2 - x1
        box_h = y2 - y1
        bbox_ratio = (box_w * box_h) / max(frame_w * frame_h, 1)

        if bbox_ratio < 0.02:
            expand_ratio = 3.0
        elif bbox_ratio < 0.1:
            expand_ratio = 2.0
        else:
            expand_ratio = 1.3

        target_w = int(round(box_w * expand_ratio))
        target_h = int(round(box_h * expand_ratio))

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        new_x1 = max(0, cx - target_w // 2)
        new_y1 = max(0, cy - target_h // 2)
        new_x2 = min(frame_w, new_x1 + target_w)
        new_y2 = min(frame_h, new_y1 + target_h)

        new_x1 = max(0, new_x2 - target_w)
        new_y1 = max(0, new_y2 - target_h)

        return new_x1, new_y1, new_x2, new_y2

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
            
            # Run OCR. Newer PaddleOCR versions no longer accept cls=... here.
            results = self._run_ocr(crop_np)
            
            if not results or not results[0]:
                logger.debug("No text detected in crop")
                return 0.0
            
            confidences = self._extract_ocr_confidences(results)
            avg_confidence = np.mean(confidences) if confidences else 0.0
            
            return float(avg_confidence)
            
        except Exception as e:
            logger.error(f"Error computing OCR confidence: {e}")
            return 0.5

    def _run_ocr(self, crop_np: np.ndarray):
        """Run PaddleOCR across old and new PaddleOCR APIs."""
        if hasattr(self.ocr_model, "ocr"):
            try:
                return self.ocr_model.ocr(crop_np)
            except TypeError as e:
                logger.debug(f"PaddleOCR.ocr failed, trying predict(): {e}")

        if hasattr(self.ocr_model, "predict"):
            return self.ocr_model.predict(crop_np)

        raise AttributeError("PaddleOCR model has neither ocr() nor predict()")

    @classmethod
    def _extract_ocr_confidences(cls, results: Any) -> List[float]:
        """Extract text recognition confidences from PaddleOCR result variants."""
        confidences = []
        cls._collect_confidences(results, confidences)
        return confidences

    @classmethod
    def _collect_confidences(
        cls,
        value: Any,
        confidences: List[float],
        allow_numeric: bool = False,
    ) -> None:
        if value is None:
            return

        if isinstance(value, dict):
            for key in ("score", "confidence", "rec_score", "rec_scores"):
                if key in value:
                    cls._collect_confidences(value[key], confidences, allow_numeric=True)
            for key in ("res", "data", "result", "ocr_result", "rec_texts"):
                if key in value:
                    cls._collect_confidences(value[key], confidences)
            return

        if isinstance(value, (list, tuple)):
            if cls._looks_like_legacy_text_info(value):
                confidences.append(float(value[1]))
                return
            for item in value:
                cls._collect_confidences(item, confidences, allow_numeric=allow_numeric)
            return

        if (
            allow_numeric
            and isinstance(value, (int, float, np.floating))
            and 0.0 <= float(value) <= 1.0
        ):
            confidences.append(float(value))

    @staticmethod
    def _looks_like_legacy_text_info(value: Any) -> bool:
        """Match PaddleOCR legacy tuples like ('text', 0.98)."""
        return (
            isinstance(value, (list, tuple))
            and len(value) >= 2
            and isinstance(value[0], str)
            and isinstance(value[1], (int, float, np.floating))
            and 0.0 <= float(value[1]) <= 1.0
        )
    
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
            crop = self._ensure_min_vlm_size(crop)
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
