"""
Main Evidence Mining Pipeline Orchestrator.

Coordinates all stages of the hierarchical evidence mining framework:
1. Question Structural Parsing
2. Frame-Level Relevant Frame Retrieval
3. Region Localization
4. OCR Visibility Scoring
5. Global + Local Evidence Fusion
6. Final VLM Reasoning
"""

import logging
from typing import Dict, Any, List, Optional, Union
import numpy as np
from PIL import Image
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

from ..parsing.question_parser import QuestionParser, parse_question
from ..retrieval.frame_retrieval import retrieve_relevant_frames
from ..retrieval.region_localization import localize_target_regions
from ..retrieval.ocr_visibility import score_crop_visibility
from ..reasoning.qwen_reasoning import run_vlm_reasoning
from ..utils.prompt_builder import (
    build_frame_retrieval_prompt,
    build_region_localization_prompt,
    build_ocr_visibility_prompt,
)

logger = logging.getLogger(__name__)


class EvidenceMiningPipeline:
    """
    Hierarchical evidence mining pipeline for OCR-centric VideoQA.
    
    This pipeline processes a video question through multiple stages to gather
    targeted evidence and generate an accurate answer.
    """
    
    def __init__(self,
                 model: Optional[Qwen2_5_VLForConditionalGeneration] = None,
                 processor: Optional[AutoProcessor] = None,
                 device: str = "cuda:0"):
        """
        Initialize the evidence mining pipeline.
        
        Args:
            model: Qwen2.5-VL model instance. If None, loads default model.
            processor: AutoProcessor instance. If None, loads default processor.
            device: Device to run on (default: cuda:0)
        """
        self.model = model
        self.processor = processor
        self.device = device
        
        # Load model if not provided
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
    
    def run(self,
            question: str,
            frames: List[Union[np.ndarray, Image.Image]],
            top_k_frames: int = 5,
            verbose: bool = False) -> Dict[str, Any]:
        """
        Run the full evidence mining pipeline.
        
        Args:
            question: The video QA question
            frames: List of video frames (numpy arrays or PIL Images)
            top_k_frames: Number of top frames to retrieve (default: 5)
            verbose: Print debug information (default: False)
            
        Returns:
            Dict with keys:
                - answer: Final generated answer
                - parsed_question: Structured question representation
                - retrieval_results: Top retrieved frames
                - localization_results: Candidate regions
                - visibility_results: Best crop selection
                - reasoning_input: Images used for final reasoning
        """
        try:
            logger.info(f"Starting evidence mining pipeline for question: {question}")
            
            # Stage 1: Question Structural Parsing
            logger.info("Stage 1: Question Structural Parsing")
            parsed_question = self._stage_1_parse_question(question, verbose)
            
            # Stage 2: Frame-Level Relevant Frame Retrieval
            logger.info("Stage 2: Frame Retrieval")
            retrieval_prompt = build_frame_retrieval_prompt(parsed_question)
            retrieval_results = self._stage_2_retrieve_frames(
                frames, retrieval_prompt, top_k_frames, verbose
            )
            
            if not retrieval_results:
                logger.warning("No frames retrieved, using first frame as fallback")
                if isinstance(frames[0], np.ndarray):
                    frames[0] = Image.fromarray(frames[0].astype(np.uint8))
                return {
                    "answer": "Unable to find relevant frames",
                    "parsed_question": parsed_question,
                    "retrieval_results": [],
                    "localization_results": [],
                    "visibility_results": None,
                }
            
            # Stage 3: Target Region Localization
            logger.info("Stage 3: Region Localization")
            region_prompt = build_region_localization_prompt(parsed_question)
            localization_results = self._stage_3_localize_regions(
                retrieval_results, region_prompt, verbose
            )
            
            if not localization_results:
                logger.warning("No regions localized, using full frame")
                global_frame = retrieval_results[0]["frame"]
                local_crop = None
            else:
                # Stage 4: OCR Visibility Scoring
                logger.info("Stage 4: OCR Visibility Scoring")
                ocr_prompt = build_ocr_visibility_prompt(parsed_question)
                visibility_results = self._stage_4_score_visibility(
                    localization_results, ocr_prompt, verbose
                )
                
                if visibility_results["success"]:
                    global_frame = retrieval_results[0]["frame"]
                    local_crop = visibility_results["best_crop"]
                else:
                    global_frame = retrieval_results[0]["frame"]
                    local_crop = None
                    visibility_results = None
            
            # Stage 5 & 6: Evidence Fusion + Final Reasoning
            logger.info("Stage 5-6: Evidence Fusion and VLM Reasoning")
            answer = self._stage_5_6_reason(question, global_frame, local_crop, verbose)
            
            logger.info(f"Pipeline completed. Answer: {answer}")
            
            return {
                "success": True,
                "answer": answer,
                "parsed_question": parsed_question,
                "retrieval_results": retrieval_results,
                "localization_results": localization_results,
                "visibility_results": visibility_results if not localization_results else {
                    "best_crop": local_crop,
                    "scores": visibility_results,
                },
                "reasoning_input": {
                    "global_frame": global_frame,
                    "local_crop": local_crop,
                },
            }
            
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            return {
                "success": False,
                "answer": f"Error: {str(e)}",
                "error": str(e),
            }
    
    def _stage_1_parse_question(self, question: str, verbose: bool = False) -> Dict[str, str]:
        """Stage 1: Parse question into structured representation."""
        parser = QuestionParser(
            model=self.model,
            processor=self.processor,
            device=self.device
        )
        result = parser.parse(question)
        
        if verbose:
            logger.info(f"Parsed question: {result}")
        
        return result
    
    def _stage_2_retrieve_frames(self,
                                 frames: List[Union[np.ndarray, Image.Image]],
                                 retrieval_prompt: str,
                                 top_k: int,
                                 verbose: bool = False) -> List[Dict[str, Any]]:
        """Stage 2: Retrieve top-K relevant frames."""
        # Convert numpy arrays to PIL Images
        frames_pil = []
        for frame in frames:
            if isinstance(frame, np.ndarray):
                if frame.dtype != np.uint8:
                    frame = (frame * 255).astype(np.uint8) if frame.max() <= 1 else frame.astype(np.uint8)
                frames_pil.append(Image.fromarray(frame))
            else:
                frames_pil.append(frame)
        
        results = retrieve_relevant_frames(
            frames_pil,
            retrieval_prompt,
            model=self.model,
            processor=self.processor,
            device=self.device,
            top_k=top_k
        )
        
        if verbose:
            logger.info(f"Retrieved {len(results)} frames with scores: {[r['score'] for r in results]}")
        
        return results
    
    def _stage_3_localize_regions(self,
                                  retrieval_results: List[Dict[str, Any]],
                                  region_prompt: str,
                                  verbose: bool = False) -> List[Dict[str, Any]]:
        """Stage 3: Localize target regions in retrieved frames."""
        results = localize_target_regions(
            retrieval_results,
            region_prompt,
            model=self.model,
            processor=self.processor,
            device=self.device
        )
        
        if verbose:
            logger.info(f"Localized {len(results)} candidate regions")
        
        return results
    
    def _stage_4_score_visibility(self,
                                  localization_results: List[Dict[str, Any]],
                                  ocr_prompt: str,
                                  verbose: bool = False) -> Dict[str, Any]:
        """Stage 4: Score OCR visibility and select best crop."""
        results = score_crop_visibility(
            localization_results,
            ocr_prompt,
            model=self.model,
            processor=self.processor,
            device=self.device
        )
        
        if verbose and results["success"]:
            logger.info(f"Best crop scores: {results['best_scores']}")
        
        return results
    
    def _stage_5_6_reason(self,
                         question: str,
                         global_frame: Image.Image,
                         local_crop: Optional[Image.Image] = None,
                         verbose: bool = False) -> str:
        """Stage 5-6: Fuse evidence and generate final answer."""
        answer = run_vlm_reasoning(
            question,
            global_frame=global_frame,
            local_crop=local_crop,
            model=self.model,
            processor=self.processor,
            device=self.device
        )
        
        if verbose:
            logger.info(f"Generated answer: {answer}")
        
        return answer


def run_pipeline(question: str,
                frames: List[Union[np.ndarray, Image.Image]],
                model: Optional[Qwen2_5_VLForConditionalGeneration] = None,
                processor: Optional[AutoProcessor] = None,
                device: str = "cuda:0",
                top_k_frames: int = 5,
                verbose: bool = False) -> Dict[str, Any]:
    """
    Run the evidence mining pipeline.
    
    Args:
        question: Video QA question
        frames: List of video frames
        model: Optional Qwen model
        processor: Optional processor
        device: Device to run on
        top_k_frames: Number of top frames to retrieve
        verbose: Print debug information
        
    Returns:
        Pipeline results including final answer
    """
    pipeline = EvidenceMiningPipeline(
        model=model,
        processor=processor,
        device=device
    )
    return pipeline.run(question, frames, top_k_frames, verbose)
