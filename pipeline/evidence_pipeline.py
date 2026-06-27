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

from parsing.question_parser import QuestionParser, parse_question
from retrieval.frame_retrieval import retrieve_relevant_frames
from retrieval.region_localization import localize_target_regions
from retrieval.ocr_visibility import score_crop_visibility, OCRVisibilityScorer
from reasoning.qwen_reasoning import run_vlm_reasoning
from utils.prompt_builder import (
    build_frame_retrieval_prompt,
    build_region_localization_prompt,
    build_ocr_visibility_prompt,
    build_crop_localization_scoring_prompt,
)

logger = logging.getLogger(__name__)

LOCALIZATION_KEYWORDS = [
    "say", "written", "text", "word", "sign",
    "license", "number", "logo", "read",
    "on the", "written on", "shown on",
]

NO_LOCALIZATION_KEYWORDS = [
    "percent", "increase", "decrease", "chart",
    "graph", "trend", "price", "stock",
]


def route_question(question: str) -> str:
    """
    Route a question to local evidence mining or global frame reasoning.

    Returns:
        "local" for OCR/small-target/region-based questions, otherwise "global".
    """
    q = question.lower()

    for kw in NO_LOCALIZATION_KEYWORDS:
        if kw in q:
            return "global"

    for kw in LOCALIZATION_KEYWORDS:
        if kw in q:
            return "local"

    return "global"


class EvidenceMiningPipeline:
    """
    Hierarchical evidence mining pipeline for OCR-centric VideoQA.
    
    This pipeline processes a video question through multiple stages to gather
    targeted evidence and generate an accurate answer.
    """
    
    def __init__(self,
                 model: Optional[Qwen2_5_VLForConditionalGeneration] = None,
                 processor: Optional[AutoProcessor] = None,
                 device: str = "cuda:0",
                 ocr_score_mode: str = "paddle",
                 reasoning_evidence_mode: str = "both",
                 reasoning_global_frame_count: int = 3,
                 reasoning_local_crop_count: int = 3):
        """
        Initialize the evidence mining pipeline.
        
        Args:
            model: Qwen2.5-VL model instance. If None, loads default model.
            processor: AutoProcessor instance. If None, loads default processor.
            device: Device to run on (default: cuda:0)
            ocr_score_mode: "paddle" or "vlm" for OCR readability scoring
            reasoning_evidence_mode: "both", "global", or "local" for final reasoning
            reasoning_global_frame_count: Number of retrieved global frames to pass to final reasoning
            reasoning_local_crop_count: Number of ranked local crops to pass to final reasoning
        """
        self.model = model
        self.processor = processor
        self.device = device
        self.ocr_score_mode = ocr_score_mode
        if reasoning_evidence_mode not in {"both", "global", "local"}:
            logger.warning(f"Unknown reasoning evidence mode '{reasoning_evidence_mode}', using both")
            reasoning_evidence_mode = "both"
        self.reasoning_evidence_mode = reasoning_evidence_mode
        self.reasoning_global_frame_count = max(1, int(reasoning_global_frame_count))
        self.reasoning_local_crop_count = max(1, int(reasoning_local_crop_count))
        
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
            verbose: bool = False,
            reasoning_global_frame_count: Optional[int] = None,
            reasoning_local_crop_count: Optional[int] = None) -> Dict[str, Any]:
        """
        Run the full evidence mining pipeline.
        
        Args:
            question: The video QA question
            frames: List of video frames (numpy arrays or PIL Images)
            top_k_frames: Number of top frames to retrieve (default: 5)
            reasoning_global_frame_count: Override number of global frames for final reasoning
            reasoning_local_crop_count: Override number of local crops for final reasoning
            verbose: Print debug information (default: False)
            
        Returns:
            Dict with keys:
                - answer: Final generated answer
                - question_type: "local" or "global" routing decision
                - parsed_question: Structured question representation
                - retrieval_results: Top retrieved frames
                - localization_results: Candidate regions
                - visibility_results: Best crop selection
                - reasoning_input: Images used for final reasoning
        """
        try:
            logger.info(f"Starting evidence mining pipeline for question: {question}")
            question_type = route_question(question)
            logger.info(f"Question Type Routing: {question_type}")
            reasoning_global_frame_count = max(
                1,
                int(reasoning_global_frame_count or self.reasoning_global_frame_count),
            )
            reasoning_local_crop_count = max(
                1,
                int(reasoning_local_crop_count or self.reasoning_local_crop_count),
            )

            if not frames:
                logger.warning("No input frames provided to evidence mining pipeline")
                return {
                    "success": False,
                    "answer": "Failed to process video",
                    "error": "no input frames",
                    "parsed_question": None,
                    "question_type": question_type,
                    "retrieval_results": [],
                    "localization_results": [],
                    "visibility_results": None,
                }

            sampled_frames = self._ensure_pil_frames(frames)
            if not sampled_frames:
                logger.warning("No valid frames after PIL conversion")
                return {
                    "success": False,
                    "answer": "Failed to process video",
                    "error": "no valid input frames",
                    "parsed_question": None,
                    "question_type": question_type,
                    "retrieval_results": [],
                    "localization_results": [],
                    "visibility_results": None,
                }
            parsed_question = None
            localization_results = []
            visibility_results = None
            local_crop_entries = []

            if question_type == "local":
                # Stage 1: Question Structural Parsing
                logger.info("Stage 1: Question Structural Parsing")
                parsed_question = self._stage_1_parse_question(question, verbose)
                retrieval_prompt = build_frame_retrieval_prompt(parsed_question)
            else:
                logger.info("Stage 1: Question Structural Parsing skipped for global route")
                retrieval_prompt = question

            # Stage 2: Frame-Level Relevant Frame Retrieval / global frame selection
            logger.info("Stage 2: Frame Retrieval")
            retrieval_results = self._stage_2_retrieve_frames(
                sampled_frames, retrieval_prompt, top_k_frames, verbose
            )
            
            if not retrieval_results:
                logger.warning("No frames retrieved")
                return {
                    "success": False,
                    "answer": "Unable to find relevant frames",
                    "parsed_question": parsed_question,
                    "question_type": question_type,
                    "retrieval_results": [],
                    "localization_results": [],
                    "visibility_results": None,
                }

            if question_type == "global":
                logger.info("Global route: skipping localization and OCR visibility scoring")
                global_frames = self._select_top_frames(retrieval_results, reasoning_global_frame_count)
                answer = self._stage_5_6_reason(question, global_frames, [], parsed_question, verbose)

                logger.info(f"Pipeline completed. Answer: {answer}")

                return {
                    "success": True,
                    "answer": answer,
                    "parsed_question": parsed_question,
                    "question_type": question_type,
                    "retrieval_results": retrieval_results,
                    "localization_results": localization_results,
                    "visibility_results": visibility_results,
                    "reasoning_input": {
                        "mode": "global",
                        "global_frames": global_frames,
                        "local_crops": [],
                        "global_frame": global_frames[0] if global_frames else None,
                        "local_crop": None,
                    },
                }

            # Stage 3: Target Region Localization
            logger.info("Stage 3: Region Localization")
            region_prompt = build_region_localization_prompt(parsed_question)
            localization_results = self._stage_3_localize_regions(
                retrieval_results, region_prompt, verbose
            )
            
            if not localization_results:
                logger.warning("No regions localized, using full frame")
                global_frames = sampled_frames
                local_crops = []
            else:
                # Stage 4: OCR Visibility Scoring
                logger.info("Stage 4: Candidate Crop Scoring")
                ocr_prompt = build_ocr_visibility_prompt(parsed_question)
                crop_localization_prompt = build_crop_localization_scoring_prompt(parsed_question)
                visibility_results = self._stage_4_score_visibility(
                    localization_results, ocr_prompt, crop_localization_prompt, verbose
                )
                
                if visibility_results["success"]:
                    global_frames = sampled_frames
                    local_crop_entries = self._select_top_crop_entries(
                        visibility_results,
                        reasoning_local_crop_count,
                        retrieval_results=retrieval_results,
                    )
                    local_crops = [entry["crop"] for entry in local_crop_entries]
                else:
                    global_frames = sampled_frames
                    local_crops = []
                    visibility_results = None
            
            # Stage 5 & 6: Evidence Fusion + Final Reasoning
            logger.info("Stage 5-6: Evidence Fusion and VLM Reasoning")
            if self.reasoning_evidence_mode == "local":
                if local_crops:
                    reasoning_global_frames = []
                    reasoning_local_crops = local_crops
                    actual_reasoning_mode = "local"
                else:
                    logger.warning("Local-only reasoning requested but local crop is missing; falling back to direct frames")
                    reasoning_global_frames = sampled_frames
                    reasoning_local_crops = []
                    actual_reasoning_mode = "direct_frames_fallback"
            else:
                reasoning_global_frames = self._build_crop_replaced_frame_sequence(
                    global_frames,
                    local_crop_entries,
                    enable_replacement=self.reasoning_evidence_mode != "global",
                )
                reasoning_local_crops = []
                actual_reasoning_mode = (
                    "crop_replaced_direct_frames"
                    if local_crop_entries and self.reasoning_evidence_mode != "global"
                    else "direct_frames"
                )
            answer = self._stage_5_6_reason(
                question,
                reasoning_global_frames,
                reasoning_local_crops,
                parsed_question,
                verbose,
                postprocess_local_crop=local_crops,
                context=(
                    "The images follow the uniformly sampled video-frame order. "
                    "Some positions may be zoomed local crop replacements for their original frames."
                    if actual_reasoning_mode == "crop_replaced_direct_frames"
                    else ""
                ),
            )

            logger.info(f"Pipeline completed. Answer: {answer}")
            
            return {
                "success": True,
                "answer": answer,
                "parsed_question": parsed_question,
                "question_type": question_type,
                "retrieval_results": retrieval_results,
                "localization_results": localization_results,
                "visibility_results": visibility_results if not localization_results else {
                    "best_crop": local_crops[0] if local_crops else None,
                    "top_crops": local_crops,
                    "scores": visibility_results,
                },
                "reasoning_input": {
                    "mode": actual_reasoning_mode,
                    "global_frames": reasoning_global_frames,
                    "local_crops": reasoning_local_crops,
                    "global_frame": reasoning_global_frames[0] if reasoning_global_frames else None,
                    "local_crop": reasoning_local_crops[0] if reasoning_local_crops else None,
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
                                  crop_localization_prompt: str,
                                  verbose: bool = False) -> Dict[str, Any]:
        """Stage 4: Score candidate crops and select best local evidence."""
        results = score_crop_visibility(
            localization_results,
            ocr_prompt,
            crop_localization_prompt=crop_localization_prompt,
            model=self.model,
            processor=self.processor,
            device=self.device,
            ocr_score_mode=self.ocr_score_mode
        )
        
        if verbose and results["success"]:
            logger.info(f"Best crop scores: {results['best_scores']}")
        
        return results
    
    def _stage_5_6_reason(self,
                         question: str,
                         global_frame: Union[Image.Image, List[Image.Image], None],
                         local_crop: Union[Image.Image, List[Image.Image], None] = None,
                         parsed_question: Optional[Dict[str, Any]] = None,
                         verbose: bool = False,
                         postprocess_local_crop: Union[Image.Image, List[Image.Image], None] = None,
                         context: str = "") -> str:
        """Stage 5-6: Fuse evidence and generate final answer."""
        answer = run_vlm_reasoning(
            question,
            global_frame=global_frame,
            local_crop=local_crop,
            model=self.model,
            processor=self.processor,
            device=self.device,
            context=context,
        )

        # Post-process answer to prefer short, extractable content for OCR-like tasks
        answer_post = self._postprocess_answer(
            answer,
            parsed_question,
            postprocess_local_crop if postprocess_local_crop is not None else local_crop,
        )
        if verbose and answer != answer_post:
            logger.info(f"Post-processed answer: '{answer}' -> '{answer_post}'")

        return answer_post

    def _select_reasoning_evidence(self,
                                   global_frames: List[Image.Image],
                                   local_crops: List[Image.Image]):
        """Select final reasoning evidence according to configured ablation mode."""
        if self.reasoning_evidence_mode == "global":
            return global_frames, [], "global"

        if self.reasoning_evidence_mode == "local":
            if local_crops:
                return [], local_crops, "local"
            logger.warning("Local-only reasoning requested but local crop is missing; falling back to global")
            return global_frames, [], "global_fallback"

        return global_frames, local_crops, "both" if local_crops else "global_fallback"

    @staticmethod
    def _select_top_frames(retrieval_results: List[Dict[str, Any]], count: int) -> List[Image.Image]:
        """Return top retrieved frames for final reasoning."""
        return [item["frame"] for item in retrieval_results[:max(1, count)] if item.get("frame") is not None]

    @staticmethod
    def _ensure_pil_frames(frames: List[Union[np.ndarray, Image.Image]]) -> List[Image.Image]:
        """Convert sampled video frames to PIL Images while preserving order and count."""
        frames_pil = []
        for frame in frames:
            if isinstance(frame, np.ndarray):
                if frame.dtype != np.uint8:
                    frame = (frame * 255).astype(np.uint8) if frame.max() <= 1 else frame.astype(np.uint8)
                frames_pil.append(Image.fromarray(frame))
            elif isinstance(frame, Image.Image):
                frames_pil.append(frame)
            else:
                logger.warning(f"Unsupported frame type: {type(frame)}")
        return frames_pil

    @staticmethod
    def _build_crop_replaced_frame_sequence(frames: List[Image.Image],
                                            crop_entries: List[Dict[str, Any]],
                                            enable_replacement: bool = True) -> List[Image.Image]:
        """
        Return the full sampled-frame sequence, replacing selected frame positions with crops.

        This keeps the final Qwen input aligned with direct-frame inference: same sampling
        method, same number of images, with local evidence substituted in-place.
        """
        reasoning_frames = list(frames)
        if not enable_replacement:
            return reasoning_frames

        replaced_frame_ids = set()
        for entry in crop_entries:
            frame_id = entry.get("frame_id")
            crop = entry.get("crop")
            try:
                frame_id = int(frame_id) if frame_id is not None else None
            except (TypeError, ValueError):
                frame_id = None
            if (
                crop is None
                or frame_id is None
                or frame_id in replaced_frame_ids
                or frame_id < 0
                or frame_id >= len(reasoning_frames)
            ):
                continue
            reasoning_frames[frame_id] = crop
            replaced_frame_ids.add(frame_id)
        return reasoning_frames

    @staticmethod
    def _select_top_crop_entries(visibility_results: Dict[str, Any],
                                 count: int,
                                 retrieval_results: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """Return ranked crop entries with frame ids, preferring one crop per retrieved frame."""
        scores = visibility_results.get("scores") or []
        count = max(1, count)
        entries = []

        def add_entry(score_idx: int, item: Dict[str, Any]) -> None:
            region = item.get("region") or {}
            crop = item.get("crop")
            frame_id = region.get("frame_id")
            if crop is None or frame_id is None:
                return
            entries.append({
                "crop": crop,
                "frame_id": frame_id,
                "score_index": score_idx,
                "score": item.get("final_score", item.get("combined_score")),
                "region": region,
            })

        if retrieval_results:
            used_score_ids = set()
            used_frame_ids = set()
            for frame_result in retrieval_results:
                frame_id = frame_result.get("frame_id")
                if frame_id in used_frame_ids:
                    continue
                for score_idx, item in enumerate(scores):
                    region = item.get("region") or {}
                    if score_idx in used_score_ids:
                        continue
                    if region.get("frame_id") == frame_id:
                        before_count = len(entries)
                        add_entry(score_idx, item)
                        if len(entries) > before_count:
                            used_score_ids.add(score_idx)
                            used_frame_ids.add(frame_id)
                        break
                if len(entries) >= count:
                    break

            if len(entries) < count:
                for score_idx, item in enumerate(scores):
                    region = item.get("region") or {}
                    frame_id = region.get("frame_id")
                    if score_idx in used_score_ids or frame_id in used_frame_ids:
                        continue
                    before_count = len(entries)
                    add_entry(score_idx, item)
                    if len(entries) > before_count:
                        used_score_ids.add(score_idx)
                        used_frame_ids.add(frame_id)
                    if len(entries) >= count:
                        break
        else:
            used_frame_ids = set()
            for score_idx, item in enumerate(scores):
                region = item.get("region") or {}
                frame_id = region.get("frame_id")
                if frame_id in used_frame_ids:
                    continue
                before_count = len(entries)
                add_entry(score_idx, item)
                if len(entries) > before_count:
                    used_frame_ids.add(frame_id)
                if len(entries) >= count:
                    break

        return entries

    @staticmethod
    def _select_top_crops(visibility_results: Dict[str, Any],
                          count: int,
                          retrieval_results: Optional[List[Dict[str, Any]]] = None) -> List[Image.Image]:
        """Return top crops, preferring one crop per retrieved frame in global-frame order."""
        scores = visibility_results.get("scores") or []
        count = max(1, count)
        crops = []

        if retrieval_results:
            used_score_ids = set()
            for frame_result in retrieval_results:
                frame_id = frame_result.get("frame_id")
                for score_idx, item in enumerate(scores):
                    region = item.get("region") or {}
                    if score_idx in used_score_ids:
                        continue
                    if region.get("frame_id") == frame_id and item.get("crop") is not None:
                        crops.append(item["crop"])
                        used_score_ids.add(score_idx)
                        break
                if len(crops) >= count:
                    break

            if len(crops) < count:
                for score_idx, item in enumerate(scores):
                    if score_idx in used_score_ids or item.get("crop") is None:
                        continue
                    crops.append(item["crop"])
                    if len(crops) >= count:
                        break
        else:
            crops = [item["crop"] for item in scores[:count] if item.get("crop") is not None]

        if not crops and visibility_results.get("best_crop") is not None:
            crops = [visibility_results["best_crop"]]
        return crops

    def _postprocess_answer(self,
                            answer: str,
                            parsed_question: Optional[Dict[str, Any]],
                            local_crop: Union[Image.Image, List[Image.Image], None]) -> str:
        """
        Heuristic post-processing to convert verbose VLM answers into concise expected answers.

        Strategies:
        - For OCR/detection tasks: extract numbers, short letter sequences, or URL-like tokens.
        - If extraction fails and PaddleOCR is available, run OCR on the selected crop and return detected text.
        - Otherwise return the original answer trimmed.
        """
        import re
        ans = (answer or "").strip()
        local_crops = local_crop if isinstance(local_crop, list) else ([local_crop] if local_crop is not None else [])
        task = None
        if parsed_question and isinstance(parsed_question, dict):
            task = parsed_question.get('task')

        # OCR-like extraction heuristics
        if task in {'ocr', 'detection'} or (parsed_question and 'number' in (parsed_question.get('target','') or '').lower()):
            # 1) Try number extraction
            m = re.search(r"\b(\d{1,5})\b", ans)
            if m:
                return m.group(1)

            # 2) Try short alpha tokens (e.g., 'QU')
            m = re.findall(r"\b([A-Za-z]{1,4})\b", ans)
            if m:
                for token in m:
                    if token.isalpha() and len(token) <= 3:
                        return token

            # 3) Try URL-like extraction
            m = re.search(r"(https?://\S+|www\.\S+|[\w.-]+\.(com|net|org|io|cn|gov)(/\S*)?)", ans, re.IGNORECASE)
            if m:
                return m.group(0)

            # 4) If we have a crop, try PaddleOCR as a fallback
            for crop in local_crops:
                try:
                    ocr_model = OCRVisibilityScorer._get_shared_ocr_model()
                    if ocr_model is not None:
                        import numpy as np
                        crop_np = np.array(crop)
                        results = ocr_model.ocr(crop_np, cls=True)
                        if results and results[0]:
                            texts = [r[1][0] for r in results[0] if r and len(r) > 1]
                            if texts:
                                return " ".join(texts).strip()
                except Exception as e:
                    logger.debug(f"PaddleOCR fallback failed: {e}")

        # Baseline-style initial cleanup (match SFA baseline)
        ans = ans.replace("Answer:", "").replace("The answer is", "").strip()
        if ans.endswith('.'):
            ans = ans[:-1].strip()

        # OCR-specific heuristics after baseline cleanup
        m = re.search(r"\b(\d{1,5})\b", ans)
        if m and (task in {'ocr', 'detection'} or (parsed_question and 'number' in (parsed_question.get('target','') or '').lower())):
            return m.group(1)

        # short alpha tokens (e.g., 'QU') for OCR/detection
        if task in {'ocr', 'detection'}:
            m = re.findall(r"\b([A-Za-z]{1,4})\b", ans)
            if m:
                for token in m:
                    if token.isalpha() and len(token) <= 3:
                        return token

            # URL-like
            m = re.search(r"(https?://\S+|www\.\S+|[\w.-]+\.(com|net|org|io|cn|gov)(/\S*)?)", ans, re.IGNORECASE)
            if m:
                return m.group(0)

            # PaddleOCR fallback on crop
            for crop in local_crops:
                try:
                    ocr_model = OCRVisibilityScorer._get_shared_ocr_model()
                    if ocr_model is not None:
                        import numpy as np
                        crop_np = np.array(crop)
                        results = ocr_model.ocr(crop_np, cls=True)
                        if results and results[0]:
                            texts = [r[1][0] for r in results[0] if r and len(r) > 1]
                            if texts:
                                return " ".join(texts).strip()
                except Exception as e:
                    logger.debug(f"PaddleOCR fallback failed: {e}")

        # Final templated shorten (e.g., 'Exit 13' -> '13')
        m = re.search(r"Exit\s*(\d{1,5})", ans, re.IGNORECASE)
        if m:
            return m.group(1)

        return ans


def run_pipeline(question: str,
                frames: List[Union[np.ndarray, Image.Image]],
                model: Optional[Qwen2_5_VLForConditionalGeneration] = None,
                processor: Optional[AutoProcessor] = None,
                device: str = "cuda:0",
                top_k_frames: int = 5,
                ocr_score_mode: str = "paddle",
                reasoning_evidence_mode: str = "both",
                reasoning_global_frame_count: int = 3,
                reasoning_local_crop_count: int = 3,
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
        ocr_score_mode: "paddle" or "vlm" for OCR readability scoring
        reasoning_evidence_mode: "both", "global", or "local" for final reasoning
        reasoning_global_frame_count: Number of global frames for final reasoning
        reasoning_local_crop_count: Number of local crops for final reasoning
        verbose: Print debug information
        
    Returns:
        Pipeline results including final answer
    """
    pipeline = EvidenceMiningPipeline(
        model=model,
        processor=processor,
        device=device,
        ocr_score_mode=ocr_score_mode,
        reasoning_evidence_mode=reasoning_evidence_mode,
        reasoning_global_frame_count=reasoning_global_frame_count,
        reasoning_local_crop_count=reasoning_local_crop_count
    )
    return pipeline.run(
        question,
        frames,
        top_k_frames=top_k_frames,
        reasoning_global_frame_count=reasoning_global_frame_count,
        reasoning_local_crop_count=reasoning_local_crop_count,
        verbose=verbose,
    )
