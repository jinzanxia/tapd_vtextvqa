"""
Frame-Level Relevant Frame Retrieval.

Stage 2 of the hierarchical evidence mining pipeline.
Identifies frames that likely contain sufficient semantic evidence for answering the question.
Uses coarse-grained retrieval focusing on object existence and spatial relations.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

logger = logging.getLogger(__name__)


class FrameRetriever:
    """Retrieve relevant frames from video using semantic relevance scoring."""
    
    def __init__(self,
                 model: Optional[Qwen2_5_VLForConditionalGeneration] = None,
                 processor: Optional[AutoProcessor] = None,
                 device: str = "cuda:0"):
        """
        Initialize frame retriever.
        
        Args:
            model: Qwen2.5-VL model instance
            processor: AutoProcessor instance
            device: Device to run on
        """
        self.model = model
        self.processor = processor
        self.device = device
    
    def retrieve(self,
                 frames: List[np.ndarray],
                 retrieval_prompt: str,
                 top_k: int = 5,
                 batch_size: int = 4) -> List[Dict[str, Any]]:
        """
        Retrieve top-K relevant frames based on retrieval prompt.
        
        Args:
            frames: List of frames as numpy arrays (H, W, 3) or PIL Images
            retrieval_prompt: Retrieval-oriented question (e.g., "Does this frame contain a blue sign?")
            top_k: Number of top frames to return
            batch_size: Batch size for VLM inference
            
        Returns:
            List of dicts with keys: frame_id, frame, score
            Sorted by score (descending)
        """
        if not frames:
            logger.warning("Empty frames list")
            return []
        
        # Convert frames to PIL Images if needed
        frames_pil = self._ensure_pil_images(frames)
        
        # Score all frames
        scores = self._score_frames(frames_pil, retrieval_prompt, batch_size)
        
        # Create results
        results = []
        for frame_id, (frame, score) in enumerate(zip(frames_pil, scores)):
            results.append({
                "frame_id": frame_id,
                "frame": frame,
                "score": float(score),
            })
        
        # Sort by score (descending) and return top-k
        results.sort(key=lambda x: x["score"], reverse=True)
        results = results[:top_k]
        
        logger.info(f"Retrieved top-{len(results)} frames. Scores: {[r['score'] for r in results]}")
        
        return results
    
    def _ensure_pil_images(self, frames: List[Any]) -> List[Image.Image]:
        """
        Ensure frames are PIL Images.
        
        Args:
            frames: List of frames (numpy arrays or PIL Images)
            
        Returns:
            List of PIL Images
        """
        pil_frames = []
        for frame in frames:
            if isinstance(frame, np.ndarray):
                # Convert numpy array to PIL Image
                if frame.dtype != np.uint8:
                    frame = (frame * 255).astype(np.uint8) if frame.max() <= 1 else frame.astype(np.uint8)
                pil_frame = Image.fromarray(frame)
            elif isinstance(frame, Image.Image):
                pil_frame = frame
            else:
                logger.warning(f"Unsupported frame type: {type(frame)}")
                continue
            pil_frames.append(pil_frame)
        
        return pil_frames
    
    def _score_frames(self,
                      frames: List[Image.Image],
                      retrieval_prompt: str,
                      batch_size: int) -> np.ndarray:
        """
        Score frames using Qwen2.5-VL.
        
        Args:
            frames: List of PIL Images
            retrieval_prompt: Retrieval question
            batch_size: Batch size for inference
            
        Returns:
            Array of scores [0, 1] for each frame
        """
        scores = []
        
        # Process frames in batches
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i:i+batch_size]
            batch_scores = self._score_batch(batch_frames, retrieval_prompt)
            scores.extend(batch_scores)
        
        return np.array(scores)
    
    def _score_batch(self,
                     frames: List[Image.Image],
                     retrieval_prompt: str) -> List[float]:
        """
        Score a batch of frames.
        
        Args:
            frames: Batch of PIL Images
            retrieval_prompt: Retrieval question
            
        Returns:
            List of scores for the batch
        """
        from ..utils.prompt_builder import build_frame_relevance_scoring_prompt
        
        batch_size = len(frames)
        scoring_prompt = build_frame_relevance_scoring_prompt(retrieval_prompt)
        
        # Build conversations
        conversations = []
        for _ in range(batch_size):
            conversation = [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": scoring_prompt},
                    ]
                },
            ]
            conversations.append(conversation)
        
        try:
            # Prepare batch inputs
            texts = [
                self.processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
                for conv in conversations
            ]
            
            inputs = self.processor(
                text=texts,
                images=frames,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.model.device)
            
            # Generate responses
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False,
                    temperature=0,
                    num_beams=1,
                )
            
            # Decode responses
            generated_ids_trimmed = [
                out_ids[len(in_ids):] 
                for in_ids, out_ids in zip(inputs.input_ids, output_ids)
            ]
            responses = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
            
            # Extract scores from responses
            scores = []
            for response in responses:
                score = self._extract_confidence_score(response)
                scores.append(score)
            
            return scores
            
        except Exception as e:
            logger.error(f"Error scoring batch: {e}")
            return [0.0] * batch_size
    
    @staticmethod
    def _extract_confidence_score(response: str) -> float:
        """
        Extract confidence score from model response.
        
        Args:
            response: Model response text
            
        Returns:
            Confidence score in [0, 1]
        """
        import json
        
        try:
            # Try to find JSON in response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx >= 0 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                data = json.loads(json_str)
                
                # Extract confidence
                if "confidence" in data:
                    return float(data["confidence"])
                elif "score" in data:
                    return float(data["score"])
                else:
                    # Default score based on yes/no
                    if data.get("answer", "").lower() == "yes":
                        return 0.8
                    else:
                        return 0.2
            else:
                # Fallback: parse text for yes/no
                if "yes" in response.lower():
                    return 0.7
                else:
                    return 0.3
        except Exception as e:
            logger.debug(f"Error extracting score from response: {e}")
            return 0.5


def retrieve_relevant_frames(frames: List[np.ndarray],
                            retrieval_prompt: str,
                            model: Optional[Qwen2_5_VLForConditionalGeneration] = None,
                            processor: Optional[AutoProcessor] = None,
                            device: str = "cuda:0",
                            top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Retrieve top-K relevant frames.
    
    Args:
        frames: List of video frames as numpy arrays
        retrieval_prompt: Retrieval-oriented prompt
        model: Optional Qwen model
        processor: Optional processor
        device: Device to run on
        top_k: Number of frames to retrieve
        
    Returns:
        List of top-K frames with scores
    """
    retriever = FrameRetriever(model=model, processor=processor, device=device)
    return retriever.retrieve(frames, retrieval_prompt, top_k=top_k)
