"""
Integration example: Using the Evidence Mining Pipeline with existing inference code.

This example demonstrates how to integrate the new hierarchical evidence mining 
pipeline into the existing SFA-based VideoQA inference pipeline.
"""

import sys
import os
import json
from pathlib import Path
from typing import List, Dict, Any

# Import the new pipeline components
from pipeline.evidence_pipeline import EvidenceMiningPipeline, run_pipeline
from parsing.question_parser import parse_question
from retrieval.frame_retrieval import retrieve_relevant_frames
from retrieval.region_localization import localize_target_regions
from retrieval.ocr_visibility import score_crop_visibility
from reasoning.qwen_reasoning import run_vlm_reasoning
from utils.prompt_builder import (
    build_frame_retrieval_prompt,
    build_region_localization_prompt,
    build_ocr_visibility_prompt,
)

# Optionally import existing modules
try:
    from infer_codes.qwen_vison_process import process_vision_info, init_ocrmodel
except ImportError:
    print("Warning: Could not import existing inference modules")


class IntegratedVideoQAPipeline:
    """
    Integrated VideoQA pipeline combining SFA frame extraction with 
    new evidence mining framework.
    """
    
    def __init__(self, 
                 model_path: str,
                 adapter_path: str = None,
                 device: str = "cuda:0"):
        """
        Initialize integrated pipeline.
        
        Args:
            model_path: Path to Qwen2.5-VL model
            adapter_path: Optional LoRA adapter path
            device: Device to run on
        """
        self.device = device
        self.model = None
        self.processor = None
        self.evidence_pipeline = None
        
        self._load_model(model_path, adapter_path)
    
    def _load_model(self, model_path: str, adapter_path: str = None):
        """Load Qwen model with optional LoRA adapter."""
        import torch
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        try:
            from peft import PeftModel
        except ImportError:
            PeftModel = None

        print(f"Loading model from {model_path}")

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            device_map=self.device,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )

        if adapter_path:
            if PeftModel is None:
                raise ImportError("Loading a LoRA adapter requires the `peft` package.")
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            self.model.eval()

        processor_path = adapter_path if adapter_path else model_path
        try:
            self.processor = AutoProcessor.from_pretrained(processor_path)
        except Exception:
            self.processor = AutoProcessor.from_pretrained(model_path)
        
        print("Model loaded successfully")
        
        # Initialize evidence mining pipeline
        self.evidence_pipeline = EvidenceMiningPipeline(
            model=self.model,
            processor=self.processor,
            device=self.device
        )
    
    def process_video_qas(self,
                         qa_data: List[Dict[str, Any]],
                         video_dir: str,
                         output_json: str,
                         use_evidence_mining: bool = True,
                         num_sampled_frames: int = 16,
                         top_k_frames: int = 5,
                         verbose: bool = False) -> Dict[str, str]:
        """
        Process VideoQA dataset using the integrated pipeline.
        
        Args:
            qa_data: List of QA dicts with keys: question, video_id/videoId, etc.
            video_dir: Directory containing video files
            output_json: Output path for predictions
            use_evidence_mining: Whether to use new evidence mining pipeline (default: True)
            num_sampled_frames: Number of frames to sample from video
            top_k_frames: Number of top frames to retrieve in evidence mining
            verbose: Print debug information
            
        Returns:
            Dict of predictions {qid: answer}
        """
        predictions = {}
        
        for data in qa_data:
            question = data['question']
            qid = data.get('question_id') or data.get('questionId')
            video_id = data.get('video_id') or data.get('videoId')
            
            # Get video path
            video_path = os.path.join(video_dir, video_id + '.mp4')
            if not os.path.exists(video_path):
                print(f"Video not found: {video_path}")
                predictions[qid] = "Video not found"
                continue
            
            # Sample frames from video
            frames = self._sample_frames_from_video(video_path, num_sampled_frames)
            if not frames:
                print(f"Failed to sample frames from {video_path}")
                predictions[qid] = "Failed to process video"
                continue
            
            try:
                if use_evidence_mining:
                    # Use new evidence mining pipeline
                    result = self.evidence_pipeline.run(
                        question,
                        frames,
                        top_k_frames=top_k_frames,
                        verbose=verbose
                    )
                    answer = result['answer']
                else:
                    # Use simple single-frame inference (baseline)
                    answer = self._simple_inference(question, frames[0])
                
                predictions[qid] = answer
                
                if verbose:
                    print(f"Q: {question}")
                    print(f"A: {answer}\n")
                
            except Exception as e:
                print(f"Error processing QA {qid}: {e}")
                predictions[qid] = "Error processing"
                torch.cuda.empty_cache()
                continue
        
        # Save predictions
        self._save_predictions(predictions, output_json)
        
        return predictions
    
    def _sample_frames_from_video(self, video_path: str, num_frames: int) -> List[Any]:
        """
        Sample frames from video at regular intervals.
        
        Args:
            video_path: Path to video file
            num_frames: Number of frames to sample
            
        Returns:
            List of sampled frames as numpy arrays
        """
        try:
            import cv2
            import numpy as np
            from PIL import Image
        except ImportError as e:
            print(f"Missing runtime dependency for frame sampling: {e}")
            return []

        try:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames <= 0:
                print(f"Invalid video: {video_path}")
                return []
            
            # Calculate sampling interval
            interval = max(1, total_frames // num_frames)
            
            frames = []
            frame_idx = 0
            
            while len(frames) < num_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
                
                frame_idx += interval
            
            cap.release()
            
            return frames
        except Exception as e:
            print(f"Error sampling frames: {e}")
            return []
    
    def _simple_inference(self, question: str, frame: Any) -> str:
        """
        Simple single-frame inference (baseline).
        
        Args:
            question: Question text
            frame: Frame as numpy array
            
        Returns:
            Answer string
        """
        from utils.prompt_builder import build_simple_reasoning_prompt
        try:
            import numpy as np
            from PIL import Image
            import torch
        except ImportError as e:
            print(f"Missing runtime dependency for simple inference: {e}")
            return "Error generating answer"
        
        try:
            # Convert frame to PIL
            if isinstance(frame, np.ndarray) and frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8)
            pil_frame = Image.fromarray(frame)
            
            # Generate answer
            prompt = build_simple_reasoning_prompt(question)
            
            conversation = [
                {"role": "system", "content": "You are a helpful assistant."},
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
            
            inputs = self.processor(
                text=[text],
                images=[pil_frame],
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
            response = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0].strip()
            
            return response.replace("Answer:", "").strip()
        
        except Exception as e:
            print(f"Error in simple inference: {e}")
            return "Error generating answer"
    
    @staticmethod
    def _save_predictions(predictions: Dict[str, str], output_json: str):
        """Save predictions to JSON file."""
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(predictions, f, indent=2, ensure_ascii=False)
        print(f"Predictions saved to {output_json}")


def main():
    """Example usage of the integrated pipeline."""
    
    # Configuration
    model_path = "Qwen/Qwen2.5-VL-7B-Instruct"  # or path to local model
    adapter_path = None  # Optional LoRA adapter
    device = "cuda:0"
    
    # Initialize pipeline
    pipeline = IntegratedVideoQAPipeline(
        model_path=model_path,
        adapter_path=adapter_path,
        device=device
    )
    
    # Example QA data
    qa_data = [
        {
            "question": "What does the sign in blue on top of the road say?",
            "video_id": "example_video_1",
            "question_id": "q1",
        },
        {
            "question": "How many vehicles are in the scene?",
            "video_id": "example_video_2",
            "question_id": "q2",
        },
    ]
    
    # Process QAs with evidence mining
    predictions = pipeline.process_video_qas(
        qa_data=qa_data,
        video_dir="/path/to/videos",
        output_json="/path/to/predictions.json",
        use_evidence_mining=True,
        num_sampled_frames=16,
        top_k_frames=5,
        verbose=True
    )
    
    print(f"Processed {len(predictions)} questions")


if __name__ == "__main__":
    main()
