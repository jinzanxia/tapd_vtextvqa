"""
Inference script for VideoQA with Evidence Mining Pipeline.

Compared to baseline qwen.py, this integrates the new hierarchical
evidence mining framework for improved accuracy on OCR-centric questions.

Usage:
    python infer_with_evidence_pipeline.py \
        --gt-json data.json \
        --video-dir /path/to/videos \
        --model-name "Qwen/Qwen2.5-VL-7B-Instruct" \
        --vts-config config.yaml \
        --vts-model model.pth \
        --output results.json \
        [--adapter-path path/to/lora] \
        [--num-sampled-frames 16] \
        [--top-k-frames 5] \
        [--use-evidence-mining]
"""

import torch
import json
import os
import codecs
import argparse
import logging
import time
import sys
import importlib
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import models
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
try:
    from peft import PeftModel
except ImportError:
    PeftModel = None

# Import evidence pipeline
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

PIPELINE_AVAILABLE = False
IntegratedVideoQAPipeline = None
try:
    pipeline_module = importlib.import_module("pipeline_integration_example")
    IntegratedVideoQAPipeline = getattr(pipeline_module, "IntegratedVideoQAPipeline")
    PIPELINE_AVAILABLE = True
except Exception as e:
    logger.warning("Could not import evidence pipeline module.")
    logger.warning(f"Import error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

# Import metrics
from metric import anls_metric, stvqa_acc_metric


def sample_frames_from_video(video_path, num_frames):
    """Sample RGB frames uniformly from a video file."""
    try:
        import cv2
    except ImportError as e:
        logger.error(f"OpenCV is required for frame sampling: {e}")
        return []

    if not os.path.exists(video_path):
        logger.warning(f"Video not found: {video_path}")
        return []

    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            logger.warning(f"Could not open video: {video_path}")
            return []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            logger.warning(f"Invalid or empty video: {video_path}")
            return []

        num_frames = max(1, min(num_frames, total_frames))
        if num_frames == 1:
            frame_indices = [total_frames // 2]
        else:
            frame_indices = [
                round(i * (total_frames - 1) / (num_frames - 1))
                for i in range(num_frames)
            ]

        frames = []
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                logger.warning(f"Failed to read frame {frame_idx} from {video_path}")
                continue
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if not frames:
            logger.warning(f"No frames sampled from {video_path}")
        return frames
    finally:
        cap.release()


def get_parser():
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description="VideoQA Inference with Evidence Mining Pipeline"
    )
    
    # Core arguments
    parser.add_argument("--gt-json", required=True, help="Ground truth JSON file path")
    parser.add_argument("--model-name", required=True, help="Qwen model path")
    parser.add_argument("--video-dir", required=True, help="Input video directory path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    
    # Optional model arguments
    parser.add_argument("--adapter-path", default=None, help="LoRA adapter path")
    parser.add_argument("--vts-config", default=None, help="VTS model config file path")
    parser.add_argument("--vts-model", default=None, help="VTS model path")
    
    # Evidence mining arguments
    parser.add_argument(
        "--use-evidence-mining",
        action="store_true",
        default=True,
        help="Use new evidence mining pipeline (default: True)"
    )
    parser.add_argument(
        "--no-evidence-mining",
        dest="use_evidence_mining",
        action="store_false",
        help="Use baseline single-frame inference"
    )
    
    parser.add_argument(
        "--num-sampled-frames",
        type=int,
        default=16,
        help="Number of frames to sample from video (default: 16)"
    )
    parser.add_argument(
        "--top-k-frames",
        type=int,
        default=5,
        help="Number of top frames to retrieve (default: 5)"
    )
    
    # Compatibility arguments (for baseline modes)
    parser.add_argument("--no-ocr-text", action="store_true", default=False)
    parser.add_argument("--no-focus-bonus", action="store_true", default=False)
    parser.add_argument("--layout-zoom", type=str, default="off")
    parser.add_argument("--crop-mode", type=str, default="fixed")
    parser.add_argument("--density-top-k", type=int, default=0)
    parser.add_argument("--density-nms", type=float, default=0.5)
    parser.add_argument("--no-object-detect", action="store_true", default=False)
    parser.add_argument("--d2-config", type=str, default="detectron2_coco.yaml")
    parser.add_argument("--d2-weights", type=str, default=None)
    parser.add_argument("--d2-obj-classes", type=str, default=None)
    
    # Verbose flag
    parser.add_argument("--verbose", action="store_true", default=False)
    
    return parser


def main():
    """Main inference function."""
    args = get_parser().parse_args()
    
    logger.info("="*70)
    logger.info("VideoQA Inference with Evidence Mining Pipeline")
    logger.info("="*70)
    
    # Validate mode
    if args.use_evidence_mining and not PIPELINE_AVAILABLE:
        logger.error("Evidence mining pipeline not available!")
        logger.error("Set --no-evidence-mining to use baseline mode")
        return
    
    mode = "Evidence Mining" if args.use_evidence_mining else "Baseline"
    logger.info(f"Mode: {mode}")
    logger.info(f"Input: {args.gt_json}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Videos: {args.video_dir}")
    logger.info("")
    
    # Setup device
    device = "cuda:0"
    torch.cuda.set_device(device)
    
    # Initialize metrics
    anls_metr = anls_metric.ANLS_metric()
    stvqa_acc_metr = stvqa_acc_metric.STVQAAcc_metric()
    
    # Load model and processor
    logger.info(f"Loading model: {args.model_name}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name,
        device_map=device,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    
    # Load LoRA adapter if provided
    if args.adapter_path:
        if PeftModel is None:
            raise ImportError("Loading LoRA requires the `peft` package")
        logger.info(f"Loading LoRA adapter: {args.adapter_path}")
        model = PeftModel.from_pretrained(model, args.adapter_path)
        model.eval()
    
    processor_path = args.adapter_path if args.adapter_path else args.model_name
    try:
        processor = AutoProcessor.from_pretrained(processor_path)
    except Exception:
        processor = AutoProcessor.from_pretrained(args.model_name)
    
    logger.info("Model loaded successfully\n")
    
    # Load ground truth data
    logger.info(f"Loading ground truth: {args.gt_json}")
    with open(args.gt_json, 'r', encoding='utf-8') as f:
        gt_data = json.load(f)
    logger.info(f"Loaded {len(gt_data['data'])} questions\n")
    
    # Initialize pipeline
    if args.use_evidence_mining:
        logger.info("Initializing Evidence Mining Pipeline...")
        try:
            pipeline = IntegratedVideoQAPipeline(
                model_path=args.model_name,
                adapter_path=args.adapter_path,
                device=device
            )
            logger.info("Pipeline initialized\n")
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {e}")
            logger.warning("Falling back to baseline mode")
            args.use_evidence_mining = False
    
    # Process all QAs
    gt_ans = {}
    pred_ans = {}
    total_time = 0
    
    logger.info("Starting inference...")
    logger.info("="*70)
    
    for data in tqdm(gt_data['data'], desc="Processing"):
        question = data['question']
        
        # Extract video info based on dataset format
        if 'M4-ViteVQA' in args.gt_json or 'video_id' in data:
            gt_answer = data.get('answers', data.get('answer', ""))
            vid = data['video_id']
            qid = data['question_id']
        elif 'RoadTextVQA' in args.gt_json or 'videoId' in data:
            gt_answer = data['answer']
            vid = data['videoId']
            qid = data['questionId']
            # Construct video path for RoadTextVQA format
            video_file = data.get('video', vid + '.mp4')
            video_path = os.path.join(args.video_dir, video_file)
        else:
            logger.warning(f"Unknown dataset format for QA: {data}")
            continue
        
        # Construct video path
        if 'RoadTextVQA' not in args.gt_json:
            video_path = os.path.join(args.video_dir, vid + '.mp4')
        
        # Store ground truth
        ann = {'video_id': vid, 'answer': gt_answer}
        gt_ans[qid] = ann
        
        # Inference
        start_time = time.time()
        try:
            if args.use_evidence_mining:
                frames = sample_frames_from_video(video_path, args.num_sampled_frames)
                if not frames:
                    logger.error(f"Skipping QA {qid}: failed to sample frames from {video_path}")
                    response = "Failed to process video"
                else:
                    # Use evidence mining pipeline
                    result = pipeline.evidence_pipeline.run(
                        question,
                        frames=frames,
                        top_k_frames=args.top_k_frames,
                        verbose=args.verbose
                    )
                    response = result['answer']
            else:
                # Fallback: use simple baseline (not fully implemented here)
                logger.warning("Baseline mode not fully implemented in this script")
                response = "Baseline mode not available"
        
        except Exception as e:
            logger.error(f"Error processing QA {qid}: {e}")
            response = "Error processing"
            torch.cuda.empty_cache()
            continue
        
        end_time = time.time()
        total_time += (end_time - start_time)
        
        # Clean response
        response = response.replace("Answer:", "").strip()
        if response.endswith('.'):
            response = response[:-1]
        
        # Store prediction
        p_ann = {'video_id': vid, 'answer': response}
        pred_ans[qid] = p_ann
        
        if args.verbose:
            logger.info(f"Q: {question}")
            logger.info(f"GT: {gt_answer}")
            logger.info(f"Pred: {response}\n")
        
        torch.cuda.empty_cache()
    
    # Save predictions
    logger.info("="*70)
    logger.info(f"\nSaving predictions to: {args.output}")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    with codecs.open(args.output, 'w', encoding='utf-8') as f:
        json.dump(pred_ans, f, indent=2, ensure_ascii=False)
    
    # Evaluate
    logger.info("\nEvaluating predictions...")
    with open(args.output, 'r', encoding='utf-8') as f:
        p_ans = json.load(f)
    
    try:
        anls = anls_metr._compute(predictions=p_ans, references=gt_ans)
        acc = stvqa_acc_metr._compute(predictions=p_ans, references=gt_ans)
    except Exception as e:
        logger.warning(f"Could not compute metrics: {e}")
        anls = 0.0
        acc = 0.0
    
    # Print summary
    logger.info("="*70)
    logger.info("INFERENCE SUMMARY")
    logger.info("="*70)
    logger.info(f"Mode: {mode}")
    logger.info(f"Total Questions: {len(pred_ans)}")
    logger.info(f"Accuracy: {acc:.4f}")
    logger.info(f"ANLS: {anls:.4f}")
    logger.info(f"Total Time: {total_time:.2f}s")
    avg_time = total_time / len(pred_ans) if pred_ans else 0.0
    logger.info(f"Avg Time per Question: {avg_time:.2f}s")
    logger.info(f"Output: {args.output}")
    logger.info("="*70)


if __name__ == "__main__":
    main()
