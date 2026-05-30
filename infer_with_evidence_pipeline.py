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
INFER_CODES_DIR = SCRIPT_DIR / "infer_codes"
if str(INFER_CODES_DIR) not in sys.path:
    sys.path.insert(0, str(INFER_CODES_DIR))

PIPELINE_AVAILABLE = False
EvidenceMiningPipeline = None
route_question = None
try:
    pipeline_module = importlib.import_module("pipeline.evidence_pipeline")
    EvidenceMiningPipeline = getattr(pipeline_module, "EvidenceMiningPipeline")
    route_question = getattr(pipeline_module, "route_question")
    PIPELINE_AVAILABLE = True
except Exception as e:
    logger.warning("Could not import evidence pipeline module.")
    logger.warning(f"Import error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

try:
    from qwen_vison_process import process_vision_info, init_ocrmodel, set_key_conf
except Exception as e:
    process_vision_info = None
    init_ocrmodel = None
    set_key_conf = None
    logger.warning(f"Could not import SFA vision pipeline: {type(e).__name__}: {e}")

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


def run_sfa_global_reasoning(question, video_path, model, processor):
    """Run the original SFA fixed-crop video route for global questions."""
    if process_vision_info is None:
        raise RuntimeError("SFA vision pipeline is not available")

    prompt = 'Please provide a brief answer based on the video, using as few words as possible. Question: ' + question
    conversation = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_path, "fps": 1.0},
                {"type": "text", "text": prompt},
            ]
        },
    ]

    image_inputs, video_inputs, video_kwargs, _ = process_vision_info(
        question,
        conversation,
        return_video_kwargs=True,
        d2_predictor=None,
        d2_class_ids=None,
    )

    text = processor.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    )
    inputs = inputs.to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
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
    return processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0].strip()


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
    parser.add_argument(
        "--ocr-score-mode",
        choices=["paddle", "vlm"],
        default="paddle",
        help="OCR readability scoring backend for local evidence crops (default: paddle)"
    )
    parser.add_argument(
        "--global-route",
        choices=["sfa", "evidence"],
        default="sfa",
        help="Route global questions to original SFA video inference or evidence top-frame inference (default: sfa)"
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
    logger.info(f"Global Route: {args.global_route}")
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
            pipeline = EvidenceMiningPipeline(
                model=model,
                processor=processor,
                device=device,
                ocr_score_mode=args.ocr_score_mode
            )
            logger.info("Pipeline initialized\n")
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {e}")
            logger.warning("Falling back to baseline mode")
            args.use_evidence_mining = False

    if args.use_evidence_mining and args.global_route == "sfa":
        if init_ocrmodel is None or set_key_conf is None:
            logger.error("SFA global route requested but SFA modules are unavailable")
            return
        logger.info("Initializing SFA global route...")
        set_key_conf(
            w_size=0.6,
            thrd=0.7,
            focus_bonus=False,
            layout_zoom="off",
            crop_mode="fixed",
            density_top_k=0,
            density_nms=0.5,
        )
        init_ocrmodel(
            cfg_path=args.vts_config,
            model_path=args.vts_model,
            device=device,
            model=model,
            processor=processor,
        )
        logger.info("SFA global route initialized\n")
    
    # Process all QAs
    gt_ans = {}
    pred_ans = {}
    total_time = 0
    route_counts = {"local": 0, "global": 0, "unknown": 0}
    pipeline_failures = 0
    
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
                question_type = route_question(question) if route_question else "local"
                route_counts[question_type] = route_counts.get(question_type, 0) + 1

                if question_type == "global" and args.global_route == "sfa":
                    response = run_sfa_global_reasoning(
                        question,
                        video_path,
                        model=model,
                        processor=processor,
                    )
                else:
                    frames = sample_frames_from_video(video_path, args.num_sampled_frames)
                    if not frames:
                        logger.error(f"Skipping QA {qid}: failed to sample frames from {video_path}")
                        response = "Failed to process video"
                        pipeline_failures += 1
                        result = None
                    else:
                        result = pipeline.run(
                            question,
                            frames=frames,
                            top_k_frames=args.top_k_frames,
                            verbose=args.verbose
                        )

                    if result is None:
                        pass
                    elif not result.get("success", False):
                        pipeline_failures += 1
                        response = result['answer']
                    else:
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
    if args.use_evidence_mining:
        logger.info(f"Route Counts: {route_counts}")
        logger.info(f"Pipeline Failures: {pipeline_failures}")
    logger.info(f"Total Time: {total_time:.2f}s")
    avg_time = total_time / len(pred_ans) if pred_ans else 0.0
    logger.info(f"Avg Time per Question: {avg_time:.2f}s")
    logger.info(f"Output: {args.output}")
    logger.info("="*70)


if __name__ == "__main__":
    main()
