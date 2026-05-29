"""
Quick Reference Guide for Evidence Mining Pipeline.

Copy-paste friendly code snippets for common use cases.
"""

# ============================================================================
# 1. BASIC USAGE - Run full pipeline with 16 frames
# ============================================================================

def basic_example():
    """Minimal example - run full pipeline."""
    import cv2
    from pipeline.evidence_pipeline import run_pipeline
    
    # Load frames from video
    video_path = "video.mp4"
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < 16:
        ret, frame = cap.read()
        if not ret: break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    
    # Run pipeline
    question = "What does the blue sign say?"
    result = run_pipeline(question, frames)
    
    print(f"Answer: {result['answer']}")


# ============================================================================
# 2. BATCH PROCESSING - Process multiple QAs from dataset
# ============================================================================

def batch_example():
    """Process multiple questions with evidence mining."""
    import json
    from pipeline_integration_example import IntegratedVideoQAPipeline
    
    # Load QA data
    with open("qa_data.json") as f:
        qa_list = json.load(f)["data"]
    
    # Initialize pipeline
    pipeline = IntegratedVideoQAPipeline(
        model_path="Qwen/Qwen2.5-VL-7B-Instruct",
        device="cuda:0"
    )
    
    # Process all QAs
    predictions = pipeline.process_video_qas(
        qa_data=qa_list,
        video_dir="./videos",
        output_json="predictions.json",
        use_evidence_mining=True,
        num_sampled_frames=16,
        top_k_frames=5,
        verbose=True
    )


# ============================================================================
# 3. STAGE-BY-STAGE INSPECTION - Debug individual stages
# ============================================================================

def stage_by_stage_example():
    """Inspect each pipeline stage individually."""
    import cv2
    from PIL import Image
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
    
    # Load frames
    video_path = "video.mp4"
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < 16:
        ret, frame = cap.read()
        if not ret: break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame_rgb))
    cap.release()
    
    question = "What does the blue sign say?"
    
    # Stage 1: Parse question
    print("=== Stage 1: Question Parsing ===")
    parsed_q = parse_question(question)
    print(f"Parsed: {parsed_q}\n")
    
    # Stage 2: Retrieve frames
    print("=== Stage 2: Frame Retrieval ===")
    retrieval_prompt = build_frame_retrieval_prompt(parsed_q)
    print(f"Retrieval prompt: {retrieval_prompt}")
    top_frames = retrieve_relevant_frames(frames, retrieval_prompt, top_k=5)
    print(f"Retrieved {len(top_frames)} frames\n")
    for i, f in enumerate(top_frames):
        print(f"  Frame {i}: score={f['score']:.3f}")
    print()
    
    # Stage 3: Localize regions
    print("=== Stage 3: Region Localization ===")
    region_prompt = build_region_localization_prompt(parsed_q)
    print(f"Region prompt: {region_prompt}")
    regions = localize_target_regions(top_frames, region_prompt)
    print(f"Localized {len(regions)} candidate regions\n")
    for i, r in enumerate(regions[:3]):
        print(f"  Region {i}: score={r['combined_score']:.3f}, bbox={r['bbox']}")
    print()
    
    # Stage 4: Score visibility
    print("=== Stage 4: OCR Visibility Scoring ===")
    ocr_prompt = build_ocr_visibility_prompt(parsed_q)
    visibility = score_crop_visibility(regions, ocr_prompt)
    if visibility['success']:
        print(f"Best crop scores:")
        for k, v in visibility['best_scores'].items():
            print(f"  {k}: {v:.3f}")
    print()
    
    # Stage 5-6: Reasoning
    print("=== Stage 5-6: Final Reasoning ===")
    answer = run_vlm_reasoning(
        question,
        global_frame=top_frames[0]['frame'],
        local_crop=visibility['best_crop'] if visibility['success'] else None
    )
    print(f"Answer: {answer}")


# ============================================================================
# 4. CUSTOM CONFIGURATION - Adjust pipeline parameters
# ============================================================================

def custom_config_example():
    """Run pipeline with custom configuration."""
    import cv2
    from pipeline.evidence_pipeline import EvidenceMiningPipeline
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    import torch
    
    # Load model with custom settings
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        device_map="cuda:0",
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    
    # Create pipeline with custom device
    pipeline = EvidenceMiningPipeline(
        model=model,
        processor=processor,
        device="cuda:0"
    )
    
    # Load frames
    video_path = "video.mp4"
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < 32:  # More frames for better retrieval
        ret, frame = cap.read()
        if not ret: break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    
    # Run with custom parameters
    question = "What does the blue sign say?"
    result = pipeline.run(
        question,
        frames,
        top_k_frames=10,  # Retrieve more frames
        verbose=True      # Print debug info
    )
    
    print(f"Final Answer: {result['answer']}")


# ============================================================================
# 5. COMPARING WITH BASELINE - Run both methods
# ============================================================================

def comparison_example():
    """Compare evidence mining vs baseline single-frame."""
    import cv2
    import time
    from pipeline.evidence_pipeline import run_pipeline
    from pipeline_integration_example import IntegratedVideoQAPipeline
    
    # Load model
    pipeline = IntegratedVideoQAPipeline(
        model_path="Qwen/Qwen2.5-VL-7B-Instruct",
        device="cuda:0"
    )
    
    # Load frames
    video_path = "video.mp4"
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < 16:
        ret, frame = cap.read()
        if not ret: break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    
    question = "What does the blue sign say?"
    
    # Method 1: Evidence Mining
    print("=== Method 1: Evidence Mining ===")
    start = time.time()
    result_evidence = run_pipeline(question, frames)
    time_evidence = time.time() - start
    print(f"Answer: {result_evidence['answer']}")
    print(f"Time: {time_evidence:.2f}s\n")
    
    # Method 2: Baseline (single frame)
    print("=== Method 2: Baseline (Single Frame) ===")
    start = time.time()
    answer_baseline = pipeline._simple_inference(question, frames[8])  # Middle frame
    time_baseline = time.time() - start
    print(f"Answer: {answer_baseline}")
    print(f"Time: {time_baseline:.2f}s\n")
    
    print(f"Evidence mining is {time_evidence/time_baseline:.1f}x the cost")


# ============================================================================
# 6. HANDLING SPECIAL CASES
# ============================================================================

def special_cases_example():
    """Handle edge cases and special scenarios."""
    from pipeline.evidence_pipeline import run_pipeline
    import cv2
    
    # Case 1: Very short video
    print("=== Case 1: Short Video ===")
    video_path = "short_video.mp4"
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    
    if len(frames) < 5:
        print(f"Only {len(frames)} frames available, adjusting...")
        result = run_pipeline("Question", frames, verbose=True)
    
    # Case 2: OCR-heavy question
    print("\n=== Case 2: OCR-Heavy Question ===")
    ocr_question = "What is the license plate number?"
    result = run_pipeline(ocr_question, frames)
    print(f"Answer: {result['answer']}")
    
    # Case 3: Non-OCR spatial question
    print("\n=== Case 3: Spatial Question ===")
    spatial_question = "Is the car on the left or right side?"
    result = run_pipeline(spatial_question, frames)
    print(f"Answer: {result['answer']}")


# ============================================================================
# 7. DEBUGGING - Enable verbose logging
# ============================================================================

def debugging_example():
    """Enable detailed logging for debugging."""
    import logging
    import cv2
    from pipeline.evidence_pipeline import run_pipeline
    
    # Set up logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(name)s - %(levelname)s - %(message)s'
    )
    
    # Load frames
    video_path = "video.mp4"
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < 8:
        ret, frame = cap.read()
        if not ret: break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    
    # Run with verbose logging
    question = "What does the sign say?"
    result = run_pipeline(question, frames, verbose=True)
    
    # Check for errors
    if not result.get('success'):
        print(f"Error: {result.get('error')}")


# ============================================================================
# 8. SAVING INTERMEDIATE RESULTS
# ============================================================================

def save_intermediate_example():
    """Save visualizations of each stage."""
    import cv2
    import os
    from PIL import Image
    from pipeline.evidence_pipeline import EvidenceMiningPipeline
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    
    # Create output directory
    os.makedirs("debug_output", exist_ok=True)
    
    # Load model and pipeline
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        device_map="cuda:0",
        torch_dtype=torch.float16,
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    pipeline = EvidenceMiningPipeline(model=model, processor=processor)
    
    # Load frames
    video_path = "video.mp4"
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < 16:
        ret, frame = cap.read()
        if not ret: break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    
    # Run pipeline
    question = "What does the sign say?"
    result = pipeline.run(question, frames, verbose=True)
    
    # Save retrieved frames
    for i, frame_dict in enumerate(result['retrieval_results'][:3]):
        frame = frame_dict['frame']
        frame.save(f"debug_output/01_retrieved_frame_{i}_score_{frame_dict['score']:.3f}.jpg")
    
    # Save localized regions
    for i, region_dict in enumerate(result['localization_results'][:3]):
        region_frame = region_dict['frame']
        bbox = region_dict['bbox']
        # Draw bbox on frame
        frame_np = cv2.cvtColor(cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2RGB)
        frame_np = cv2.rectangle(
            frame_np,
            (int(bbox['x1']), int(bbox['y1'])),
            (int(bbox['x2']), int(bbox['y2'])),
            (0, 255, 0), 2
        )
        Image.fromarray(frame_np).save(f"debug_output/02_localized_region_{i}.jpg")
    
    # Save best crop
    if result['visibility_results'] and result['visibility_results'].get('success'):
        best_crop = result['visibility_results']['best_crop']
        best_crop.save("debug_output/03_best_crop_ocr.jpg")
    
    print("Debug visualizations saved to debug_output/")


if __name__ == "__main__":
    print("Evidence Mining Pipeline - Quick Reference Examples")
    print("=" * 60)
    print("\nAvailable examples:")
    print("  1. basic_example() - Minimal usage")
    print("  2. batch_example() - Process dataset")
    print("  3. stage_by_stage_example() - Inspect each stage")
    print("  4. custom_config_example() - Custom parameters")
    print("  5. comparison_example() - vs baseline")
    print("  6. special_cases_example() - Edge cases")
    print("  7. debugging_example() - Debug logging")
    print("  8. save_intermediate_example() - Save visualizations")
    print("\nUncomment the example you want to run at the bottom of this file")
    
    # Uncomment to run an example:
    # basic_example()
