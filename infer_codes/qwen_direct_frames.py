import argparse
import codecs
import json
import os
import time
import warnings

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

try:
    from transformers import Qwen3VLForConditionalGeneration
except ImportError:
    Qwen3VLForConditionalGeneration = None

try:
    from peft import PeftModel
except ImportError:
    PeftModel = None

from metric import anls_metric, stvqa_acc_metric

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=UserWarning)


def get_qwen_vl_model_class(model_name: str):
    """Select the correct Hugging Face class for Qwen2.5-VL vs Qwen3-VL."""
    if "qwen3" in (model_name or "").lower():
        if Qwen3VLForConditionalGeneration is None:
            raise ImportError(
                "Qwen3-VL requires a transformers version with "
                "Qwen3VLForConditionalGeneration. Please upgrade transformers "
                "inside the container, e.g. pip install -U transformers accelerate."
            )
        return Qwen3VLForConditionalGeneration
    return Qwen2_5_VLForConditionalGeneration


def sample_frames_from_video(video_path, num_frames):
    """Sample RGB PIL frames uniformly from a video file."""
    try:
        import cv2
    except ImportError as e:
        raise ImportError("OpenCV is required for direct-frame inference.") from e

    if not os.path.exists(video_path):
        return []

    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            return []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            return []

        num_frames = max(1, min(int(num_frames), total_frames))
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
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))
        return frames
    finally:
        cap.release()


def get_video_path(data, gt_json, video_dir):
    """Resolve dataset-specific video path and metadata."""
    if "M4-ViteVQA" in gt_json or "video_id" in data:
        return {
            "gt_answer": data.get("answers", data.get("answer", "")),
            "vid": data["video_id"],
            "qid": data["question_id"],
            "video_path": os.path.join(video_dir, data["video_id"] + ".mp4"),
        }
    if "RoadTextVQA" in gt_json or "videoId" in data:
        video_file = data.get("video", data["videoId"] + ".mp4")
        return {
            "gt_answer": data["answer"],
            "vid": data["videoId"],
            "qid": data["questionId"],
            "video_path": os.path.join(video_dir, video_file),
        }
    raise ValueError(f"Unsupported dataset format for {gt_json}")


def get_parser():
    parser = argparse.ArgumentParser(description="Qwen direct-frame VideoQA inference without SFA")
    parser.add_argument("--gt-json", required=True, help="Ground truth JSON file path")
    parser.add_argument("--model-name", required=True, help="Qwen model path")
    parser.add_argument("--adapter-path", default=None, help="Optional LoRA adapter path")
    parser.add_argument("--video-dir", required=True, help="Input video directory")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--num-sampled-frames", type=int, default=8, help="Number of video frames to sample")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Max generated answer tokens")
    parser.add_argument("--verbose", action="store_true", default=False)
    return parser


def main():
    args = get_parser().parse_args()

    device = "cuda:0"
    torch.cuda.set_device(device)

    model_cls = get_qwen_vl_model_class(args.model_name)
    model = model_cls.from_pretrained(
        args.model_name,
        device_map=device,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    if args.adapter_path:
        if PeftModel is None:
            raise ImportError("Loading a LoRA adapter requires the `peft` package.")
        model = PeftModel.from_pretrained(model, args.adapter_path)
        model.eval()

    processor_path = args.adapter_path if args.adapter_path else args.model_name
    try:
        processor = AutoProcessor.from_pretrained(processor_path)
    except Exception:
        processor = AutoProcessor.from_pretrained(args.model_name)

    with open(args.gt_json, "r", encoding="utf-8") as f:
        gt = json.load(f)

    anls_metr = anls_metric.ANLS_metric()
    stvqa_acc_metr = stvqa_acc_metric.STVQAAcc_metric()
    gt_ans = {}
    pred_ans = {}
    total_time = 0.0

    for data in tqdm(gt["data"]):
        question = data["question"]
        meta = get_video_path(data, args.gt_json, args.video_dir)
        gt_ans[meta["qid"]] = {"video_id": meta["vid"], "answer": meta["gt_answer"]}

        frames = sample_frames_from_video(meta["video_path"], args.num_sampled_frames)
        if not frames:
            response = "Failed to process video"
            pred_ans[meta["qid"]] = {"video_id": meta["vid"], "answer": response}
            continue

        prompt = (
            "Please provide a brief answer based on the sampled video frames, "
            "using as few words as possible. Question: " + question
        )
        conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [{"type": "image"} for _ in frames] + [
                    {"type": "text", "text": prompt}
                ],
            },
        ]

        text = processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = processor(
            text=[text],
            images=frames,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        start_time = time.time()
        try:
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    temperature=0,
                    num_beams=1,
                )
            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, output_ids)
            ]
            response = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()
        except Exception as e:
            response = "Unanswerable."
            if args.verbose:
                print(f"Generation error for {meta['qid']}: {e}")
        total_time += time.time() - start_time

        response = response.replace("Answer:", "").strip()
        if response.endswith("."):
            response = response[:-1].strip()

        pred_ans[meta["qid"]] = {"video_id": meta["vid"], "answer": response}
        if args.verbose:
            print("Q:", question)
            print("GT:", meta["gt_answer"], " Pred:", response)
        torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with codecs.open(args.output, "w", encoding="utf-8") as f:
        f.write(json.dumps(pred_ans, indent=2, ensure_ascii=False))

    with open(args.output, "r", encoding="utf-8") as f:
        p_ans = json.load(f)
    anls = anls_metr._compute(predictions=p_ans, references=gt_ans)
    acc = stvqa_acc_metr._compute(predictions=p_ans, references=gt_ans)
    filename = args.output.split(".")[0].split("/")[-1]
    print(filename + " ACC: " + str(acc) + " ANLS: " + str(anls) + " Time: " + str(total_time))


if __name__ == "__main__":
    main()
