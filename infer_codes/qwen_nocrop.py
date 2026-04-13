import argparse
import codecs
import json
import os
import time
import warnings

import torch
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from metric import anls_metric, stvqa_acc_metric
from qwen_vison_process import process_vision_info_nocrop

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=UserWarning)


def get_parser():
    parser = argparse.ArgumentParser(description="Qwen baseline without crop/zoom/OCR")
    parser.add_argument("--gt-json", help="gt json file path")
    parser.add_argument("--model-name", help="video-llm path")
    parser.add_argument("--video-dir", help="input video dir path")
    parser.add_argument("--output", help="output json path")
    parser.add_argument("--fps", type=float, default=1.0, help="video sampling fps")
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    gt_json = args.gt_json
    save_json = args.output

    anls_metr = anls_metric.ANLS_metric()
    stvqa_acc_metr = stvqa_acc_metric.STVQAAcc_metric()

    device = "cuda:0"
    torch.cuda.set_device(device)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name,
        device_map=device,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    processor = AutoProcessor.from_pretrained(args.model_name)

    print(gt_json, save_json)
    with open(gt_json, "r", encoding="utf-8") as f:
        gt = json.load(f)

    gt_ans = {}
    pred_ans = {}
    total_time = 0.0

    for data in tqdm(gt["data"]):
        question = data["question"]

        if "M4-ViteVQA" in gt_json:
            gt_answer = data["answers"]
            vid = data["video_id"]
            qid = data["question_id"]
            video_path = os.path.join(args.video_dir, vid + ".mp4")
        elif "RoadTextVQA" in gt_json:
            gt_answer = data["answer"]
            vid = data["videoId"]
            qid = data["questionId"]
            video_path = os.path.join(args.video_dir, data["video"])
        else:
            raise ValueError(f"Unsupported dataset format for {gt_json}")

        gt_ans[qid] = {"video_id": vid, "answer": gt_answer}

        prompt = "Please provide a brief answer based on the video, using as few words as possible. Question: " + question
        conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video_path, "fps": args.fps},
                    {"type": "text", "text": prompt},
                ],
            },
        ]

        image_inputs, video_inputs, video_kwargs = process_vision_info_nocrop(
            conversation,
            return_video_kwargs=True,
        )

        text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        ).to(model.device)

        start_time = time.time()
        try:
            output_ids = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                temperature=0,
                num_beams=1,
            )
            generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, output_ids)]
            response = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()
        except Exception:
            response = "Unanswerable."
        total_time += time.time() - start_time

        response = response.replace("Answer:", "").strip()
        if response.endswith("."):
            response = response[:-1]

        pred_ans[qid] = {"video_id": vid, "answer": response}
        print("GT: ", gt_answer, "  Pred: ", response)
        torch.cuda.empty_cache()

    json_fp = codecs.open(save_json, "w", encoding="utf-8")
    json_fp.write(json.dumps(pred_ans, indent=2, ensure_ascii=False))
    json_fp.close()

    with open(save_json, "r", encoding="utf-8") as f:
        p_ans = json.load(f)
    anls = anls_metr._compute(predictions=p_ans, references=gt_ans)
    acc = stvqa_acc_metr._compute(predictions=p_ans, references=gt_ans)

    filename = save_json.split(".")[0].split("/")[-1]
    print(filename + " ACC: " + str(acc) + " ANLS: " + str(anls) + " Time: " + str(total_time))
