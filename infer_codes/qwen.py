import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import json
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from qwen_vison_process import process_vision_info, init_ocrmodel, set_key_conf, format_ocr_prompt, should_inject_ocr, collect_ocr_texts, ocr_post_correct
from metric import anls_metric, stvqa_acc_metric
import codecs
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import time
import argparse

WIN_SIZE = 0.6
THRESHOLD = 0.7

def get_parser():
    parser = argparse.ArgumentParser(description="builtin configs")
    parser.add_argument("--gt-json", help="gt json file path",)
    parser.add_argument("--model-name", help="video-llm path",)
    parser.add_argument("--vts-config", help="VTS model config file path")
    parser.add_argument("--vts-model", help="VTS model path")
    parser.add_argument("--video-dir", help="input video dir path")
    parser.add_argument("--output", help="output json path")
    parser.add_argument("--use-ocr-text", action="store_true", default=True, help="inject OCR text into VLM prompt")
    parser.add_argument("--no-ocr-text", dest="use_ocr_text", action="store_false", help="disable OCR text injection")
    parser.add_argument("--conditional-ocr", action="store_true", default=False, help="only inject OCR for text/number questions")
    parser.add_argument("--max-ocr-chars", type=int, default=500, help="max chars for OCR text in prompt")
    parser.add_argument("--ocr-top-k", type=int, default=5, help="keep top-k most frequent OCR texts")
    parser.add_argument("--ocr-min-freq", type=int, default=2, help="min frames a text must appear in")
    parser.add_argument("--use-focus-bonus", action="store_true", default=True, help="text matching bonus in Focus stage")
    parser.add_argument("--no-focus-bonus", dest="use_focus_bonus", action="store_false", help="disable Focus text matching bonus")
    parser.add_argument("--layout-zoom", type=str, default="off", choices=["off", "centroid", "center", "full"],
                        help="layout-guided zoom mode: off, centroid (text centroid only), center (frame center only), full (centroid + frame center)")
    parser.add_argument("--kf-sample", type=str, default="off", choices=["off", "center", "sharpness", "random"],
                        help="keyframe pre-sampling mode: off, center, sharpness (Laplacian), random")
    parser.add_argument("--kf-segments", type=int, default=8, help="number of temporal segments for keyframe sampling")
    parser.add_argument("--kf-neighbors", type=int, default=1, help="neighbor expansion radius k around each keyframe")
    parser.add_argument("--ocr-post-correct", action="store_true", default=False, help="post-correct VLM answer using OCR edit distance")
    parser.add_argument("--ocr-pc-max-edit", type=int, default=2, help="max edit distance for OCR post-correction")
    parser.add_argument("--ocr-pc-top-k", type=int, default=20, help="top-k OCR texts for post-correction pool")
    parser.add_argument("--ocr-pc-min-freq", type=int, default=1, help="min frame freq for post-correction OCR pool")
    return parser

if __name__ == "__main__":
    args = get_parser().parse_args()
    gt_json = args.gt_json

    save_json = args.output

    set_key_conf(w_size=WIN_SIZE, thrd=THRESHOLD, focus_bonus=args.use_focus_bonus, layout_zoom=args.layout_zoom,
                 kf_sample=args.kf_sample, kf_n_segments=args.kf_segments, kf_neighbors=args.kf_neighbors)

    anls_metr = anls_metric.ANLS_metric()
    stvqa_acc_metr = stvqa_acc_metric.STVQAAcc_metric()

    device = "cuda:0" # auto cuda:0
    torch.cuda.set_device(device)

    model_path = args.model_name
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        device_map=device,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        # resume_download=True,
    )
    processor = AutoProcessor.from_pretrained(model_path)


    init_ocrmodel(cfg_path=args.vts_config, model_path=args.vts_model, device=device, model=model, processor=processor)

    print_cnt = []

    print(gt_json, save_json)
    gt_ans = {}
    pred_ans = {}
    with open(gt_json, 'r', encoding='utf-8') as f:
        gt = json.load(f)
        f.close()
    total_time = 0
    for data in tqdm(gt['data']):
        question = data['question']

        ### for m4-vitevqa
        if 'M4-ViteVQA' in gt_json:
            gt_answer = data['answers']
            vid = data['video_id']
            qid = data['question_id']
            video_dir = args.video_dir
            video_path = os.path.join(video_dir, vid + '.mp4')

        ### for roadtextvqa
        if 'RoadTextVQA' in gt_json:
            gt_answer = data['answer']
            vid = data['videoId']
            qid = data['questionId']
            video_dir = args.video_dir
            video_path = os.path.join(video_dir, data['video'])

        ann = {'video_id': vid, 'answer': gt_answer}
        gt_ans[qid] = ann

        promt = 'Please provide a brief answer based on the video, using as few words as possible. Question: ' + question
        conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video_path, "fps": 1.0},
                    {"type": "text", "text": promt},
                ]
            },
        ]

        image_inputs, video_inputs, video_kwargs, all_text_lists = process_vision_info(question, conversation, return_video_kwargs=True)

        # rebuild prompt with OCR text if available
        ocr_prefix = ''
        inject = args.use_ocr_text and (not args.conditional_ocr or should_inject_ocr(question))
        if inject and all_text_lists:
            for tl in all_text_lists:
                ocr_prefix += format_ocr_prompt(tl, max_chars=args.max_ocr_chars, top_k=args.ocr_top_k, min_freq=args.ocr_min_freq)
        if ocr_prefix:
            promt = ocr_prefix + promt
            conversation[1]["content"][-1]["text"] = promt
            print(f'[OCR DEBUG] Q: {question}')
            print(f'[OCR DEBUG] OCR prefix: {ocr_prefix.strip()}')

        text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        )
        inputs = inputs.to(model.device)
        start_time = time.time()
        try:
            output_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False, temperature=0, num_beams=1,)
            generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, output_ids)]
            response = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
        except:
            response = "Unanswerable."
        end_time = time.time()
        total_time += (end_time - start_time)

        response = response.replace("Answer:", "").strip()
        if response.endswith('.'):
            response = response[:-1]

        if args.ocr_post_correct and all_text_lists:
            ocr_pool = []
            for tl in all_text_lists:
                ocr_pool.extend(collect_ocr_texts(tl, top_k=args.ocr_pc_top_k, min_freq=args.ocr_pc_min_freq))
            original = response
            response = ocr_post_correct(response, ocr_pool, max_edit_dist=args.ocr_pc_max_edit)
            if response != original:
                print(f'[OCR CORRECT] "{original}" -> "{response}"')

        p_ann = {'video_id': vid, 'answer': response}
        pred_ans[qid] = p_ann
        print('GT: ', gt_answer, '  Pred: ', response)
        torch.cuda.empty_cache()

    json_fp = codecs.open(save_json, 'w', encoding='utf-8')  # use codecs to speed up dump
    json_str = json.dumps(pred_ans, indent=2, ensure_ascii=False)
    json_fp.write(json_str)
    json_fp.close()

    del pred_ans
    with open(save_json, 'r', encoding='utf-8') as f1:
        p_ans = json.load(f1)
        f1.close()
    anls = anls_metr._compute(predictions=p_ans, references=gt_ans)
    acc = stvqa_acc_metr._compute(predictions=p_ans, references=gt_ans)

    filename = save_json.split('.')[0].split('/')[-1] 
    cont = filename + ' ACC: ' +  str(acc) + ' ANLS: ' + str(anls) + ' Time: ' + str(total_time)
    print(cont)
