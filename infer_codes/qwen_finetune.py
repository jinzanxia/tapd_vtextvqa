import argparse
import json
import os
import random
import logging
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    get_scheduler,
)
try:
    from peft import LoraConfig, TaskType, get_peft_model
except ImportError:
    LoraConfig = None
    TaskType = None
    get_peft_model = None

from qwen_vison_process import (
    process_vision_info,
    init_ocrmodel,
    set_key_conf,
    should_inject_ocr,
    format_ocr_prompt,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1

    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed training requires CUDA in this script.")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    return distributed, rank, local_rank, world_size, device


def is_main_process(rank):
    return rank == 0


def parse_lora_targets(target_modules):
    return [module.strip() for module in target_modules.split(",") if module.strip()]


def apply_lora(model, args, rank):
    if not args.use_lora:
        return model
    if get_peft_model is None:
        raise ImportError("LoRA training requires the `peft` package. Please install peft in the training environment.")

    if hasattr(model, "config"):
        model.config.use_cache = False
    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if args.gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=parse_lora_targets(args.lora_target_modules),
    )
    model = get_peft_model(model, lora_config)
    if is_main_process(rank):
        model.print_trainable_parameters()
    return model


class VQAFineTuneDataset(Dataset):
    def __init__(
        self,
        gt_json,
        video_dir,
        max_samples=None,
        use_ocr_text=False,
        conditional_ocr=False,
        max_ocr_chars=500,
        ocr_top_k=5,
        ocr_min_freq=2,
    ):
        self.use_ocr_text = use_ocr_text
        self.conditional_ocr = conditional_ocr
        self.max_ocr_chars = max_ocr_chars
        self.ocr_top_k = ocr_top_k
        self.ocr_min_freq = ocr_min_freq

        with open(gt_json, "r", encoding="utf-8") as f:
            gt = json.load(f)

        samples = []
        for data in gt.get("data", []):
            if "M4-ViteVQA" in gt_json:
                question = data["question"]
                answer = data["answers"]
                if isinstance(answer, list):
                    answer = answer[0] if answer else ""
                vid = data["video_id"]
                video_path = os.path.join(video_dir, vid + ".mp4")
            elif "RoadTextVQA" in gt_json:
                question = data["question"]
                answer = data["answer"]
                vid = data["videoId"]
                video_path = os.path.join(video_dir, data["video"])
            else:
                raise ValueError(f"Unsupported dataset format for {gt_json}")

            if not os.path.isfile(video_path):
                continue

            samples.append({
                "question": question,
                "answer": answer,
                "video_path": video_path,
                "qid": data.get("question_id", data.get("questionId", None)),
            })

        if max_samples is not None:
            samples = samples[:max_samples]

        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        question = sample["question"]
        answer = sample["answer"]
        video_path = sample["video_path"]

        prompt = "Please provide a brief answer based on the video, using as few words as possible. Question: " + question
        conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video_path, "fps": 1.0},
                    {"type": "text", "text": prompt},
                ],
            },
        ]

        image_inputs, video_inputs, video_kwargs, all_text_lists = process_vision_info(
            question,
            conversation,
            return_video_kwargs=True,
            d2_predictor=None,
            d2_class_ids=None,
        )

        if self.use_ocr_text and all_text_lists and (not self.conditional_ocr or should_inject_ocr(question)):
            ocr_prefix = ""
            for tl in all_text_lists:
                ocr_prefix += format_ocr_prompt(
                    tl,
                    max_chars=self.max_ocr_chars,
                    top_k=self.ocr_top_k,
                    min_freq=self.ocr_min_freq,
                )
            if ocr_prefix:
                prompt = ocr_prefix + prompt
                conversation[1]["content"][1]["text"] = prompt

        return {
            "conversation": conversation,
            "image_inputs": image_inputs,
            "video_inputs": video_inputs,
            "video_kwargs": video_kwargs,
            "answer": answer,
            "qid": sample["qid"],
        }


class QwenDataCollator:
    def __init__(self, processor, tokenizer, max_target_length=64):
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_target_length = max_target_length

    def __call__(self, features):
        prompt_texts = []
        full_texts = []
        image_inputs = []
        video_inputs = []
        fps = []

        for feature in features:
            prompt_text = self.processor.apply_chat_template(
                feature["conversation"], tokenize=False, add_generation_prompt=True
            )
            full_conversation = feature["conversation"] + [
                {"role": "assistant", "content": [{"type": "text", "text": str(feature["answer"])}]},
            ]
            full_text = self.processor.apply_chat_template(
                full_conversation, tokenize=False, add_generation_prompt=False
            )
            prompt_texts.append(prompt_text)
            full_texts.append(full_text)
            if feature["image_inputs"] is not None:
                image_inputs.append(feature["image_inputs"][0])
            if feature["video_inputs"] is not None:
                video_inputs.append(feature["video_inputs"][0])
            if feature["video_kwargs"] is not None:
                fps.append(feature["video_kwargs"]["fps"][0])

        if len(image_inputs) == 0:
            image_inputs = None
        if len(video_inputs) == 0:
            video_inputs = None

        inputs = self.processor(
            text=full_texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            fps=fps if fps else None,
        )

        prompt_inputs = self.processor(
            text=prompt_texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            fps=fps if fps else None,
        )

        labels = inputs["input_ids"].clone()
        pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        labels = labels.masked_fill(labels == pad_token_id, -100)
        labels = labels.masked_fill(inputs["attention_mask"] == 0, -100)

        padding_side = getattr(self.tokenizer, "padding_side", "right")
        for i in range(labels.size(0)):
            full_len = int(inputs["attention_mask"][i].sum().item())
            prompt_len = int(prompt_inputs["attention_mask"][i].sum().item())
            prompt_len = min(prompt_len, full_len)
            if padding_side == "left":
                start = labels.size(1) - full_len
            else:
                start = 0
            labels[i, start : start + prompt_len] = -100

        inputs["labels"] = labels
        return inputs


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen on video QA data")
    parser.add_argument("--gt-json", required=True, help="Ground truth JSON file path")
    parser.add_argument("--video-dir", required=True, help="Video directory")
    parser.add_argument("--model-name", required=True, help="HuggingFace model name or local path")
    parser.add_argument("--vts-config", required=True, help="VTS config file for OCR/crop")
    parser.add_argument("--vts-model", required=True, help="VTS model path")
    parser.add_argument("--output-dir", required=True, help="Output checkpoint directory")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Maximum number of training samples")
    parser.add_argument("--num-train-epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--train-batch-size", type=int, default=1, help="Per-device batch size")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker processes. Keep 0 when CUDA OCR/cropping is enabled.")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--learning-rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--warmup-steps", type=int, default=0, help="Warmup steps")
    parser.add_argument("--max-target-length", type=int, default=64, help="Maximum target answer length")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--fp16", action="store_true", help="Use FP16 training")
    parser.add_argument("--use-lora", action="store_true", default=True, dest="use_lora", help="Enable LoRA adapter fine-tuning")
    parser.add_argument("--no-lora", action="store_false", dest="use_lora", help="Disable LoRA and fine-tune all model parameters")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument(
        "--lora-target-modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated module names to apply LoRA to",
    )
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True, help="Enable gradient checkpointing")
    parser.add_argument("--no-gradient-checkpointing", action="store_false", dest="gradient_checkpointing", help="Disable gradient checkpointing")
    parser.add_argument("--use-ocr-text", action="store_true", default=False, dest="use_ocr_text", help="Enable OCR text injection")
    parser.add_argument("--no-ocr-text", action="store_false", dest="use_ocr_text", help="Disable OCR text injection")
    parser.add_argument("--conditional-ocr", action="store_true", help="Inject OCR only for text/number questions")
    parser.add_argument("--max-ocr-chars", type=int, default=500)
    parser.add_argument("--ocr-top-k", type=int, default=5)
    parser.add_argument("--ocr-min-freq", type=int, default=2)
    parser.add_argument("--use-focus-bonus", action="store_true", default=False)
    parser.add_argument("--no-focus-bonus", dest="use_focus_bonus", action="store_false")
    parser.add_argument("--layout-zoom", type=str, default="off", choices=["off", "centroid", "center", "full"])
    parser.add_argument("--crop-mode", type=str, default="fixed", choices=["fixed", "density", "hybrid", "hybrid_ob", "hybrid_04", "hybrid_06", "hybrid_08"])
    parser.add_argument("--density-top-k", type=int, default=4)
    parser.add_argument("--density-nms", type=float, default=0.5)
    parser.add_argument("--cluster-expand-ratio", type=float, default=0.0)
    parser.add_argument("--cluster-min-size-ratio", type=float, default=0.0)
    parser.add_argument("--cluster-multi-scales", type=str, default=None)
    parser.add_argument("--cluster-add-density-scale", type=float, default=0.0)
    parser.add_argument("--cluster-add-density-top-k", type=int, default=1)
    parser.add_argument("--text-anchor-mode", type=str, default="off", choices=["off", "fixed", "adaptive"])
    parser.add_argument("--text-anchor-fixed-scale", type=float, default=0.4)
    parser.add_argument("--text-anchor-scales", type=str, default="0.4,0.6,0.8")
    parser.add_argument("--text-rerank-weight", type=float, default=0.0)
    parser.add_argument("--text-rerank-mode", type=str, default="off", choices=["off", "token", "cluster"])
    parser.add_argument("--d2-config", type=str, default="detectron2_coco.yaml")
    parser.add_argument("--d2-weights", type=str, default=None)
    parser.add_argument("--no-object-detect", action="store_true", default=True, help="Disable object detection during crop proposal generation")
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    distributed, rank, local_rank, world_size, device = setup_distributed()
    set_seed(args.seed)

    if is_main_process(rank):
        logger.info(f"Using device: {device} | distributed={distributed} | world_size={world_size}")

    set_key_conf(
        w_size=0.6,
        thrd=0.7,
        focus_bonus=args.use_focus_bonus,
        layout_zoom=args.layout_zoom,
        kf_sample='off',
        kf_n_segments=8,
        kf_neighbors=1,
        crop_mode=args.crop_mode,
        density_top_k=args.density_top_k,
        density_nms=args.density_nms,
        cluster_expand_ratio=args.cluster_expand_ratio,
        cluster_min_size_ratio=args.cluster_min_size_ratio,
        cluster_multi_scales=args.cluster_multi_scales,
        cluster_add_density_scale=args.cluster_add_density_scale,
        cluster_add_density_top_k=args.cluster_add_density_top_k,
        text_anchor_mode=args.text_anchor_mode,
        text_anchor_fixed_scale=args.text_anchor_fixed_scale,
        text_anchor_scales=args.text_anchor_scales,
        text_rerank_weight=args.text_rerank_weight,
        text_rerank_mode=args.text_rerank_mode,
    )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else None,
        attn_implementation="sdpa", #"flash_attention_2",
    )
    model.to(device)
    model = apply_lora(model, args, rank)
    processor = AutoProcessor.from_pretrained(args.model_name)

    init_ocrmodel(cfg_path=args.vts_config, model_path=args.vts_model, device=device, model=model, processor=processor)

    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    dataset = VQAFineTuneDataset(
        args.gt_json,
        args.video_dir,
        max_samples=args.max_train_samples,
        use_ocr_text=args.use_ocr_text,
        conditional_ocr=args.conditional_ocr,
        max_ocr_chars=args.max_ocr_chars,
        ocr_top_k=args.ocr_top_k,
        ocr_min_freq=args.ocr_min_freq,
    )
    if is_main_process(rank):
        logger.info(f"Loaded {len(dataset)} training examples")

    collator = QwenDataCollator(processor=processor, tokenizer=processor.tokenizer, max_target_length=args.max_target_length)
    train_sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True) if distributed else None

    train_dataloader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        collate_fn=collator,
        num_workers=args.num_workers,
    )

    optimizer = AdamW((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate)
    num_update_steps_per_epoch = max(1, len(train_dataloader) // args.gradient_accumulation_steps)
    num_training_steps = args.num_train_epochs * num_update_steps_per_epoch
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=num_training_steps,
    )

    scaler = torch.cuda.amp.GradScaler() if args.fp16 and device.type == "cuda" else None
    model.train()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    for epoch in range(args.num_train_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        running_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            with torch.cuda.amp.autocast(enabled=scaler is not None):
                outputs = model(**batch)
                loss = outputs.loss
                loss = loss / args.gradient_accumulation_steps

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            running_loss += loss.item() * args.gradient_accumulation_steps
            if is_main_process(rank) and (step + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{args.num_train_epochs} | step {step + 1}/{len(train_dataloader)} | loss {running_loss / (step + 1):.4f}"
                )

        if distributed:
            dist.barrier()

        if is_main_process(rank):
            checkpoint_path = output_dir / f"checkpoint-epoch-{epoch + 1}"
            model_to_save = model.module if distributed else model
            model_to_save.save_pretrained(checkpoint_path)
            processor.save_pretrained(checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")

        if distributed:
            dist.barrier()

    final_path = output_dir / "final"
    if is_main_process(rank):
        model_to_save = model.module if distributed else model
        model_to_save.save_pretrained(final_path)
        processor.save_pretrained(final_path)
        logger.info(f"Training completed. Final model saved to {final_path}")

    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
