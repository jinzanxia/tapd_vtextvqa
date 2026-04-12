import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from qwen_vison_process import (  # noqa: E402
    _read_video_decord,
    detectron2_object_det,
    init_ocrmodel,
    ocr_det_with_text,
    text_density_proposals,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize clustered text proposals and object-expanded proposals.")
    parser.add_argument("--video-path", required=True, help="Input video path")
    parser.add_argument("--output-dir", required=True, help="Directory to save visualizations")
    parser.add_argument("--vts-config", required=True, help="GoMatching config path")
    parser.add_argument("--vts-model", required=True, help="GoMatching model path")
    parser.add_argument("--fps", type=float, default=1.0, help="Sampling fps for decord video reader")
    parser.add_argument("--device", type=str, default="cuda:0", help="Inference device")
    parser.add_argument("--top-k", type=int, default=2, help="Number of text-cluster proposals to visualize")
    parser.add_argument("--density-nms", type=float, default=0.5, help="Unused legacy arg kept for compatibility")
    parser.add_argument("--stride-ratio", type=float, default=0.25, help="Unused legacy arg kept for compatibility")
    parser.add_argument("--max-save", type=int, default=8, help="Max number of frames to save")
    parser.add_argument("--d2-config", type=str, default="detectron2_coco.yaml", help="Detectron2 config")
    parser.add_argument("--d2-weights", type=str, default=None, help="Detectron2 weights override")
    parser.add_argument("--d2-obj-classes", type=str, default=None, help="Comma-separated COCO class ids to keep")
    return parser.parse_args()


def build_d2_predictor(device, cfg_path, weights=None):
    from detectron2.config import get_cfg
    from detectron2.engine.defaults import DefaultPredictor

    cfg = get_cfg()
    cfg.merge_from_file(cfg_path)
    if weights:
        cfg.MODEL.WEIGHTS = weights
    cfg.MODEL.DEVICE = device
    cfg.freeze()
    return DefaultPredictor(cfg)


def expand_text_props_with_objects(proposals, object_boxes, h, w):
    if not object_boxes:
        return list(proposals)

    def iou(box_a, box_b):
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    expanded = []
    related_boxes = []
    for sx, sy, pw, ph in proposals:
        px1, py1, px2, py2 = sx, sy, sx + pw, sy + ph
        related = []
        for ob in object_boxes:
            ox1, oy1, ox2, oy2 = ob
            if iou([px1, py1, px2, py2], ob) > 0.1:
                related.append(ob)
                continue
            cx, cy = (ox1 + ox2) // 2, (oy1 + oy2) // 2
            if px1 <= cx <= px2 and py1 <= cy <= py2:
                related.append(ob)
        if related:
            ex1 = max(0, min([px1] + [b[0] for b in related]))
            ey1 = max(0, min([py1] + [b[1] for b in related]))
            ex2 = min(w, max([px2] + [b[2] for b in related]))
            ey2 = min(h, max([py2] + [b[3] for b in related]))
            expanded.append((ex1, ey1, ex2 - ex1, ey2 - ey1))
        else:
            expanded.append((px1, py1, pw, ph))
        related_boxes.append(related)
    return expanded, related_boxes


def draw_boxes(image, boxes, color, prefix, thickness=2):
    canvas = image.copy()
    for idx, (x, y, w, h) in enumerate(boxes, start=1):
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, thickness)
        cv2.putText(
            canvas,
            f"{prefix}{idx}",
            (x + 4, max(18, y + 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )
    return canvas


def draw_xyxy_boxes(image, boxes, color, thickness=1):
    canvas = image.copy()
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
    return canvas


def add_title(image, title):
    canvas = image.copy()
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(canvas, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def proposal_area(box):
    return box[2] * box[3]


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    d2_predictor = build_d2_predictor(args.device, args.d2_config, args.d2_weights)
    d2_class_ids = None
    if args.d2_obj_classes:
        d2_class_ids = [int(x) for x in args.d2_obj_classes.split(",") if x.strip().isdigit()]

    init_ocrmodel(
        cfg_path=args.vts_config,
        model_path=args.vts_model,
        device=args.device,
        model=None,
        processor=None,
    )

    video, sample_fps = _read_video_decord({"video": args.video_path, "fps": args.fps})
    text_boxes_list, text_list = ocr_det_with_text(video)
    object_boxes_list = detectron2_object_det(video, d2_predictor, class_ids=d2_class_ids)

    h, w = video.shape[1:3]
    win_sizes = [
        (int(0.4 * w), int(0.4 * h)),
        (int(0.6 * w), int(0.6 * h)),
        (int(0.8 * w), int(0.8 * h)),
    ]

    scored_frames = []
    frame_payloads = []
    for frame_idx, frame in enumerate(video):
        text_boxes = text_boxes_list[frame_idx]
        object_boxes = object_boxes_list[frame_idx]
        text_props = text_density_proposals(
            text_boxes,
            h,
            w,
            win_sizes,
            top_k=args.top_k,
            nms_thresh=args.density_nms,
            stride_ratio=args.stride_ratio,
        )
        expanded_props, related = expand_text_props_with_objects(text_props, object_boxes, h, w)
        growth = sum(proposal_area(b) for b in expanded_props) - sum(proposal_area(b) for b in text_props)
        scored_frames.append((growth, frame_idx))
        frame_payloads.append((frame, text_boxes, text_list[frame_idx], object_boxes, text_props, expanded_props, related))

    scored_frames.sort(key=lambda x: x[0], reverse=True)
    kept = [idx for growth, idx in scored_frames if growth > 0][: args.max_save]
    if not kept:
        kept = [idx for _, idx in scored_frames[: args.max_save]]

    summary_lines = [
        f"video_path: {args.video_path}",
        f"sample_fps: {sample_fps:.3f}",
        f"saved_frames: {len(kept)}",
        f"d2_obj_classes: {args.d2_obj_classes or 'all'}",
    ]

    for rank, frame_idx in enumerate(kept, start=1):
        frame, text_boxes, texts, object_boxes, text_props, expanded_props, related = frame_payloads[frame_idx]
        rgb = np.asarray(frame).copy()
        base = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        left = draw_xyxy_boxes(base, text_boxes, (0, 220, 0), thickness=2)
        left = draw_xyxy_boxes(left, object_boxes, (255, 120, 0), thickness=1)
        left = draw_boxes(left, text_props, (0, 0, 255), "T")
        left = add_title(left, f"Frame {frame_idx}: top-{len(text_props)} text clusters")

        right = draw_xyxy_boxes(base, text_boxes, (0, 220, 0), thickness=2)
        right = draw_xyxy_boxes(right, object_boxes, (255, 120, 0), thickness=1)
        right = draw_boxes(right, expanded_props, (0, 255, 255), "O")
        right = add_title(right, f"Frame {frame_idx}: object-expanded clusters")

        merged = cv2.hconcat([left, right])
        out_path = os.path.join(args.output_dir, f"frame_{rank:02d}_idx_{frame_idx:04d}.jpg")
        cv2.imwrite(out_path, merged)

        summary_lines.append(
            f"frame={frame_idx} text_props={len(text_props)} object_boxes={len(object_boxes)} "
            f"texts={texts[:5]} output={os.path.basename(out_path)}"
        )
        for prop_idx, (t_prop, e_prop, rel) in enumerate(zip(text_props, expanded_props, related), start=1):
            summary_lines.append(
                f"  prop{prop_idx}: text={t_prop} expanded={e_prop} related_objects={len(rel)}"
            )

    with open(os.path.join(args.output_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")


if __name__ == "__main__":
    main()
