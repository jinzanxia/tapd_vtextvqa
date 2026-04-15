from __future__ import annotations

import base64
import copy
import logging
import math
import os
import sys
import time
import warnings
from functools import lru_cache
from io import BytesIO
from typing import Optional

import requests
import torch
import torchvision
from packaging import version
from PIL import Image
from torchvision import io, transforms
from torchvision.transforms import InterpolationMode
import json
import random
import re
import numpy as np

# gomatching
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode
sys.path.insert(0, './')
from GoMatching.gomatching.config import add_gom_config
from adet.config import add_deepsolo_cfg
import cv2
import pickle

from GoMatching.gomatching.text_track_visualizer import TextTrackingVisualizer, GoMBatchPredictor
from collections import defaultdict


logger = logging.getLogger(__name__)

IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200

VIDEO_MIN_PIXELS = 128 * 28 * 28
VIDEO_MAX_PIXELS = 768 * 28 * 28
FRAME_FACTOR = 2
FPS = 2.0
FPS_MIN_FRAMES = 4
FPS_MAX_FRAMES = 768

# Set the maximum number of video token inputs.
# Here, 128K represents the maximum number of input tokens for the VLLM model.
# Remember to adjust it according to your own configuration.
VIDEO_TOTAL_PIXELS = int(float(os.environ.get('VIDEO_MAX_PIXELS', 128000 * 28 * 28 * 0.9)))
logger.info(f"set VIDEO_TOTAL_PIXELS: {VIDEO_TOTAL_PIXELS}")

### key frame zoom select
a_token_id = 32 ### 'A' token id


def detectron2_object_det(frames, predictor, class_ids=None):
    """
    frames: list of np.ndarray (H,W,3) RGB
    predictor: Detectron2 DefaultPredictor
    class_ids: list of int, 只保留这些类别的物体（可为None，保留全部）
    return: list of list of [x1, y1, x2, y2] for each frame
    """
    results = []
    for img in frames:
        outputs = predictor(img[:, :, ::-1])  # Detectron2 expects BGR
        boxes = []
        if "instances" in outputs and len(outputs["instances"]):
            inst = outputs["instances"]
            pred_boxes = inst.pred_boxes.tensor.cpu().numpy()
            pred_classes = inst.pred_classes.cpu().numpy()
            for i, box in enumerate(pred_boxes):
                if class_ids is None or pred_classes[i] in class_ids:
                    x1, y1, x2, y2 = map(int, box)
                    boxes.append([x1, y1, x2, y2])
        results.append(boxes)
    return results

def set_key_conf(w_size=0.6, thrd=0.7, focus_bonus=True, layout_zoom='off',
                 kf_sample='off', kf_n_segments=8, kf_neighbors=1,
                 crop_mode='fixed', density_top_k=4, density_nms=0.5,
                 cluster_expand_ratio=0.0, cluster_min_size_ratio=0.0,
                 cluster_multi_scales=None, cluster_add_density_scale=0.0,
                 cluster_add_density_top_k=1, text_anchor_mode='off',
                 text_anchor_fixed_scale=0.4, text_anchor_scales='0.4,0.6,0.8',
                 text_rerank_weight=0.0, text_rerank_mode='off'):
    global win_size, threshold, use_focus_bonus, use_layout_zoom
    global kf_sample_mode, kf_n_seg, kf_k
    global g_crop_mode, g_density_top_k, g_density_nms
    global g_cluster_expand_ratio, g_cluster_min_size_ratio, g_cluster_multi_scales, g_cluster_add_density_scale, g_cluster_add_density_top_k
    global g_text_anchor_mode, g_text_anchor_fixed_scale, g_text_anchor_scales, g_text_rerank_weight, g_text_rerank_mode
    win_size = w_size
    threshold = thrd
    use_focus_bonus = focus_bonus
    use_layout_zoom = layout_zoom
    kf_sample_mode = kf_sample
    kf_n_seg = kf_n_segments
    kf_k = kf_neighbors
    g_crop_mode = crop_mode
    g_density_top_k = density_top_k
    g_density_nms = density_nms
    g_cluster_expand_ratio = cluster_expand_ratio
    g_cluster_min_size_ratio = cluster_min_size_ratio
    if cluster_multi_scales:
        g_cluster_multi_scales = [float(x) for x in cluster_multi_scales.split(",") if x.strip()]
    else:
        g_cluster_multi_scales = None
    g_cluster_add_density_scale = cluster_add_density_scale
    g_cluster_add_density_top_k = cluster_add_density_top_k
    g_text_anchor_mode = text_anchor_mode
    g_text_anchor_fixed_scale = text_anchor_fixed_scale
    g_text_anchor_scales = [float(x) for x in text_anchor_scales.split(",") if x.strip()]
    g_text_rerank_weight = text_rerank_weight
    g_text_rerank_mode = text_rerank_mode


def build_text_clusters(text_boxes, frame_texts=None):
    if not text_boxes:
        return []

    boxes = [list(map(int, b)) for b in text_boxes]
    texts = frame_texts if frame_texts and len(frame_texts) == len(text_boxes) else [""] * len(text_boxes)
    n = len(boxes)
    parents = list(range(n))

    def find(x):
        while parents[x] != x:
            parents[x] = parents[parents[x]]
            x = parents[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parents[rb] = ra

    def overlap_1d(a1, a2, b1, b2):
        return max(0, min(a2, b2) - max(a1, b1))

    def should_merge(box_a, box_b):
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        aw, ah = max(1, ax2 - ax1), max(1, ay2 - ay1)
        bw, bh = max(1, bx2 - bx1), max(1, by2 - by1)
        x_overlap = overlap_1d(ax1, ax2, bx1, bx2)
        y_overlap = overlap_1d(ay1, ay2, by1, by2)
        min_h = max(1, min(ah, bh))
        min_w = max(1, min(aw, bw))

        horiz_gap = max(0, max(ax1, bx1) - min(ax2, bx2))
        same_line = y_overlap / min_h >= 0.4 and horiz_gap <= 1.5 * max(ah, bh)

        vert_gap = max(0, max(ay1, by1) - min(ay2, by2))
        same_column = x_overlap / min_w >= 0.3 and vert_gap <= 1.0 * max(ah, bh)

        inter = x_overlap * y_overlap
        area_a = aw * ah
        area_b = bw * bh
        union_area = area_a + area_b - inter
        iou = inter / union_area if union_area > 0 else 0.0
        acx, acy = (ax1 + ax2) / 2.0, (ay1 + ay2) / 2.0
        bcx, bcy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
        center_close = abs(acx - bcx) <= 2.0 * max(aw, bw) and abs(acy - bcy) <= 1.5 * max(ah, bh)
        return iou > 0 or same_line or same_column or center_close

    for i in range(n):
        for j in range(i + 1, n):
            if should_merge(boxes[i], boxes[j]):
                union(i, j)

    grouped = {}
    for idx, box in enumerate(boxes):
        root = find(idx)
        grouped.setdefault(root, {"boxes": [], "texts": []})
        grouped[root]["boxes"].append(box)
        txt = texts[idx].strip()
        if txt:
            grouped[root]["texts"].append(txt)

    clusters = []
    for item in grouped.values():
        cluster_boxes = item["boxes"]
        x1 = min(b[0] for b in cluster_boxes)
        y1 = min(b[1] for b in cluster_boxes)
        x2 = max(b[2] for b in cluster_boxes)
        y2 = max(b[3] for b in cluster_boxes)
        cluster_area = sum(max(1, (b[2] - b[0]) * (b[3] - b[1])) for b in cluster_boxes)
        union_area = max(1, (x2 - x1) * (y2 - y1))
        clusters.append({
            "box_xyxy": (x1, y1, x2, y2),
            "proposal_xywh": (x1, y1, x2 - x1, y2 - y1),
            "texts": item["texts"],
            "text": " ".join(item["texts"]).strip(),
            "score": (len(cluster_boxes), cluster_area / union_area, cluster_area, -y1, -x1),
        })
    clusters.sort(key=lambda c: c["score"], reverse=True)
    return clusters


def keyframe_sample(video):
    """Pre-filter frames via temporal segmentation + sharpness-based selection.

    Returns indices of selected frames.
    """
    T = video.shape[0]
    if T <= kf_n_seg:
        return list(range(T))

    seg_len = T / kf_n_seg
    representatives = []

    for i in range(kf_n_seg):
        start = int(i * seg_len)
        end = min(int((i + 1) * seg_len), T)

        if kf_sample_mode == 'center':
            t_i = (start + end) // 2
        elif kf_sample_mode == 'sharpness':
            best_score, best_idx = -1, start
            for idx in range(start, end):
                gray = cv2.cvtColor(video[idx], cv2.COLOR_RGB2GRAY)
                score = cv2.Laplacian(gray, cv2.CV_64F).var()
                if score > best_score:
                    best_score = score
                    best_idx = idx
            t_i = best_idx
        else:  # 'random'
            t_i = random.randint(start, end - 1)

        representatives.append(t_i)

    # neighbor expansion
    selected = set()
    for t_i in representatives:
        for offset in range(-kf_k, kf_k + 1):
            selected.add(max(0, min(T - 1, t_i + offset)))

    return sorted(selected)


def should_inject_ocr(question):
    """Decide whether OCR text is likely useful for this question.

    Returns True for text/number-reading questions, False for purely visual ones.
    """
    ql = question.lower()
    # positive signals: question asks about text, numbers, labels, etc.
    inject_kw = [
        'written', 'write', 'text', 'word', 'brand', 'name of', 'named',
        'title', 'say', 'said', 'read', 'spell', 'letter', 'sign', 'label',
        'slogan', 'logo', 'called', 'message', 'headline', 'caption',
        'price', 'cost', 'number', 'how much', 'how many', 'score',
        'total', 'amount', 'plate', 'date', 'time', 'year', 'floor',
        'channel', 'speed', 'limit', 'billboard', 'banner', 'display',
        'screen', 'subtitle', 'section', 'line of', 'shown', 'printed',
        'team', 'play for', 'result', 'input', 'enter',
    ]
    # negative signals: purely visual questions
    reject_kw = [
        'color', 'colour', 'wear', 'wearing', 'look like', 'looks like',
        'expression', 'emotion', 'feeling', 'gesture', 'posture',
        'how old', 'tall', 'shape',
    ]
    for kw in reject_kw:
        if kw in ql:
            return False
    for kw in inject_kw:
        if kw in ql:
            return True
    # default: inject (more questions benefit from OCR than not)
    return True


def format_ocr_prompt(text_lists, max_chars=500, top_k=5, min_freq=2):
    """Format OCR text from all frames into a deduplicated prompt string.
    
    Args:
        text_lists: list[list[str]], OCR texts per frame from ocr_det_with_text()
        max_chars: maximum characters for the OCR portion of the prompt
        top_k: keep only the top-k most frequent texts
        min_freq: minimum number of frames a text must appear in
    Returns:
        str: formatted prompt string, empty string if no valid text found
    """
    # count occurrences across frames as a proxy for confidence
    text_counts = defaultdict(int)
    for frame_texts in text_lists:
        frame_seen = set()
        for t in frame_texts:
            t_stripped = t.strip()
            if len(t_stripped) >= 3 and t_stripped not in frame_seen:
                frame_seen.add(t_stripped)
                text_counts[t_stripped] += 1
    if not text_counts:
        return ''
    # frequency filter: only keep texts appearing in >= min_freq frames
    reliable = {t: c for t, c in text_counts.items() if c >= min_freq}
    if not reliable:
        # fallback: keep all if nothing survives the threshold
        reliable = text_counts
    # sort by frequency (descending) and take top-k
    sorted_texts = sorted(reliable.keys(), key=lambda x: reliable[x], reverse=True)[:top_k]
    text_str = ', '.join(f'"{t}"' for t in sorted_texts)
    if len(text_str) > max_chars:
        text_str = text_str[:max_chars].rsplit(',', 1)[0]
    return f'Hint: the video may contain these texts: {text_str}\n'


def collect_ocr_texts(text_lists, top_k=20, min_freq=1):
    """Return deduplicated OCR texts sorted by frame frequency (descending)."""
    text_counts = defaultdict(int)
    for frame_texts in text_lists:
        frame_seen = set()
        for t in frame_texts:
            t_stripped = t.strip()
            if len(t_stripped) >= 2 and t_stripped not in frame_seen:
                frame_seen.add(t_stripped)
                text_counts[t_stripped] += 1
    if not text_counts:
        return []
    reliable = {t: c for t, c in text_counts.items() if c >= min_freq}
    if not reliable:
        reliable = text_counts
    return sorted(reliable.keys(), key=lambda x: reliable[x], reverse=True)[:top_k]


def ocr_post_correct(vlm_answer, ocr_texts, max_edit_dist=2):
    """Post-correct VLM answer by matching against OCR texts via edit distance.

    Only replaces when VLM answer is close-but-not-identical to an OCR text.
    """
    import editdistance
    if not ocr_texts or not vlm_answer:
        return vlm_answer
    vlm_lower = vlm_answer.lower().strip()
    best_match, best_dist = None, float('inf')
    for t in ocr_texts:
        d = editdistance.eval(vlm_lower, t.lower().strip())
        if d < best_dist:
            best_dist = d
            best_match = t
    # dynamic threshold: allow ~20% difference, at least max_edit_dist
    threshold = max(max_edit_dist, len(vlm_lower) // 5)
    if 0 < best_dist <= threshold:
        return best_match
    return vlm_answer


def setup_cfg(cfg_path, model_path, device):
    global CTLABELS, voc_size
    cfg = get_cfg()
    add_deepsolo_cfg(cfg)
    add_gom_config(cfg)
    cfg.merge_from_file(cfg_path)
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.ASSO_HEAD.ASSO_THRESH_TEST = cfg.MODEL.TRANSFORMER.INFERENCE_TH_TEST
    cfg.VIDEO_TEST.MIN_TRACK_LEN = 1
    cfg.MODEL.DEVICE = device
    cfg.freeze()
    with open(cfg.MODEL.TRANSFORMER.CUSTOM_DICT, 'rb') as fp:
        CTLABELS = pickle.load(fp)
    voc_size = cfg.MODEL.TRANSFORMER.VOC_SIZE
    return cfg

def ctc_decode_recognition(rec):
    global CTLABELS, voc_size
    last_char = '###'
    s = ''
    for c in rec:
        c = int(c)
        if c < voc_size - 1:
            if last_char != c:
                s += str(chr(CTLABELS[c]))
                last_char = c
        else:
            last_char = '###'
    return s

def init_ocrmodel(cfg_path, model_path, device, model, processor):
    global video_text_spotter, tracker_visualizer, vlm_model, vlm_processor
    cfg = setup_cfg(cfg_path, model_path, device)
    video_text_spotter = GoMBatchPredictor(cfg)
    det_device = torch.device(device)
    for name, param in video_text_spotter.model.named_parameters():
        assert param.device == det_device, f"param {name} on {param.device}"
    for name, buf in video_text_spotter.model.named_buffers():
        assert buf.device == det_device, f"buffer {name} on {buf.device}"
    metadata = MetadataCatalog.get("__unused")
    instance_mode = ColorMode.IMAGE
    tracker_visualizer = TextTrackingVisualizer(metadata, cfg, instance_mode)
    vlm_model = model
    vlm_processor = processor
    # 兼容Qwen2.5-VL Flash Attention，强制左padding
    if hasattr(vlm_processor, 'tokenizer'):
        vlm_processor.tokenizer.padding_side = 'left'
    

def ocr_det(video):
    global video_text_spotter, tracker_visualizer
    ### ocr phase
    frames_batch = defaultdict(list)
    for idx, frame in enumerate(video):
        frames_batch[idx // 100].append(frame[:, :, ::-1]) # 100 to BGR
    
    time_cost = {'total_time': 0, 'pre_process': 0, 'backbone': 0, 'detector': 0, 'rescore': 0, 'tracker': 0, 'long_match': 0, 'short_match': 0, 'post_process': 0}
    instances = []
    last_batch = False
    id_count = 0
    for batch_id in range(len(frames_batch)):
        frames = frames_batch[batch_id]
        if batch_id == len(frames_batch) - 1:
            last_batch = True
        instances, id_count = video_text_spotter(frames, instances, batch_id, id_count, last_batch, time_cost, return_time=False)
    
    h, w = video.shape[1:3]
    box_list = []
    for frame_id, (frame, prediction) in enumerate(zip(frames_batch[0], instances)):
        prediction = tracker_visualizer.pre_vis_process(prediction["instances"].to('cpu'))
        boxes = []
        ins_polys = prediction.polys
        for poly in ins_polys:
            x, y, w, h = cv2.boundingRect(poly)
            if w < 5 or h < 5:
                continue
            boxes.append([x, y, x + w, y + h])
        box_list.append(boxes)
        
    
    return box_list

def ocr_det_with_text(video):
    global video_text_spotter, tracker_visualizer
    ### ocr phase
    frames_batch = defaultdict(list)
    for idx, frame in enumerate(video):
        frames_batch[idx // 100].append(frame[:, :, ::-1]) # 100 to BGR
    
    time_cost = {'total_time': 0, 'pre_process': 0, 'backbone': 0, 'detector': 0, 'rescore': 0, 'tracker': 0, 'long_match': 0, 'short_match': 0, 'post_process': 0}
    instances = []
    last_batch = False
    id_count = 0
    for batch_id in range(len(frames_batch)):
        frames = frames_batch[batch_id]
        if batch_id == len(frames_batch) - 1:
            last_batch = True
        instances, id_count = video_text_spotter(frames, instances, batch_id, id_count, last_batch, time_cost, return_time=False)
    
    h, w = video.shape[1:3]
    box_list = []
    text_list = []
    for frame_id, (frame, prediction) in enumerate(zip(frames_batch[0], instances)):
        prediction = tracker_visualizer.pre_vis_process(prediction["instances"].to('cpu'))
        boxes = []
        texts = []
        ins_polys = prediction.polys
        recs = prediction.recs
        for poly, rec in zip(ins_polys, recs):
            x, y, w, h = cv2.boundingRect(poly)
            if w < 5 or h < 5:
                continue
            boxes.append([x, y, x + w, y + h])
            texts.append(ctc_decode_recognition(rec))
        box_list.append(boxes)
        text_list.append(texts)
    
    return box_list, text_list
    
def text_density_proposals(text_boxes, h, w, win_sizes, top_k=4, nms_thresh=0.5, stride_ratio=0.25):
    """Generate proposals from clustered text boxes instead of sliding windows.

    Args:
        text_boxes: list of [x1, y1, x2, y2] bboxes from GoMatching
        h, w: frame dimensions
        win_sizes, nms_thresh, stride_ratio: kept for backward compatibility
        top_k: max number of clustered text proposals
    Returns:
        list of (sx, sy, win_w, win_h) proposals sorted by cluster score descending
    """
    if not text_boxes:
        return []

    proposals = []
    clusters = build_text_clusters(text_boxes)
    for cluster in clusters[:top_k]:
        proposals.append(cluster["proposal_xywh"])
    return proposals


def build_text_items(text_boxes, frame_texts):
    if not text_boxes or not frame_texts:
        return []
    items = []
    for box, text in zip(text_boxes, frame_texts):
        text = (text or "").strip()
        if not text:
            continue
        items.append({
            "box_xyxy": tuple(map(int, box)),
            "text": text,
        })
    return items


def get_text_rerank_items(text_boxes, frame_texts, mode):
    if mode == 'token':
        return build_text_items(text_boxes, frame_texts)
    if mode == 'cluster':
        clusters = build_text_clusters(text_boxes, frame_texts)
        return [{"box_xyxy": c["box_xyxy"], "text": c["text"]} for c in clusters if c["text"]]
    return []


def score_text_relevance_with_qwen(question, texts, mode):
    global vlm_model, vlm_processor, a_token_id
    if not texts:
        return []

    cache = globals().setdefault("_text_rerank_cache", {})
    cached_scores = {}
    uncached = []
    prompt_texts = []

    for text in texts:
        clean_text = (text or "").strip()
        if not clean_text:
            cached_scores[text] = 0.0
            continue
        key = (mode, question, clean_text)
        if key in cache:
            cached_scores[text] = cache[key]
            continue
        uncached.append((text, key, clean_text))
        if mode == 'token':
            item_label = "OCR token"
            guidance = "Can this single OCR token, by itself or as a key clue, help answer the question?"
        else:
            item_label = "OCR text span"
            guidance = "Can this OCR text span provide enough useful evidence to help answer the question?"
        prompt_texts.append(
            f"Question: {question}\n"
            f"{item_label}: {clean_text}\n"
            f"{guidance}\n"
            "A. yes\n"
            "B. no\n"
            "Answer with only A or B."
        )

    if prompt_texts:
        inputs = vlm_processor(text=prompt_texts, padding=True, return_tensors="pt").to(vlm_model.device)
        with torch.no_grad():
            outputs = vlm_model(**inputs)
            logits = outputs.logits[:, -1, :]
            a_probabilities = torch.nn.functional.softmax(logits, dim=-1)[:, a_token_id]
        for (orig_text, key, _), score in zip(uncached, a_probabilities.tolist()):
            cache[key] = float(score)
            cached_scores[orig_text] = float(score)

    return [cached_scores.get(text, 0.0) for text in texts]


def proposal_text_item_bonus(proposal_xywh, scored_items, weight):
    if weight <= 0 or not scored_items:
        return 0.0
    px1, py1, pw, ph = proposal_xywh
    px2, py2 = px1 + pw, py1 + ph
    best = 0.0
    for item in scored_items:
        rel = item.get("relevance", 0.0)
        if rel <= 0:
            continue
        cx1, cy1, cx2, cy2 = item["box_xyxy"]
        ix1, iy1 = max(px1, cx1), max(py1, cy1)
        ix2, iy2 = min(px2, cx2), min(py2, cy2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        cluster_area = max(1, (cx2 - cx1) * (cy2 - cy1))
        coverage = inter / cluster_area
        ccx, ccy = (cx1 + cx2) / 2.0, (cy1 + cy2) / 2.0
        center_inside = 1.0 if (px1 <= ccx <= px2 and py1 <= ccy <= py2) else 0.0
        score = rel * max(coverage, center_inside * 0.8)
        best = max(best, score)
    return weight * best


def anchored_text_windows(cluster_props, h, w, mode='off', fixed_scale=0.4, adaptive_scales=None):
    """Convert cluster union boxes into fixed-size windows while preserving rough relative position."""
    if mode == 'off':
        return list(cluster_props)
    if adaptive_scales is None or len(adaptive_scales) == 0:
        adaptive_scales = [0.4, 0.6, 0.8]

    def anchor_window(prop, scale):
        x1, y1, bw, bh = prop
        x2, y2 = x1 + bw, y1 + bh
        ww = max(1, min(w, int(round(scale * w))))
        wh = max(1, min(h, int(round(scale * h))))
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        rx = cx / max(w, 1)
        ry = cy / max(h, 1)
        sx = int(round(cx - rx * ww))
        sy = int(round(cy - ry * wh))

        # Keep the cluster fully inside the anchored window.
        if sx > x1:
            sx = x1
        if sx + ww < x2:
            sx = x2 - ww
        if sy > y1:
            sy = y1
        if sy + wh < y2:
            sy = y2 - wh

        sx = max(0, min(sx, w - ww))
        sy = max(0, min(sy, h - wh))
        return (sx, sy, ww, wh)

    windows = []
    for prop in cluster_props:
        _, _, bw, bh = prop
        if mode == 'adaptive':
            scale = adaptive_scales[-1]
            for cand in adaptive_scales:
                if bw <= cand * w and bh <= cand * h:
                    scale = cand
                    break
        else:
            scale = fixed_scale
        windows.append(anchor_window(prop, scale))
    return windows


def text_window_density_proposals(text_boxes, h, w, win_sizes, top_k=4, nms_thresh=0.5, stride_ratio=0.25):
    """Legacy sliding-window text-density proposals."""
    if not text_boxes:
        return []

    scale = 4
    dh, dw = h // scale, w // scale
    density = np.zeros((dh, dw), dtype=np.float32)
    for bx1, by1, bx2, by2 in text_boxes:
        y1d = max(0, int(by1 / scale))
        y2d = min(dh, int(by2 / scale))
        x1d = max(0, int(bx1 / scale))
        x2d = min(dw, int(bx2 / scale))
        if y2d > y1d and x2d > x1d:
            density[y1d:y2d, x1d:x2d] += 1.0

    integral = np.zeros((dh + 1, dw + 1), dtype=np.float64)
    integral[1:, 1:] = np.cumsum(np.cumsum(density, axis=0), axis=1)

    def window_sum(y1, x1, y2, x2):
        return integral[y2, x2] - integral[y1, x2] - integral[y2, x1] + integral[y1, x1]

    candidates = []
    for ww, wh in win_sizes:
        dww, dwh = ww // scale, wh // scale
        if dww < 1 or dwh < 1:
            continue
        stride_x = max(1, int(dww * stride_ratio))
        stride_y = max(1, int(dwh * stride_ratio))
        for dy in range(0, dh - dwh + 1, stride_y):
            for dx in range(0, dw - dww + 1, stride_x):
                s = window_sum(dy, dx, dy + dwh, dx + dww)
                if s > 0:
                    candidates.append((s, dx * scale, dy * scale, ww, wh))

    if not candidates:
        return []

    candidates.sort(key=lambda x: x[0], reverse=True)

    def iou(a, b):
        ax1, ay1 = a[1], a[2]
        ax2, ay2 = a[1] + a[3], a[2] + a[4]
        bx1, by1 = b[1], b[2]
        bx2, by2 = b[1] + b[3], b[2] + b[4]
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0

    kept = []
    for c in candidates:
        if all(iou(c, k) < nms_thresh for k in kept):
            kept.append(c)
        if len(kept) >= top_k:
            break

    proposals = []
    for _, sx, sy, ww, wh in kept:
        sx = max(0, min(sx, w - ww))
        sy = max(0, min(sy, h - wh))
        proposals.append((sx, sy, ww, wh))
    return proposals


def select_key_zoom(text_boxes_list, video, question, text_list=None, object_boxes_list=None):
    """
    object_boxes_list: list of list of [x1, y1, x2, y2] for each frame (可为None)
    """
    global vlm_model, vlm_processor, threshold, win_size, use_focus_bonus, use_layout_zoom
    global g_crop_mode, g_density_top_k, g_density_nms
    global g_cluster_expand_ratio, g_cluster_min_size_ratio, g_cluster_multi_scales, g_cluster_add_density_scale, g_cluster_add_density_top_k
    global g_text_anchor_mode, g_text_anchor_fixed_scale, g_text_anchor_scales, g_text_rerank_weight, g_text_rerank_mode
    h, w = video.shape[1:3]
    # 支持纯baseline：直接返回原始帧，无裁剪/缩放
    if globals().get('g_crop_mode', None) == 'off':
        return [np.array(f) for f in video]
    def resize_box(box, scale, h, w):
        x, y, bw, bh = box
        cx = x + bw / 2.0
        cy = y + bh / 2.0
        new_w = bw * scale
        new_h = bh * scale
        x1 = max(0, int(round(cx - new_w / 2.0)))
        y1 = max(0, int(round(cy - new_h / 2.0)))
        x2 = min(w, int(round(cx + new_w / 2.0)))
        y2 = min(h, int(round(cy + new_h / 2.0)))
        if x2 <= x1:
            x2 = min(w, x1 + 1)
        if y2 <= y1:
            y2 = min(h, y1 + 1)
        return (x1, y1, x2 - x1, y2 - y1)

    def apply_expand_and_min_size(box, h, w, expand_ratio=0.0, min_size_ratio=0.0):
        x, y, bw, bh = box
        x1, y1, x2, y2 = x, y, x + bw, y + bh
        if expand_ratio > 0:
            pad_w = bw * expand_ratio
            pad_h = bh * expand_ratio
            x1 -= pad_w
            y1 -= pad_h
            x2 += pad_w
            y2 += pad_h

        cur_w = x2 - x1
        cur_h = y2 - y1
        min_w = w * min_size_ratio
        min_h = h * min_size_ratio
        if cur_w < min_w:
            extra = (min_w - cur_w) / 2.0
            x1 -= extra
            x2 += extra
        if cur_h < min_h:
            extra = (min_h - cur_h) / 2.0
            y1 -= extra
            y2 += extra

        x1 = max(0, int(round(x1)))
        y1 = max(0, int(round(y1)))
        x2 = min(w, int(round(x2)))
        y2 = min(h, int(round(y2)))
        if x2 <= x1:
            x2 = min(w, x1 + 1)
        if y2 <= y1:
            y2 = min(h, y1 + 1)
        return (x1, y1, x2 - x1, y2 - y1)

    def dedup_props(props):
        out = []
        seen = set()
        for p in props:
            key = tuple(map(int, p))
            if key not in seen:
                seen.add(key)
                out.append(key)
        return out

    def text_object_density_proposals(text_boxes, object_boxes, h, w, win_sizes, top_k=4, nms_thresh=0.5, stride_ratio=0.25):
        """
        先用文本框生成 density proposals，再扩展到包住 proposal+相关物体的最小外接矩形。
        相关物体定义：与 proposal IoU>0.1 或中心点落在 proposal 内。
        """
        proposals = text_density_proposals(text_boxes, h, w, win_sizes, top_k=top_k, nms_thresh=nms_thresh, stride_ratio=stride_ratio)
        proposals = anchored_text_windows(
            proposals,
            h,
            w,
            mode=g_text_anchor_mode,
            fixed_scale=g_text_anchor_fixed_scale,
            adaptive_scales=g_text_anchor_scales,
        )
        if not object_boxes or not proposals:
            return proposals
        def iou(boxA, boxB):
            ax1, ay1, ax2, ay2 = boxA
            bx1, by1, bx2, by2 = boxB
            ix1, iy1 = max(ax1, bx1), max(ay1, by1)
            ix2, iy2 = min(ax2, bx2), min(ay2, by2)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            areaA = (ax2 - ax1) * (ay2 - ay1)
            areaB = (bx2 - bx1) * (by2 - by1)
            union = areaA + areaB - inter
            return inter / union if union > 0 else 0
        new_props = []
        for sx, sy, ww, wh in proposals:
            px1, py1, px2, py2 = sx, sy, sx+ww, sy+wh
            related = []
            for ob in object_boxes:
                ox1, oy1, ox2, oy2 = ob
                # IoU > 0.1
                if iou([px1, py1, px2, py2], ob) > 0.1:
                    related.append(ob)
                    continue
                # center in proposal
                cx, cy = (ox1+ox2)//2, (oy1+oy2)//2
                if px1 <= cx <= px2 and py1 <= cy <= py2:
                    related.append(ob)
            if related:
                all_x1 = min([px1]+[b[0] for b in related])
                all_y1 = min([py1]+[b[1] for b in related])
                all_x2 = max([px2]+[b[2] for b in related])
                all_y2 = max([py2]+[b[3] for b in related])
                prop = (all_x1, all_y1, all_x2-all_x1, all_y2-all_y1)
            else:
                prop = (px1, py1, ww, wh)

            prop = apply_expand_and_min_size(
                prop,
                h,
                w,
                expand_ratio=g_cluster_expand_ratio,
                min_size_ratio=g_cluster_min_size_ratio,
            )

            if g_cluster_multi_scales:
                for scale in g_cluster_multi_scales:
                    new_props.append(resize_box(prop, scale, h, w))
            else:
                new_props.append(prop)
        return dedup_props(new_props)
    aspect_ratio = w / h
    key_frame_zoom = []
    
    promt = f'Are the objects and text in this frame relevant to answering the question: \'{question}\'?\n A. yes, B. no\nAnswer with the option\'s letter directly.'
    conversation = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": promt},
            ]
        },
    ]
    
    question_lower = question.lower().split()

    for f_id, (text_boxes, frame) in enumerate(zip(text_boxes_list, video)):
        object_boxes = object_boxes_list[f_id] if object_boxes_list is not None and f_id < len(object_boxes_list) else None
        frame_texts = text_list[f_id] if text_list and f_id < len(text_list) else []
        rerank_items = get_text_rerank_items(text_boxes, frame_texts, g_text_rerank_mode)
        if g_text_rerank_weight > 0 and rerank_items:
            rerank_scores = score_text_relevance_with_qwen(
                question,
                [item["text"] for item in rerank_items],
                g_text_rerank_mode,
            )
            for item, score in zip(rerank_items, rerank_scores):
                item["relevance"] = score
        # check if any OCR text in this frame overlaps with question words
        text_bonus = 0.0
        if use_focus_bonus and frame_texts:
            frame_text_lower = ' '.join(t.lower() for t in frame_texts)
            if any(qw in frame_text_lower for qw in question_lower if len(qw) > 2):
                text_bonus = 0.15
        win_w, win_h = int(win_size * w), int(win_size * h)

        if g_crop_mode == 'density' and text_boxes:
            # 文本+物体联合 proposal
            win_sizes = [
                (int(0.4 * w), int(0.4 * h)),
                (int(0.6 * w), int(0.6 * h)),
                (int(0.8 * w), int(0.8 * h)),
            ]
            proposals = text_object_density_proposals(
                text_boxes, object_boxes, h, w, win_sizes,
                top_k=g_density_top_k, nms_thresh=g_density_nms)
            if proposals:
                sub_imgs = []
                inf_imgs = []
                conversations = []
                for sx, sy, pw, ph in proposals:
                    cropped_img = Image.fromarray(frame[sy:sy+ph, sx:sx+pw])
                    sub_imgs.append(cropped_img)
                    cropped_img = cropped_img.resize((win_w, win_h))
                    inf_imgs.append(cropped_img)
                    conversations.append(conversation)

                if len(sub_imgs) > 0:
                    text_prompt = vlm_processor.apply_chat_template(conversations, add_generation_prompt=True)
                    inputs = vlm_processor(text=text_prompt, images=inf_imgs, padding=True, return_tensors="pt").to(vlm_model.device)
                    with torch.no_grad():
                        outputs = vlm_model(**inputs)
                        logits = outputs.logits[:, -1, :]
                        a_probabilities = torch.nn.functional.softmax(logits, dim=-1)[:, a_token_id]
                        a_probabilities = a_probabilities + text_bonus
                        if g_text_rerank_weight > 0 and rerank_items:
                            rerank_bonus = torch.tensor(
                                [
                                    proposal_text_item_bonus(p, rerank_items, g_text_rerank_weight)
                                    for p in proposals
                                ],
                                device=a_probabilities.device,
                                dtype=a_probabilities.dtype,
                            )
                            a_probabilities = a_probabilities + rerank_bonus
                        max_val, max_id = torch.max(a_probabilities, dim=0)
                        if max_val > threshold:
                            key_frame_zoom.append(np.array(sub_imgs[max_id].resize((w, h))))
                continue  # skip fixed-corner logic for this frame

        # fixed corner crops
        crop_candidates = [
            (0, 0, win_w, win_h),
            (w - win_w, 0, win_w, win_h),
            (0, h - win_h, win_w, win_h),
            (w - win_w, h - win_h, win_w, win_h),
        ]

        # hybrid ablation: 只用单一 window size
        if g_crop_mode == 'hybrid_04' and text_boxes:
            d_win_sizes = [
                (int(0.4 * w), int(0.4 * h)),
            ]
            d_proposals = text_window_density_proposals(
                text_boxes, h, w, d_win_sizes, top_k=g_density_top_k, nms_thresh=g_density_nms
            )
            for sx, sy, pw, ph in d_proposals:
                candidate = (sx, sy, pw, ph)
                if candidate not in crop_candidates:
                    crop_candidates.append(candidate)

        if g_crop_mode == 'hybrid_06' and text_boxes:
            d_win_sizes = [
                (int(0.6 * w), int(0.6 * h)),
            ]
            d_proposals = text_window_density_proposals(
                text_boxes, h, w, d_win_sizes, top_k=g_density_top_k, nms_thresh=g_density_nms
            )
            for sx, sy, pw, ph in d_proposals:
                candidate = (sx, sy, pw, ph)
                if candidate not in crop_candidates:
                    crop_candidates.append(candidate)

        if g_crop_mode == 'hybrid_08' and text_boxes:
            d_win_sizes = [
                (int(0.8 * w), int(0.8 * h)),
            ]
            d_proposals = text_window_density_proposals(
                text_boxes, h, w, d_win_sizes, top_k=g_density_top_k, nms_thresh=g_density_nms
            )
            for sx, sy, pw, ph in d_proposals:
                candidate = (sx, sy, pw, ph)
                if candidate not in crop_candidates:
                    crop_candidates.append(candidate)

        # layout-guided: add text-centered crops based on bbox positions
        if use_layout_zoom != 'off' and text_boxes:
            # text centroid crop (centroid / full mode)
            if use_layout_zoom in ('centroid', 'full'):
                cx = int(np.mean([(b[0] + b[2]) / 2 for b in text_boxes]))
                cy = int(np.mean([(b[1] + b[3]) / 2 for b in text_boxes]))
                sx = max(0, min(cx - win_w // 2, w - win_w))
                sy = max(0, min(cy - win_h // 2, h - win_h))
                candidate = (sx, sy, win_w, win_h)
                if candidate not in crop_candidates:
                    crop_candidates.append(candidate)
            # frame center crop (center / full mode)
            if use_layout_zoom in ('center', 'full'):
                csx = max(0, (w - win_w) // 2)
                csy = max(0, (h - win_h) // 2)
                candidate = (csx, csy, win_w, win_h)
                if candidate not in crop_candidates:
                    crop_candidates.append(candidate)

        sub_imgs = []
        inf_imgs = []
        conversations = []
        proposal_meta = []
        for s_id, (s_x, s_y, base_w, base_h) in enumerate(crop_candidates):
            e_x, e_y = s_x + base_w, s_y + base_h
            relevant_boxes = []

            for box in text_boxes:
                bx1, by1, bx2, by2 = box
                if not (bx2 <= s_x or bx1 >= e_x or by2 <= s_y or by1 >= e_y):
                    relevant_boxes.append(box)
                    
            if not relevant_boxes: 
                continue
            
            extended_x1, extended_y1 = s_x, s_y
            extended_x2, extended_y2 = e_x, e_y

            for bx1, by1, bx2, by2 in relevant_boxes:
                if bx1 < s_x:
                    extended_x1 = min(extended_x1, bx1)
                if bx2 > e_x:
                    extended_x2 = max(extended_x2, min(bx2, w))
                if by1 < s_y:
                    extended_y1 = min(extended_y1, by1)
                if by2 > e_y:
                    extended_y2 = max(extended_y2, min(by2, h))
            
            cur_w = extended_x2 - extended_x1
            cur_h = extended_y2 - extended_y1
            cur_ratio = cur_w / cur_h

            if abs(cur_ratio - aspect_ratio) > 1e-2:
                if cur_ratio > aspect_ratio:
                    target_h = cur_w / aspect_ratio
                    if extended_y1 + target_h < h:
                        extended_y2 = min(h, int(extended_y1 + target_h))
                    else:
                        extended_y1 = max(0, int(extended_y2 - target_h))
                else:
                    target_w = cur_h * aspect_ratio
                    if extended_x1 + target_w < w:
                        extended_x2 = min(w, int(extended_x1 + target_w))
                    else:
                        extended_x1 = max(0, int(extended_x2 - target_w))

            cropped_img = Image.fromarray(frame[extended_y1:extended_y2, extended_x1:extended_x2])
            
            
            sub_imgs.append(cropped_img)
            cropped_img = cropped_img.resize((win_w, win_h))
            inf_imgs.append(cropped_img)
            conversations.append(conversation)
            proposal_meta.append((extended_x1, extended_y1, extended_x2 - extended_x1, extended_y2 - extended_y1))
        
        
        if len(sub_imgs) > 0:
            text_prompt = vlm_processor.apply_chat_template(conversations, add_generation_prompt=True)
            inputs = vlm_processor(text=text_prompt, images=inf_imgs, padding=True, return_tensors="pt").to(vlm_model.device)
            with torch.no_grad():
                outputs = vlm_model(**inputs)
                logits = outputs.logits[:, -1, :]

                ### threshold
                a_probabilities = torch.nn.functional.softmax(logits, dim=-1)[:, a_token_id]
                a_probabilities = a_probabilities + text_bonus
                if g_text_rerank_weight > 0 and proposal_meta and rerank_items:
                    rerank_bonus = torch.tensor(
                        [
                            proposal_text_item_bonus(p, rerank_items, g_text_rerank_weight)
                            for p in proposal_meta
                        ],
                        device=a_probabilities.device,
                        dtype=a_probabilities.dtype,
                    )
                    a_probabilities = a_probabilities + rerank_bonus
                max_val, max_id = torch.max(a_probabilities, dim=0)
                if max_val > threshold:
                    key_frame_zoom.append(np.array(sub_imgs[max_id].resize((w, h))))
                
    return key_frame_zoom


all_oframes = 0
all_pframes = 0
all_avg = []

def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


def smart_resize(
    height: int, width: int, factor: int = IMAGE_FACTOR, min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS
) -> tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def to_rgb(pil_image: Image.Image) -> Image.Image:
    if pil_image.mode == 'RGBA':
        white_background = Image.new("RGB", pil_image.size, (255, 255, 255))
        white_background.paste(pil_image, mask=pil_image.split()[3])  # Use alpha channel as mask
        return white_background
    else:
        return pil_image.convert("RGB")


def fetch_image(ele: dict[str, str | Image.Image], size_factor: int = IMAGE_FACTOR) -> Image.Image:
    if "image" in ele:
        image = ele["image"]
    else:
        image = ele["image_url"]
    image_obj = None
    if isinstance(image, Image.Image):
        image_obj = image
    elif image.startswith("http://") or image.startswith("https://"):
        # fix memory leak issue while using BytesIO
        with requests.get(image, stream=True) as response:
            response.raise_for_status()
            with BytesIO(response.content) as bio:
                image_obj = copy.deepcopy(Image.open(bio))
    elif image.startswith("file://"):
        image_obj = Image.open(image[7:])
    elif image.startswith("data:image"):
        if "base64," in image:
            _, base64_data = image.split("base64,", 1)
            data = base64.b64decode(base64_data)
            # fix memory leak issue while using BytesIO
            with BytesIO(data) as bio:
                image_obj = copy.deepcopy(Image.open(bio))
    else:
        image_obj = Image.open(image)
    if image_obj is None:
        raise ValueError(f"Unrecognized image input, support local path, http url, base64 and PIL.Image, got {image}")
    image = to_rgb(image_obj)
    ## resize
    if "resized_height" in ele and "resized_width" in ele:
        resized_height, resized_width = smart_resize(
            ele["resized_height"],
            ele["resized_width"],
            factor=size_factor,
        )
    else:
        width, height = image.size
        min_pixels = ele.get("min_pixels", MIN_PIXELS)
        max_pixels = ele.get("max_pixels", MAX_PIXELS)
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=size_factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
    image = image.resize((resized_width, resized_height))

    return image


def smart_nframes(
    ele: dict,
    total_frames: int,
    video_fps: int | float,
) -> int:
    """calculate the number of frames for video used for model inputs.

    Args:
        ele (dict): a dict contains the configuration of video.
            support either `fps` or `nframes`:
                - nframes: the number of frames to extract for model inputs.
                - fps: the fps to extract frames for model inputs.
                    - min_frames: the minimum number of frames of the video, only used when fps is provided.
                    - max_frames: the maximum number of frames of the video, only used when fps is provided.
        total_frames (int): the original total number of frames of the video.
        video_fps (int | float): the original fps of the video.

    Raises:
        ValueError: nframes should in interval [FRAME_FACTOR, total_frames].

    Returns:
        int: the number of frames for video used for model inputs.
    """
    assert not ("fps" in ele and "nframes" in ele), "Only accept either `fps` or `nframes`"
    if "nframes" in ele:
        nframes = round_by_factor(ele["nframes"], FRAME_FACTOR)
    else:
        fps = ele.get("fps", FPS)
        min_frames = ceil_by_factor(ele.get("min_frames", FPS_MIN_FRAMES), FRAME_FACTOR)
        max_frames = floor_by_factor(ele.get("max_frames", min(FPS_MAX_FRAMES, total_frames)), FRAME_FACTOR)
        nframes = total_frames / video_fps * fps
        if nframes > total_frames:
            logger.warning(f"smart_nframes: nframes[{nframes}] > total_frames[{total_frames}]")
        nframes = min(min(max(nframes, min_frames), max_frames), total_frames)
        nframes = floor_by_factor(nframes, FRAME_FACTOR)
    if not (FRAME_FACTOR <= nframes and nframes <= total_frames):
        raise ValueError(f"nframes should in interval [{FRAME_FACTOR}, {total_frames}], but got {nframes}.")
    return nframes



def is_decord_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("decord") is not None


def calculate_video_frame_range(
    ele: dict,
    total_frames: int,
    video_fps: float,
) -> tuple[int, int, int]:
    """
    Calculate the start and end frame indices based on the given time range.

    Args:
        ele (dict): A dictionary containing optional 'video_start' and 'video_end' keys (in seconds).
        total_frames (int): Total number of frames in the video.
        video_fps (float): Frames per second of the video.

    Returns:
        tuple: A tuple containing (start_frame, end_frame, frame_count).

    Raises:
        ValueError: If input parameters are invalid or the time range is inconsistent.
    """
    # Validate essential parameters
    if video_fps <= 0:
        raise ValueError("video_fps must be a positive number")
    if total_frames <= 0:
        raise ValueError("total_frames must be a positive integer")

    # Get start and end time in seconds
    video_start = ele.get("video_start", None)
    video_end = ele.get("video_end", None)
    if video_start is None and video_end is None:
        return 0, total_frames - 1, total_frames

    max_duration = total_frames / video_fps
    # Process start frame
    if video_start is not None:
        video_start_clamped = max(0.0, min(video_start, max_duration))
        start_frame = math.ceil(video_start_clamped * video_fps)
    else:
        start_frame = 0
    # Process end frame
    if video_end is not None:
        video_end_clamped = max(0.0, min(video_end, max_duration))
        end_frame = math.floor(video_end_clamped * video_fps)
        end_frame = min(end_frame, total_frames - 1)
    else:
        end_frame = total_frames - 1

    # Validate frame order
    if start_frame >= end_frame:
        raise ValueError(
            f"Invalid time range: Start frame {start_frame} (at {video_start_clamped if video_start is not None else 0}s) "
            f"exceeds end frame {end_frame} (at {video_end_clamped if video_end is not None else max_duration}s). "
            f"Video duration: {max_duration:.2f}s ({total_frames} frames @ {video_fps}fps)"
        )

    logger.info(f"calculate video frame range: {start_frame=}, {end_frame=}, {total_frames=} from {video_start=}, {video_end=}, {video_fps=:.3f}")
    return start_frame, end_frame, end_frame - start_frame + 1


def _read_video_decord(
    ele: dict,
) -> (torch.Tensor, float):
    """read video using decord.VideoReader

    Args:
        ele (dict): a dict contains the configuration of video.
        support keys:
            - video: the path of video. support "file://", "http://", "https://" and local path.
            - video_start: the start time of video.
            - video_end: the end time of video.
    Returns:
        torch.Tensor: the video tensor with shape (T, C, H, W).
    """
    import decord
    video_path = ele["video"]
    st = time.time()
    vr = decord.VideoReader(video_path)
    total_frames, video_fps = len(vr), vr.get_avg_fps()
    start_frame, end_frame, total_frames = calculate_video_frame_range(
        ele,
        total_frames,
        video_fps,
    )
    nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
    idx = torch.linspace(start_frame, end_frame, nframes).round().long().tolist()
    video = vr.get_batch(idx).asnumpy()
    # video = torch.tensor(video).permute(0, 3, 1, 2)  # Convert to TCHW format
    logger.info(f"decord:  {video_path=}, {total_frames=}, {video_fps=}, time={time.time() - st:.3f}s")
    sample_fps = nframes / max(total_frames, 1e-6) * video_fps
    return video, sample_fps


def is_torchcodec_available() -> bool:
    """Check if torchcodec is available and properly installed."""
    try:
        import importlib.util
        if importlib.util.find_spec("torchcodec") is None:
            return False
        from torchcodec.decoders import VideoDecoder
        return True
    except (ImportError, AttributeError, Exception):
        return False




VIDEO_READER_BACKENDS = {
    "decord": _read_video_decord,
}

FORCE_QWENVL_VIDEO_READER = os.getenv("FORCE_QWENVL_VIDEO_READER", None)


@lru_cache(maxsize=1)
def get_video_reader_backend() -> str:
    if FORCE_QWENVL_VIDEO_READER is not None:
        video_reader_backend = FORCE_QWENVL_VIDEO_READER
    else:
        assert is_decord_available()
        video_reader_backend = "decord"

    print(f"qwen-vl-utils using {video_reader_backend} to read video.", file=sys.stderr)
    return video_reader_backend


def fetch_video(
    ele: dict,
    question: str,
    image_factor: int = IMAGE_FACTOR,
    return_video_sample_fps: bool = False,
    d2_predictor=None,
    d2_class_ids=None,
) -> torch.Tensor | list[Image.Image]:
    global video_text_spotter, tracker_visualizer, all_oframes, all_pframes, all_avg
    if isinstance(ele["video"], str):
        video_reader_backend = get_video_reader_backend()
        try:
            video, sample_fps = VIDEO_READER_BACKENDS[video_reader_backend](ele)
        except Exception as e:
            logger.warning(f"video_reader_backend {video_reader_backend} error, use torchvision as default, msg: {e}")
            video, sample_fps = VIDEO_READER_BACKENDS["torchvision"](ele)
        
        # optional keyframe pre-sampling
        if kf_sample_mode != 'off':
            kf_indices = keyframe_sample(video)
            video_sampled = video[kf_indices]
            print(f'[KF SAMPLE] {video.shape[0]} frames -> {len(kf_indices)} frames (mode={kf_sample_mode}, N={kf_n_seg}, k={kf_k})')
        else:
            video_sampled = video

        text_boxes_list, text_list = ocr_det_with_text(video_sampled)
        obj_boxes = None
        if d2_predictor is not None:
            sampled_frames = []
            for frame in video_sampled:
                if hasattr(frame, "cpu"):
                    sampled_frames.append(frame.cpu().numpy())
                else:
                    sampled_frames.append(np.asarray(frame))
            obj_boxes = detectron2_object_det(sampled_frames, d2_predictor, class_ids=d2_class_ids)
        key_frame_zoom = select_key_zoom(text_boxes_list, video_sampled, question, text_list=text_list, object_boxes_list=obj_boxes)
        
        oframes = video.shape[0]
        all_oframes += oframes
        
        if len(key_frame_zoom) > 0:
            pframes = len(key_frame_zoom) 
            all_pframes += pframes
            video = np.array(key_frame_zoom)
        else:
            all_pframes += oframes 
            pframes = oframes 
            print('use org fps frames')
            
        avg = pframes / oframes
        all_avg.append(avg)
        print('cur all: ', oframes, 'cur key: ', pframes, 'cur avg: ', avg, ' all_oframes: ', all_oframes, ' all_pframes: ', all_pframes, ' all_avg: ', all_pframes/all_oframes, ' all_mean: ', np.mean(all_avg))
        
        video = torch.tensor(video).permute(0, 3, 1, 2)
        nframes, _, height, width = video.shape
        min_pixels = ele.get("min_pixels", VIDEO_MIN_PIXELS)
        total_pixels = ele.get("total_pixels", VIDEO_TOTAL_PIXELS)
        max_pixels = max(min(VIDEO_MAX_PIXELS, total_pixels / nframes * FRAME_FACTOR), int(min_pixels * 1.05))
        max_pixels_supposed = ele.get("max_pixels", max_pixels)
        if max_pixels_supposed > max_pixels:
            logger.warning(f"The given max_pixels[{max_pixels_supposed}] exceeds limit[{max_pixels}].")
        max_pixels = min(max_pixels_supposed, max_pixels)
        if "resized_height" in ele and "resized_width" in ele:
            resized_height, resized_width = smart_resize(
                ele["resized_height"],
                ele["resized_width"],
                factor=image_factor,
            )
        else:
            resized_height, resized_width = smart_resize(
                height,
                width,
                factor=image_factor,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
        video = transforms.functional.resize(
            video,
            [resized_height, resized_width],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        ).float()
        if return_video_sample_fps:
            return video, sample_fps, text_list
        return video
    else:
        assert isinstance(ele["video"], (list, tuple))
        process_info = ele.copy()
        process_info.pop("type", None)
        process_info.pop("video", None)
        images = [
            fetch_image({"image": video_element, **process_info}, size_factor=image_factor)
            for video_element in ele["video"]
        ]
        nframes = ceil_by_factor(len(images), FRAME_FACTOR)
        if len(images) < nframes:
            images.extend([images[-1]] * (nframes - len(images)))
        if return_video_sample_fps:
            return images, process_info.pop("fps", 2.0), []
        return images


def fetch_video_nocrop(
    ele: dict,
    image_factor: int = IMAGE_FACTOR,
    return_video_sample_fps: bool = False,
) -> torch.Tensor | list[Image.Image]:
    """Read video frames by fps/nframes and resize for Qwen, without OCR/crop/zoom."""
    if isinstance(ele["video"], str):
        video_reader_backend = get_video_reader_backend()
        try:
            video, sample_fps = VIDEO_READER_BACKENDS[video_reader_backend](ele)
        except Exception as e:
            logger.warning(f"video_reader_backend {video_reader_backend} error, use torchvision as default, msg: {e}")
            video, sample_fps = VIDEO_READER_BACKENDS["torchvision"](ele)

        video = torch.tensor(video).permute(0, 3, 1, 2)
        nframes, _, height, width = video.shape
        min_pixels = ele.get("min_pixels", VIDEO_MIN_PIXELS)
        total_pixels = ele.get("total_pixels", VIDEO_TOTAL_PIXELS)
        max_pixels = max(min(VIDEO_MAX_PIXELS, total_pixels / nframes * FRAME_FACTOR), int(min_pixels * 1.05))
        max_pixels_supposed = ele.get("max_pixels", max_pixels)
        if max_pixels_supposed > max_pixels:
            logger.warning(f"The given max_pixels[{max_pixels_supposed}] exceeds limit[{max_pixels}].")
        max_pixels = min(max_pixels_supposed, max_pixels)
        if "resized_height" in ele and "resized_width" in ele:
            resized_height, resized_width = smart_resize(
                ele["resized_height"],
                ele["resized_width"],
                factor=image_factor,
            )
        else:
            resized_height, resized_width = smart_resize(
                height,
                width,
                factor=image_factor,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
        video = transforms.functional.resize(
            video,
            [resized_height, resized_width],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        ).float()
        if return_video_sample_fps:
            return video, sample_fps
        return video
    else:
        assert isinstance(ele["video"], (list, tuple))
        process_info = ele.copy()
        process_info.pop("type", None)
        process_info.pop("video", None)
        images = [
            fetch_image({"image": video_element, **process_info}, size_factor=image_factor)
            for video_element in ele["video"]
        ]
        nframes = ceil_by_factor(len(images), FRAME_FACTOR)
        if len(images) < nframes:
            images.extend([images[-1]] * (nframes - len(images)))
        if return_video_sample_fps:
            return images, process_info.pop("fps", 2.0)
        return images


def extract_vision_info(conversations: list[dict] | list[list[dict]]) -> list[dict]:
    vision_infos = []
    if isinstance(conversations[0], dict):
        conversations = [conversations]
    for conversation in conversations:
        for message in conversation:
            if isinstance(message["content"], list):
                for ele in message["content"]:
                    if (
                        "image" in ele
                        or "image_url" in ele
                        or "video" in ele
                        or ele.get("type","") in ("image", "image_url", "video")
                    ):
                        vision_infos.append(ele)
    return vision_infos


def process_vision_info(
    question,
    conversations: list[dict] | list[list[dict]],
    return_video_kwargs: bool = False,
    d2_predictor=None,
    d2_class_ids=None,
) -> tuple[list[Image.Image] | None, list[torch.Tensor | list[Image.Image]] | None, Optional[dict]]:

    vision_infos = extract_vision_info(conversations)
    ## Read images or videos
    image_inputs = []
    video_inputs = []
    video_sample_fps_list = []
    all_text_lists = []
    for idx, vision_info in enumerate(vision_infos):
        if "image" in vision_info or "image_url" in vision_info:
            image_inputs.append(fetch_image(vision_info))
        elif "video" in vision_info:
            video_input, video_sample_fps, text_list = fetch_video(
                vision_info,
                question,
                return_video_sample_fps=True,
                d2_predictor=d2_predictor,
                d2_class_ids=d2_class_ids,
            )
            video_sample_fps_list.append(video_sample_fps)
            video_inputs.append(video_input)
            all_text_lists.append(text_list)
        else:
            raise ValueError("image, image_url or video should in content.")
    if len(image_inputs) == 0:
        image_inputs = None
    if len(video_inputs) == 0:
        video_inputs = None
    if return_video_kwargs:
        return image_inputs, video_inputs, {'fps': video_sample_fps_list}, all_text_lists
    return image_inputs, video_inputs


def process_vision_info_nocrop(
    conversations: list[dict] | list[list[dict]],
    return_video_kwargs: bool = False,
) -> tuple[list[Image.Image] | None, list[torch.Tensor | list[Image.Image]] | None, Optional[dict]]:
    """Read images/videos for Qwen without OCR, crop, zoom, or object proposals."""
    vision_infos = extract_vision_info(conversations)
    image_inputs = []
    video_inputs = []
    video_sample_fps_list = []
    for vision_info in vision_infos:
        if "image" in vision_info or "image_url" in vision_info:
            image_inputs.append(fetch_image(vision_info))
        elif "video" in vision_info:
            video_input, video_sample_fps = fetch_video_nocrop(
                vision_info,
                return_video_sample_fps=True,
            )
            video_sample_fps_list.append(video_sample_fps)
            video_inputs.append(video_input)
        else:
            raise ValueError("image, image_url or video should in content.")
    if len(image_inputs) == 0:
        image_inputs = None
    if len(video_inputs) == 0:
        video_inputs = None
    if return_video_kwargs:
        return image_inputs, video_inputs, {'fps': video_sample_fps_list}
    return image_inputs, video_inputs
