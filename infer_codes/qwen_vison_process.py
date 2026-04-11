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

def set_key_conf(w_size=0.6, thrd=0.7, focus_bonus=True, layout_zoom='off',
                 kf_sample='off', kf_n_segments=8, kf_neighbors=1,
                 crop_mode='fixed', density_top_k=4, density_nms=0.5):
    global win_size, threshold, use_focus_bonus, use_layout_zoom
    global kf_sample_mode, kf_n_seg, kf_k
    global g_crop_mode, g_density_top_k, g_density_nms
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
    """Generate crop proposals ranked by text density using bbox-based heatmap.

    Args:
        text_boxes: list of [x1, y1, x2, y2] bboxes from GoMatching
        h, w: frame dimensions
        win_sizes: list of (win_w, win_h) tuples to try
        top_k: max proposals after NMS
        nms_thresh: IoU threshold for NMS
        stride_ratio: stride as fraction of window size
    Returns:
        list of (sx, sy, win_w, win_h) proposals sorted by score descending
    """
    if not text_boxes:
        return []

    # build density map at 1/4 resolution for speed
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

    # integral image for fast window sum
    integral = np.zeros((dh + 1, dw + 1), dtype=np.float64)
    integral[1:, 1:] = np.cumsum(np.cumsum(density, axis=0), axis=1)

    def window_sum(y1, x1, y2, x2):
        return integral[y2, x2] - integral[y1, x2] - integral[y2, x1] + integral[y1, x1]

    # slide windows at multiple scales
    candidates = []  # (score, sx, sy, win_w, win_h)
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

    # sort by score descending
    candidates.sort(key=lambda x: x[0], reverse=True)

    # NMS
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

    # convert to (sx, sy, win_w, win_h), clamp to frame
    proposals = []
    for _, sx, sy, ww, wh in kept:
        sx = max(0, min(sx, w - ww))
        sy = max(0, min(sy, h - wh))
        proposals.append((sx, sy, ww, wh))

    return proposals


def select_key_zoom(text_boxes_list, video, question, text_list=None):
    global vlm_model, vlm_processor, threshold, win_size, use_focus_bonus, use_layout_zoom
    global g_crop_mode, g_density_top_k, g_density_nms
    h, w = video.shape[1:3]
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
        frame_texts = text_list[f_id] if text_list and f_id < len(text_list) else []
        # check if any OCR text in this frame overlaps with question words
        text_bonus = 0.0
        if use_focus_bonus and frame_texts:
            frame_text_lower = ' '.join(t.lower() for t in frame_texts)
            if any(qw in frame_text_lower for qw in question_lower if len(qw) > 2):
                text_bonus = 0.15
        win_w, win_h = int(win_size * w), int(win_size * h)

        if g_crop_mode == 'density' and text_boxes:
            # text-density-aware region proposals (pure density, no corners)
            win_sizes = [
                (int(0.4 * w), int(0.4 * h)),
                (int(0.6 * w), int(0.6 * h)),
                (int(0.8 * w), int(0.8 * h)),
            ]
            proposals = text_density_proposals(
                text_boxes, h, w, win_sizes,
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
                        max_val, max_id = torch.max(a_probabilities, dim=0)
                        if max_val > threshold:
                            key_frame_zoom.append(np.array(sub_imgs[max_id].resize((w, h))))
                continue  # skip fixed-corner logic for this frame

        # fixed corner crops
        start_coords = [
        (0, 0),
        (w - win_w, 0),
        (0, h - win_h),
        (w - win_w, h - win_h)]

        # hybrid mode: append density proposals as extra crops alongside corners
        if g_crop_mode == 'hybrid' and text_boxes:
            d_win_sizes = [
                (int(0.4 * w), int(0.4 * h)),
                (int(0.6 * w), int(0.6 * h)),
                (int(0.8 * w), int(0.8 * h)),
            ]
            d_proposals = text_density_proposals(
                text_boxes, h, w, d_win_sizes,
                top_k=g_density_top_k, nms_thresh=g_density_nms)
            for sx, sy, pw, ph in d_proposals:
                # clamp to win_size crop for consistent scoring
                cx, cy = sx + pw // 2, sy + ph // 2
                sx2 = max(0, min(cx - win_w // 2, w - win_w))
                sy2 = max(0, min(cy - win_h // 2, h - win_h))
                if (sx2, sy2) not in start_coords:
                    start_coords.append((sx2, sy2))

        # layout-guided: add text-centered crops based on bbox positions
        if use_layout_zoom != 'off' and text_boxes:
            # text centroid crop (centroid / full mode)
            if use_layout_zoom in ('centroid', 'full'):
                cx = int(np.mean([(b[0] + b[2]) / 2 for b in text_boxes]))
                cy = int(np.mean([(b[1] + b[3]) / 2 for b in text_boxes]))
                sx = max(0, min(cx - win_w // 2, w - win_w))
                sy = max(0, min(cy - win_h // 2, h - win_h))
                if (sx, sy) not in start_coords:
                    start_coords.append((sx, sy))
            # frame center crop (center / full mode)
            if use_layout_zoom in ('center', 'full'):
                csx = max(0, (w - win_w) // 2)
                csy = max(0, (h - win_h) // 2)
                if (csx, csy) not in start_coords:
                    start_coords.append((csx, csy))

        sub_imgs = []
        inf_imgs = []
        conversations = []
        for s_id, (s_x, s_y) in enumerate(start_coords):
            e_x, e_y = s_x + win_w, s_y + win_h
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
        
        
        if len(sub_imgs) > 0:
            text_prompt = vlm_processor.apply_chat_template(conversations, add_generation_prompt=True)
            inputs = vlm_processor(text=text_prompt, images=inf_imgs, padding=True, return_tensors="pt").to(vlm_model.device)
            with torch.no_grad():
                outputs = vlm_model(**inputs)
                logits = outputs.logits[:, -1, :]

                ### threshold
                a_probabilities = torch.nn.functional.softmax(logits, dim=-1)[:, a_token_id]
                a_probabilities = a_probabilities + text_bonus
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


def fetch_video(ele: dict, question: str, image_factor: int = IMAGE_FACTOR, return_video_sample_fps: bool = False) -> torch.Tensor | list[Image.Image]:
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
        key_frame_zoom = select_key_zoom(text_boxes_list, video_sampled, question, text_list=text_list)
        
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
) -> tuple[list[Image.Image] | None, list[torch.Tensor | list[Image.Image]] | None, Optional[dict]]:

    vision_infos = extract_vision_info(conversations)
    ## Read images or videos
    image_inputs = []
    video_inputs = []
    video_sample_fps_list = []
    all_text_lists = []
    for vision_info in vision_infos:
        if "image" in vision_info or "image_url" in vision_info:
            image_inputs.append(fetch_image(vision_info))
        elif "video" in vision_info:
            video_input, video_sample_fps, text_list = fetch_video(vision_info, question, return_video_sample_fps=True)
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

