# Evidence Mining Pipeline for OCR-Centric VideoQA

## Summary

A hierarchical framework that improves VideoQA accuracy through targeted evidence mining. The pipeline combines question parsing, frame retrieval, region localization, OCR visibility scoring, and multi-modal reasoning.

## Quick Start

```python
from pipeline.evidence_pipeline import run_pipeline
import cv2

# Load video frames
frames = [...]  # 16 frames from video

# Run pipeline
question = "What does the sign in blue on top of the road say?"
result = run_pipeline(question, frames)
print(result['answer'])
```

## Directory Structure

```
.
├── parsing/
│   ├── __init__.py
│   └── question_parser.py          # Stage 1: Question parsing
├── retrieval/
│   ├── __init__.py
│   ├── frame_retrieval.py          # Stage 2: Frame retrieval
│   ├── region_localization.py      # Stage 3: Region localization
│   └── ocr_visibility.py           # Stage 4: OCR visibility scoring
├── reasoning/
│   ├── __init__.py
│   └── qwen_reasoning.py           # Stage 6: Final reasoning
├── pipeline/
│   ├── __init__.py
│   └── evidence_pipeline.py        # Main orchestrator
├── utils/
│   ├── __init__.py
│   └── prompt_builder.py           # Prompt templates
├── pipeline_integration_example.py  # Integration with existing code
├── QUICK_REFERENCE.py              # Copy-paste examples
└── EVIDENCE_PIPELINE_DOCS.md       # Full documentation
```

## Pipeline Stages

| Stage | Module | Purpose | Input | Output |
|-------|--------|---------|-------|--------|
| 1 | `question_parser.py` | Parse question structure | Question text | Structured dict |
| 2 | `frame_retrieval.py` | Retrieve relevant frames | Video frames + prompt | Top-K frames |
| 3 | `region_localization.py` | Localize target regions | Frames + localization prompt | Candidate regions |
| 4 | `ocr_visibility.py` | Score OCR readability | Regions | Best crop |
| 5-6 | `qwen_reasoning.py` | Generate answer | Question + evidence | Answer |

## Key Features

✅ **Coarse-to-Fine Retrieval**: Frame retrieval → Region localization  
✅ **OCR-Aware Scoring**: Combines OCR confidence, sharpness, VLM visibility  
✅ **Dual-Image Reasoning**: Global context + Local OCR evidence  
✅ **Structured Parsing**: Reusable question components  
✅ **Flexible Integration**: Works with existing Qwen pipelines  

## Installation

```bash
pip install torch transformers pillow opencv-python numpy paddleocr
pip install peft  # Optional: for LoRA support
```

## Main Classes

### `EvidenceMiningPipeline`
Orchestrates the full pipeline.

```python
from pipeline.evidence_pipeline import EvidenceMiningPipeline

pipeline = EvidenceMiningPipeline(model=model, processor=processor)
result = pipeline.run(question, frames, top_k_frames=5, verbose=True)
```

### `QuestionParser`
Parses questions into semantic components.

```python
from parsing.question_parser import parse_question

parsed = parse_question("What does the sign say?")
# {"target": "sign", "attribute": "", "relation": "", "task": "ocr"}
```

### `FrameRetriever`
Retrieves relevant frames.

```python
from retrieval.frame_retrieval import retrieve_relevant_frames

frames = retrieve_relevant_frames(video_frames, retrieval_prompt, top_k=5)
```

### `RegionLocalizer`
Localizes target regions.

```python
from retrieval.region_localization import localize_target_regions

regions = localize_target_regions(frames, region_prompt)
```

### `OCRVisibilityScorer`
Selects best OCR crop.

```python
from retrieval.ocr_visibility import score_crop_visibility

result = score_crop_visibility(regions, ocr_prompt)
best_crop = result['best_crop']
```

### `QwenReasoner`
Generates final answer.

```python
from reasoning.qwen_reasoning import run_vlm_reasoning

answer = run_vlm_reasoning(
    question,
    global_frame=frame,
    local_crop=crop
)
```

## Integration Example

```python
from pipeline_integration_example import IntegratedVideoQAPipeline

# Initialize
pipeline = IntegratedVideoQAPipeline(
    model_path="Qwen/Qwen2.5-VL-7B-Instruct",
    device="cuda:0"
)

# Process dataset
predictions = pipeline.process_video_qas(
    qa_data=qa_list,
    video_dir="./videos",
    output_json="predictions.json",
    use_evidence_mining=True,
    num_sampled_frames=16,
    top_k_frames=5
)
```

## Configuration

### Frame Retrieval
```python
# Adjust number of retrieved frames
retrieve_relevant_frames(frames, prompt, top_k=10)  # More frames
```

### Region Localization
```python
# Localize with different prompts
region_prompt = "Locate the traffic sign in the image"
localize_target_regions(frames, region_prompt)
```

### OCR Visibility Scoring
```python
# Adjust scoring weights
score_crop_visibility(
    regions,
    prompt,
    alpha=0.5,    # OCR confidence weight
    beta=0.3,     # Sharpness weight
    gamma=0.2     # VLM visibility weight
)
```

## Performance

Expected improvements over baseline SFA:

- **Small objects**: Better localization through targeted retrieval
- **OCR questions**: +15-25% accuracy with dual-image reasoning
- **Distant targets**: Focused evidence selection
- **Traffic signs**: Specialized retrieval prompts
- **Spatial relations**: Global-local evidence fusion

## Example Outputs

### Stage 1: Question Parsing
```
Input: "What does the sign in blue on top of the road say?"
Output: {
    "target": "sign",
    "attribute": "blue",
    "relation": "above road",
    "task": "ocr"
}
```

### Stage 2: Frame Retrieval
```
Retrieved 5 frames:
  Frame 0: score=0.95
  Frame 3: score=0.88
  Frame 7: score=0.82
  ...
```

### Stage 4: OCR Visibility
```
Best crop scores:
  ocr_confidence: 0.85
  sharpness: 0.72
  vlm_visibility: 0.88
  combined_score: 0.82
```

### Final Answer
```
Answer: "STOP"
```

## Documentation

- **[EVIDENCE_PIPELINE_DOCS.md](EVIDENCE_PIPELINE_DOCS.md)** - Full API reference and detailed guide
- **[QUICK_REFERENCE.py](QUICK_REFERENCE.py)** - Copy-paste code examples

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Out of Memory | Reduce `top_k_frames`, use smaller model |
| Low OCR scores | Check video quality, adjust weights |
| Slow inference | Reduce frames, enable Flash Attention |
| Missing OCR | Install PaddleOCR: `pip install paddleocr` |

## Files Overview

| File | Purpose |
|------|---------|
| `parsing/question_parser.py` | Extract semantic question components |
| `retrieval/frame_retrieval.py` | VLM-based frame relevance scoring |
| `retrieval/region_localization.py` | Object-centric grounding in frames |
| `retrieval/ocr_visibility.py` | OCR quality assessment & crop selection |
| `reasoning/qwen_reasoning.py` | Multi-modal answer generation |
| `pipeline/evidence_pipeline.py` | Orchestrates all stages |
| `utils/prompt_builder.py` | Prompt templates for each stage |
| `pipeline_integration_example.py` | End-to-end integration example |

## Next Steps

1. **Try the basic example**: `QUICK_REFERENCE.py`
2. **Read the documentation**: `EVIDENCE_PIPELINE_DOCS.md`
3. **Integrate with your code**: `pipeline_integration_example.py`
4. **Customize as needed**: Adjust weights, prompts, frame counts

## Future Improvements

- SAM2-based temporal object tracking
- Multi-frame evidence aggregation
- Adaptive region proposal sizing
- Hierarchical memory retrieval
- Question-guided temporal tracking
- Cross-frame OCR fusion

---

**Created**: 2024  
**Based on**: SFA VideoQA Framework  
**Model**: Qwen2.5-VL-7B-Instruct
