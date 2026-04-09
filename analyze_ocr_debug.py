import re
import sys

lines = open(sys.argv[1]).read()
ocr_blocks = re.findall(
    r'\[OCR DEBUG\] Q: (.+?)\n\[OCR DEBUG\] OCR prefix: (.+?)\nGT:  (.+?)   Pred:  (.+)',
    lines
)

hit, miss, total = 0, 0, len(ocr_blocks)
hit_but_wrong, miss_and_wrong = 0, 0
ocr_len_sum = 0
examples_miss = []
examples_hit_wrong = []

for q, ocr, gt_raw, pred in ocr_blocks:
    gt_answers = re.findall(r"'([^']+)'", gt_raw)
    ocr_lower = ocr.lower()
    ocr_items = re.findall(r'"([^"]+)"', ocr)
    ocr_len_sum += len(ocr_items)

    gt_in_ocr = any(a.lower() in ocr_lower for a in gt_answers)
    pred_correct = any(pred.lower().strip() == a.lower().strip() for a in gt_answers)

    if gt_in_ocr:
        hit += 1
        if not pred_correct:
            hit_but_wrong += 1
            examples_hit_wrong.append((q, gt_answers, pred, ocr_items[:5]))
    else:
        miss += 1
        if not pred_correct:
            miss_and_wrong += 1
            examples_miss.append((q, gt_answers, pred, ocr_items[:5]))

print(f"Total QA with OCR: {total}")
print(f"GT answer IN OCR text: {hit}/{total} ({100*hit/total:.1f}%)")
print(f"GT answer NOT in OCR:  {miss}/{total} ({100*miss/total:.1f}%)")
print(f"Avg OCR items per question: {ocr_len_sum/total:.1f}")
print()
print(f"GT in OCR but pred WRONG: {hit_but_wrong}/{hit}")
print(f"GT NOT in OCR and pred WRONG: {miss_and_wrong}/{miss}")
print()
print("=== Examples: GT in OCR but pred wrong ===")
for q, gt, pred, ocr5 in examples_hit_wrong[:8]:
    print(f"  Q: {q}")
    print(f"  GT: {gt}, Pred: {pred}")
    print(f"  OCR(top5): {ocr5}")
    print()
print("=== Examples: GT NOT in OCR, pred wrong ===")
for q, gt, pred, ocr5 in examples_miss[:8]:
    print(f"  Q: {q}")
    print(f"  GT: {gt}, Pred: {pred}")
    print(f"  OCR(top5): {ocr5}")
