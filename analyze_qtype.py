import re, sys
from collections import defaultdict

lines = open(sys.argv[1]).read()
blocks = re.findall(
    r'\[OCR DEBUG\] Q: (.+?)\n\[OCR DEBUG\] OCR prefix: (.+?)\nGT:  (.+?)   Pred:  (.+)',
    lines
)

text_kw = ['written', 'write', 'text', 'word', 'brand', 'name of', 'title',
           'say', 'said', 'read', 'spell', 'letter', 'sign', 'label', 'slogan', 'logo']
num_kw = ['price', 'cost', 'number', 'how much', 'how many', 'score',
          'total', 'amount', 'plate', 'date', 'time', 'year', 'floor', 'channel']
visual_kw = ['color', 'colour', 'wear', 'look like', 'doing', 'action',
             'where is', 'who is', 'expression']

def classify(q):
    ql = q.lower()
    for kw in text_kw:
        if kw in ql:
            return 'text_related'
    for kw in num_kw:
        if kw in ql:
            return 'number_related'
    for kw in visual_kw:
        if kw in ql:
            return 'visual'
    return 'other'

stats = defaultdict(lambda: {'total': 0, 'gt_in_ocr': 0, 'pred_correct': 0, 'both': 0})

for q, ocr, gt_raw, pred in blocks:
    cat = classify(q)
    gt_answers = re.findall(r"'([^']+)'", gt_raw)
    ocr_lower = ocr.lower()
    gt_in_ocr = any(a.lower() in ocr_lower for a in gt_answers)
    pred_correct = any(pred.lower().strip() == a.lower().strip() for a in gt_answers)
    stats[cat]['total'] += 1
    if gt_in_ocr:
        stats[cat]['gt_in_ocr'] += 1
    if pred_correct:
        stats[cat]['pred_correct'] += 1
    if gt_in_ocr and pred_correct:
        stats[cat]['both'] += 1

print(f"{'Category':<18} {'Total':>5}  {'GT_in_OCR':>12}  {'Pred_OK':>12}  {'GT_in_OCR&OK':>14}")
print('-' * 70)
for cat in ['text_related', 'number_related', 'visual', 'other']:
    s = stats[cat]
    t = max(s['total'], 1)
    print(f"{cat:<18} {s['total']:>5}  "
          f"{s['gt_in_ocr']:>4} ({100*s['gt_in_ocr']/t:>4.0f}%)  "
          f"{s['pred_correct']:>4} ({100*s['pred_correct']/t:>4.0f}%)  "
          f"{s['both']:>4} ({100*s['both']/t:>4.0f}%)")

# also show "other" question examples
print("\n=== 'other' category questions ===")
for q, ocr, gt_raw, pred in blocks:
    if classify(q) == 'other':
        print(f"  Q: {q}")
