import jsonlines
import json

def iou(pred, gt):
    start1, end1 = pred[0], pred[1]
    start2, end2 = gt[0], gt[1]
    inter = max(0, min(end1, end2) - max(start1, start2))
    union = (end1 - start1) + (end2 - start2) - inter
    return inter / union if union > 0 else 0.0

# 파일 로드
refined_dict = {}
with jsonlines.open("data/qvhighlights/query/val_semantic.jsonl") as r:
    for line in r:
        qid = str(line["qid"])
        refined_dict[qid] = {
            "action": line.get("action"),
            "similarity": line.get("similarity"),
            "original_query": line.get("original_query"),
            "refined_query": line.get("refined_query"),
            "top1_caption": line.get("top1_caption"),
        }

infer_dict = {}
with jsonlines.open("outputs/qvhighlights/infer_val.jsonl") as r:
    for line in r:
        qid = str(line["qid"])
        infer_dict[qid] = {
            "pred": line.get("pred_relevant_windows", []),
            "gt": line.get("relevant_windows", []),
        }

# 베이스라인 로드
baseline_dict = {}
with jsonlines.open("outputs/qvhighlights/infer_val_7B.jsonl") as r:
    for line in r:
        qid = str(line["qid"])
        pred_list = line.get("pred_relevant_windows", [])
        gt_list = line.get("relevant_windows", [])
        if pred_list and gt_list:
            top1_pred = pred_list[0][:2]
            max_iou = max(iou(top1_pred, gt) for gt in gt_list)
            baseline_dict[qid] = {
                "pred": top1_pred,
                "correct": max_iou >= 0.5,
                "iou": round(max_iou, 3),
            }

# 분석
results = []
for qid in refined_dict:
    if qid not in infer_dict:
        continue
    r = refined_dict[qid]
    p = infer_dict[qid]

    pred_list = p["pred"]
    gt_list = p["gt"]

    if not pred_list or not gt_list:
        continue

    top1_pred = pred_list[0][:2]
    max_iou = max(iou(top1_pred, gt) for gt in gt_list)

    baseline = baseline_dict.get(qid, {})

    results.append({
        "qid": qid,
        "action": r["action"],
        "similarity": r["similarity"],
        "original_query": r["original_query"],
        "refined_query": r["refined_query"],
        "top1_caption": r["top1_caption"],
        "top1_pred": top1_pred,
        "gt": gt_list,
        "iou": round(max_iou, 3),
        "correct": max_iou >= 0.5,
        "baseline_pred": baseline.get("pred", None),
        "baseline_iou": baseline.get("iou", None),
        "baseline_correct": baseline.get("correct", None),
    })

# 결과 저장
with open("analysis_results.json", "w") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

# 통계
rewrite_correct = [r for r in results if r["action"] == "rewrite" and r["correct"]]
rewrite_wrong = [r for r in results if r["action"] == "rewrite" and not r["correct"]]
keep_correct = [r for r in results if r["action"] == "keep" and r["correct"]]
keep_wrong = [r for r in results if r["action"] == "keep" and not r["correct"]]

print(f"rewrite 성공 + 맞음: {len(rewrite_correct)}")
print(f"rewrite 성공 + 틀림: {len(rewrite_wrong)}")
print(f"keep + 맞음:         {len(keep_correct)}")
print(f"keep + 틀림:         {len(keep_wrong)}")

keep_high = [r for r in results if r["action"] == "keep" and r["similarity"] >= 0.7]
keep_low  = [r for r in results if r["action"] == "keep" and r["similarity"] <= 0.4]

print(f"keep_high 맞음: {sum(1 for r in keep_high if r['correct'])} / {len(keep_high)}")
print(f"keep_low  맞음: {sum(1 for r in keep_low  if r['correct'])} / {len(keep_low)}")

# failure case 분류
ours_win  = [r for r in results if r["action"] == "rewrite"
             and r["baseline_correct"] == False and r["correct"] == True]
ours_lose = [r for r in results if r["action"] == "rewrite"
             and r["baseline_correct"] == True and r["correct"] == False]

print(f"\n[Failure Case 분석]")
print(f"베이스라인 틀림 → 우리 맞음 (우리가 구한 케이스): {len(ours_win)}")
print(f"베이스라인 맞음 → 우리 틀림 (우리가 망친 케이스): {len(ours_lose)}")

# 케이스 별도 저장
with open("failure_cases.json", "w") as f:
    json.dump({
        "ours_win": ours_win,
        "ours_lose": ours_lose,
    }, f, ensure_ascii=False, indent=2)

print("failure_cases.json 저장 완료")