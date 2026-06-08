import json

with open("analysis_results.json") as f:
    results = json.load(f)

ours_win_all  = [r for r in results if r["action"] == "rewrite"
                 and r["baseline_correct"] == False and r["correct"] == True]
ours_lose_all = [r for r in results if r["action"] == "rewrite"
                 and r["baseline_correct"] == True and r["correct"] == False]

ours_win  = ours_win_all
ours_lose = ours_lose_all


def rect(start, end, color, y, offset, scale):
    x = offset + start * scale
    w = max((end - start) * scale, 4)
    return f'  <rect x="{round(x,1)}" y="{y}" width="{round(w,1)}" height="14" fill="{color}" rx="2" opacity="0.85"/>\n'

def label(text, y):
    return f'  <text x="2" y="{y+11}" font-size="10" fill="#475569">{text}</text>\n'

def timeline_bar_compare(baseline_pred, our_pred, gt_list, total=150, width=400):
    scale = width / total
    offset = 60
    svg_width = width + offset
    total_h = 64

    lines = []
    lines.append(f'<svg width="{svg_width}" height="{total_h}" style="margin-top:8px;display:block;">\n')
    lines.append(f'  <rect width="{svg_width}" height="{total_h}" fill="#f8fafc" rx="4"/>\n')
    lines.append(label("GT", 4))
    for g in gt_list:
        lines.append(rect(g[0], g[1], "#22c55e", 4, offset, scale))
    lines.append(label("Base", 24))
    lines.append(rect(baseline_pred[0], baseline_pred[1], "#f97316", 24, offset, scale))
    lines.append(label("Ours", 44))
    lines.append(rect(our_pred[0], our_pred[1], "#3b82f6", 44, offset, scale))
    lines.append("</svg>\n")
    return "".join(lines)

def make_card(r, bg):
    pred = r["top1_pred"]
    gt = r["gt"]
    baseline_pred = r.get("baseline_pred") or [0, 0]
    baseline_iou = r.get("baseline_iou", "N/A")
    top1_caption = r.get("top1_caption", "")

    # refined_query가 리스트인 경우 Simplified/Detailed 분리
    refined_query = r.get("refined_query", "")
    if isinstance(refined_query, list):
        simplified = refined_query[0] if len(refined_query) > 0 else ""
        detailed   = refined_query[1] if len(refined_query) > 1 else ""
    else:
        simplified = refined_query
        detailed   = ""

    correct_bg = "#dcfce7" if r["correct"] else "#fee2e2"
    correct_color = "#166534" if r["correct"] else "#991b1b"
    correct_text = "✓ Ours 맞음" if r["correct"] else "✗ Ours 틀림"
    timeline = timeline_bar_compare(baseline_pred, pred, gt)

    lines = []
    lines.append(f'<div style="background:{bg};border-radius:12px;padding:20px;margin-bottom:16px;font-family:sans-serif;box-shadow:0 1px 4px rgba(0,0,0,0.07);">\n')
    lines.append(f'  <div style="display:flex;gap:8px;margin-bottom:12px;align-items:center;flex-wrap:wrap;">\n')
    lines.append(f'    <span style="background:#1E2761;color:#fff;border-radius:6px;padding:3px 10px;font-size:12px;">qid: {r["qid"]}</span>\n')
    lines.append(f'    <span style="background:#e2e8f0;border-radius:6px;padding:3px 10px;font-size:12px;">유사도: {r["similarity"]}</span>\n')
    lines.append(f'    <span style="background:#fef3c7;border-radius:6px;padding:3px 10px;font-size:12px;color:#92400e;">Base IoU: {baseline_iou}</span>\n')
    lines.append(f'    <span style="background:{correct_bg};border-radius:6px;padding:3px 10px;font-size:12px;color:{correct_color};">{correct_text} (IoU: {r["iou"]})</span>\n')
    lines.append(f'  </div>\n')
    lines.append(f'  <table style="width:100%;border-collapse:collapse;font-size:13px;margin-bottom:10px;">\n')
    lines.append(f'    <tr><td style="padding:6px 8px;color:#64748b;width:110px;">원본 쿼리</td><td style="padding:6px 8px;">{r["original_query"]}</td></tr>\n')
    lines.append(f'    <tr style="background:rgba(0,0,0,0.03)"><td style="padding:6px 8px;color:#64748b;">Simplified</td><td style="padding:6px 8px;color:#1E2761;font-weight:500;">{simplified}</td></tr>\n')
    lines.append(f'    <tr><td style="padding:6px 8px;color:#64748b;">Detailed</td><td style="padding:6px 8px;color:#1E40AF;font-weight:500;">{detailed}</td></tr>\n')
    lines.append(f'    <tr style="background:rgba(0,0,0,0.03)"><td style="padding:6px 8px;color:#64748b;vertical-align:top;">Top-1 Caption</td><td style="padding:6px 8px;color:#64748b;font-size:11px;">{top1_caption}</td></tr>\n')
    lines.append(f'    <tr><td style="padding:6px 8px;color:#64748b;">GT 구간</td><td style="padding:6px 8px;">{gt}</td></tr>\n')
    lines.append(f'    <tr style="background:rgba(0,0,0,0.03)"><td style="padding:6px 8px;color:#64748b;">Baseline 예측</td><td style="padding:6px 8px;color:#ea580c;">{baseline_pred}</td></tr>\n')
    lines.append(f'    <tr><td style="padding:6px 8px;color:#64748b;">Ours 예측</td><td style="padding:6px 8px;color:#2563eb;">{pred}</td></tr>\n')
    lines.append(f'  </table>\n')
    lines.append(timeline)
    lines.append(f'</div>\n')
    return "".join(lines)

def make_section(title, cases, bg, color):
    lines = []
    lines.append(f'<div style="margin-bottom:48px;">\n')
    lines.append(f'  <h2 style="color:{color};font-family:sans-serif;border-left:4px solid {color};padding-left:12px;">{title}</h2>\n')
    for r in cases:
        lines.append(make_card(r, bg))
    lines.append(f'</div>\n')
    return "".join(lines)

lines = []
lines.append('<!DOCTYPE html>\n')
lines.append('<html>\n')
lines.append('<head>\n')
lines.append('  <meta charset="utf-8">\n')
lines.append('  <title>Failure Case Analysis</title>\n')
lines.append('</head>\n')
lines.append('<body style="max-width:960px;margin:40px auto;padding:0 24px;background:#f1f5f9;">\n')
lines.append('<h1 style="font-family:sans-serif;color:#1E2761;">Failure Case Analysis</h1>\n')
lines.append('<p style="font-family:sans-serif;color:#64748b;margin-bottom:32px;">🟢 GT &nbsp;|&nbsp; 🟠 Baseline &nbsp;|&nbsp; 🔵 Ours</p>\n')
lines.append(make_section(
    f"✅ 베이스라인 틀림 → ours 맞음 ({len(ours_win_all)}개 전체)",
    ours_win, "#f0fdf4", "#166534"
))
lines.append(make_section(
    f"❌ 베이스라인 맞음 → ours 틀림 ({len(ours_lose_all)}개 전체)",
    ours_lose, "#fff5f5", "#dc2626"
))
lines.append('</body>\n')
lines.append('</html>\n')

with open("analysis.html", "w", encoding="utf-8") as f:
    f.writelines(lines)

print(f"✅ analysis.html 생성 완료!")
print(f"   Win cases : {len(ours_win_all)}개")
print(f"   Lose cases: {len(ours_lose_all)}개")