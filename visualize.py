import html
import json
from pathlib import Path


INPUT_PATH = Path("rewrite_success_cases.json")
OUTPUT_PATH = Path("analysis_rewrite.html")


def fmt(value):
    if value is None:
        return "N/A"
    return value


def esc(value):
    return html.escape(str(value), quote=True)


def rect(start, end, color, y, offset, scale):
    x = offset + start * scale
    w = max((end - start) * scale, 4)
    return (
        f'  <rect x="{round(x, 1)}" y="{y}" width="{round(w, 1)}" '
        f'height="14" fill="{color}" rx="2" opacity="0.85"/>\n'
    )


def label(text, y):
    return f'  <text x="2" y="{y + 11}" font-size="10" fill="#475569">{esc(text)}</text>\n'


def timeline_bar_compare(baseline_pred, our_pred, gt_list, width=400):
    intervals = list(gt_list or [])
    if baseline_pred:
        intervals.append(baseline_pred)
    if our_pred:
        intervals.append(our_pred)

    max_end = max((float(pair[1]) for pair in intervals if pair and len(pair) >= 2), default=150)
    total = max(150, int(max_end + 10))
    scale = width / total
    offset = 60
    svg_width = width + offset
    total_h = 64

    lines = []
    lines.append(f'<svg width="{svg_width}" height="{total_h}" style="margin-top:8px;display:block;">\n')
    lines.append(f'  <rect width="{svg_width}" height="{total_h}" fill="#f8fafc" rx="4"/>\n')
    lines.append(label("GT", 4))
    for g in gt_list or []:
        lines.append(rect(g[0], g[1], "#22c55e", 4, offset, scale))
    lines.append(label("Base", 24))
    if baseline_pred:
        lines.append(rect(baseline_pred[0], baseline_pred[1], "#f97316", 24, offset, scale))
    lines.append(label("Ours", 44))
    if our_pred:
        lines.append(rect(our_pred[0], our_pred[1], "#3b82f6", 44, offset, scale))
    lines.append("</svg>\n")
    return "".join(lines)


def normalize_refined_query(refined_query):
    if isinstance(refined_query, list):
        return " / ".join(str(item) for item in refined_query if item)
    return refined_query or ""


def make_card(r):
    pred = r.get("top1_pred") or []
    gt = r.get("gt") or []
    baseline_pred = r.get("baseline_pred") or []
    baseline_iou = r.get("baseline_iou")
    top1_caption = r.get("top1_caption", "")
    refined_query = normalize_refined_query(r.get("refined_query", ""))

    is_correct = bool(r.get("correct"))
    bg = "#f0fdf4" if is_correct else "#fff5f5"
    correct_bg = "#dcfce7" if is_correct else "#fee2e2"
    correct_color = "#166534" if is_correct else "#991b1b"
    correct_text = "Ours 맞음" if is_correct else "Ours 틀림"
    result_mark = "OK" if is_correct else "MISS"
    timeline = timeline_bar_compare(baseline_pred, pred, gt)

    qid = r.get("qid", "")
    iou = r.get("iou")
    similarity = r.get("similarity")
    baseline_sort = baseline_iou if baseline_iou is not None else -1
    baseline_correct = r.get("baseline_correct")
    baseline_correct_value = "unknown" if baseline_correct is None else str(bool(baseline_correct)).lower()

    lines = []
    lines.append(
        f'<article class="case-card" data-qid="{esc(qid)}" data-iou="{esc(iou or 0)}" '
        f'data-similarity="{esc(similarity or 0)}" data-baseline-iou="{esc(baseline_sort)}" '
        f'data-correct="{str(is_correct).lower()}" data-baseline-correct="{baseline_correct_value}" '
        f'style="background:{bg};">\n'
    )
    lines.append('  <div class="badge-row">\n')
    lines.append(f'    <span class="badge badge-qid">vid/qid: {esc(qid)}</span>\n')
    lines.append(f'    <span class="badge">유사도: {esc(fmt(similarity))}</span>\n')
    lines.append(f'    <span class="badge badge-base">Base IoU: {esc(fmt(baseline_iou))}</span>\n')
    lines.append(
        f'    <span class="badge" style="background:{correct_bg};color:{correct_color};">'
        f'{result_mark} {correct_text} (IoU: {esc(fmt(iou))})</span>\n'
    )
    lines.append("  </div>\n")
    lines.append('  <table class="case-table">\n')
    lines.append(f'    <tr><td>원본 쿼리</td><td>{esc(r.get("original_query", ""))}</td></tr>\n')
    lines.append(f'    <tr><td>Refined Query</td><td class="text-primary">{esc(refined_query)}</td></tr>\n')
    lines.append(f'    <tr><td>Top-1 Caption</td><td class="caption">{esc(top1_caption)}</td></tr>\n')
    lines.append("  </table>\n")
    lines.append(timeline)
    lines.append("</article>\n")
    return "".join(lines)


def make_html(results):
    total = len(results)

    lines = []
    lines.append("<!DOCTYPE html>\n")
    lines.append('<html lang="ko">\n')
    lines.append("<head>\n")
    lines.append('  <meta charset="utf-8">\n')
    lines.append('  <meta name="viewport" content="width=device-width, initial-scale=1">\n')
    lines.append("  <title>Rewrite Success Viewer</title>\n")
    lines.append(
        """  <style>
    * { box-sizing: border-box; }
    body {
      max-width: 1080px;
      margin: 32px auto;
      padding: 0 24px 48px;
      background: #f1f5f9;
      color: #0f172a;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    h1 { margin: 0 0 18px; color: #1E2761; font-size: 30px; }
    .toolbar {
      position: sticky;
      top: 0;
      z-index: 10;
      display: grid;
      grid-template-columns: minmax(220px, 1.2fr) repeat(3, minmax(150px, 0.6fr));
      gap: 10px;
      align-items: end;
      padding: 14px;
      margin-bottom: 18px;
      background: rgba(248, 250, 252, 0.96);
      border: 1px solid #cbd5e1;
      border-radius: 8px;
      box-shadow: 0 2px 10px rgba(15, 23, 42, 0.08);
      backdrop-filter: blur(8px);
    }
    label { display: grid; gap: 5px; color: #475569; font-size: 12px; font-weight: 600; }
    input, select {
      width: 100%;
      min-height: 36px;
      border: 1px solid #cbd5e1;
      border-radius: 6px;
      background: #fff;
      color: #0f172a;
      padding: 7px 9px;
      font: inherit;
      font-size: 14px;
    }
    .stats {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin: 0 0 18px;
      color: #475569;
      font-size: 13px;
    }
    .stat-pill {
      background: #fff;
      border: 1px solid #cbd5e1;
      border-radius: 6px;
      padding: 6px 10px;
    }
    .case-card {
      border-radius: 12px;
      padding: 20px;
      margin-bottom: 16px;
      box-shadow: 0 1px 4px rgba(0,0,0,0.07);
    }
    .badge-row { display: flex; gap: 8px; margin-bottom: 12px; align-items: center; flex-wrap: wrap; }
    .badge { background: #e2e8f0; border-radius: 6px; padding: 3px 10px; font-size: 12px; }
    .badge-qid { background: #1E2761; color: #fff; }
    .badge-base { background: #fef3c7; color: #92400e; }
    .case-table { width: 100%; border-collapse: collapse; font-size: 13px; margin-bottom: 10px; }
    .case-table tr:nth-child(even) { background: rgba(0,0,0,0.03); }
    .case-table td { padding: 6px 8px; vertical-align: top; }
    .case-table td:first-child { width: 118px; color: #64748b; }
    .text-primary { color: #1E2761; font-weight: 600; }
    .text-blue { color: #1E40AF; font-weight: 600; }
    .text-orange { color: #ea580c; }
    .caption { color: #64748b; font-size: 11px; line-height: 1.45; }
    .empty {
      display: none;
      padding: 40px 16px;
      text-align: center;
      color: #64748b;
      background: #fff;
      border: 1px dashed #cbd5e1;
      border-radius: 8px;
    }
    @media (max-width: 820px) {
      body { margin-top: 20px; padding: 0 14px 36px; }
      .toolbar { grid-template-columns: 1fr 1fr; }
    }
    @media (max-width: 560px) {
      .toolbar { grid-template-columns: 1fr; }
      .case-card { padding: 14px; }
      svg { max-width: 100%; height: auto; }
    }
  </style>
"""
    )
    lines.append("</head>\n")
    lines.append("<body>\n")
    lines.append("  <h1>Rewrite Success Viewer</h1>\n")
    lines.append('  <section class="toolbar" aria-label="filters">\n')
    lines.append('    <label>vid/qid 검색<input id="searchInput" type="search" placeholder="예: 8636"></label>\n')
    lines.append('    <label>점수 기준<select id="scoreKey">\n')
    lines.append('      <option value="iou">Ours IoU</option>\n')
    lines.append('      <option value="similarity">유사도</option>\n')
    lines.append('      <option value="baseline-iou">Baseline IoU</option>\n')
    lines.append("    </select></label>\n")
    lines.append('    <label>정렬<select id="sortOrder">\n')
    lines.append('      <option value="asc">낮은순</option>\n')
    lines.append('      <option value="desc">높은순</option>\n')
    lines.append("    </select></label>\n")
    lines.append('    <label>결과<select id="correctFilter">\n')
    lines.append('      <option value="all">전체</option>\n')
    lines.append('      <option value="true">Ours 맞음</option>\n')
    lines.append('      <option value="false">Ours 틀림</option>\n')
    lines.append('      <option value="base_true_ours_false">Baseline 맞음 → Ours 틀림</option>\n')
    lines.append('      <option value="base_false_ours_true">Baseline 틀림 → Ours 맞음</option>\n')
    lines.append("    </select></label>\n")
    lines.append("  </section>\n")
    lines.append('  <div class="stats">\n')
    lines.append(f'    <span class="stat-pill">전체: {total}</span>\n')
    lines.append('    <span class="stat-pill">현재 표시: <strong id="visibleCount"></strong></span>\n')
    lines.append("  </div>\n")
    lines.append('  <main id="caseList">\n')
    for r in results:
        lines.append(make_card(r))
    lines.append("  </main>\n")
    lines.append('  <div id="emptyState" class="empty">조건에 맞는 케이스가 없습니다.</div>\n')
    lines.append(
        """  <script>
    const caseList = document.getElementById("caseList");
    const cards = Array.from(document.querySelectorAll(".case-card"));
    const searchInput = document.getElementById("searchInput");
    const scoreKey = document.getElementById("scoreKey");
    const sortOrder = document.getElementById("sortOrder");
    const correctFilter = document.getElementById("correctFilter");
    const visibleCount = document.getElementById("visibleCount");
    const emptyState = document.getElementById("emptyState");

    function numberValue(card, key) {
      const raw = key === "baseline-iou" ? card.getAttribute("data-baseline-iou") : card.dataset[key];
      const value = Number(raw);
      return Number.isFinite(value) ? value : -1;
    }

    function applyFilters() {
      const query = searchInput.value.trim().toLowerCase();
      const correctness = correctFilter.value;
      const key = scoreKey.value;
      const direction = sortOrder.value === "asc" ? 1 : -1;

      const sorted = [...cards].sort((a, b) => {
        const diff = numberValue(a, key) - numberValue(b, key);
        if (diff !== 0) return diff * direction;
        return Number(a.dataset.qid) - Number(b.dataset.qid);
      });

      let shown = 0;
      for (const card of sorted) {
        const matchesQuery = !query || card.dataset.qid.toLowerCase().includes(query);
        let matchesCorrect = correctness === "all" || card.dataset.correct === correctness;
        if (correctness === "base_true_ours_false") {
          matchesCorrect = card.dataset.baselineCorrect === "true" && card.dataset.correct === "false";
        }
        if (correctness === "base_false_ours_true") {
          matchesCorrect = card.dataset.baselineCorrect === "false" && card.dataset.correct === "true";
        }
        const visible = matchesQuery && matchesCorrect;
        card.style.display = visible ? "" : "none";
        if (visible) shown += 1;
        caseList.appendChild(card);
      }

      visibleCount.textContent = shown;
      emptyState.style.display = shown === 0 ? "block" : "none";
    }

    searchInput.addEventListener("input", applyFilters);
    scoreKey.addEventListener("change", applyFilters);
    sortOrder.addEventListener("change", applyFilters);
    correctFilter.addEventListener("change", applyFilters);
    applyFilters();
  </script>
"""
    )
    lines.append("</body>\n")
    lines.append("</html>\n")
    return "".join(lines)


with INPUT_PATH.open(encoding="utf-8") as f:
    results = json.load(f)

with OUTPUT_PATH.open("w", encoding="utf-8") as f:
    f.write(make_html(results))

print(f"{OUTPUT_PATH} 생성 완료")
print(f"표시 케이스: {len(results)}개")
