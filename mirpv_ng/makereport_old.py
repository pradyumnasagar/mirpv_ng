#!/usr/bin/env python3
"""
miRPV-NG sRNA-seq report generator (HTML + optional PDF)

Enhancements implemented:
- Clear headline counts: Known-Confirmed / Known-Atypical / Novel-High
- Summary tables + top-N lists (Novel-High prioritized)
- Plots (PNG): label counts, RF score histogram, reject reasons/stages
- Structure gallery: attempts RNAplot SVG per top candidates (if ViennaRNA RNAplot is available)
- Always non-interactive matplotlib backend (no Qt/Wayland issues)

Inputs expected (as produced by your ladder):
- final_candidates.tsv (Stage 13 output)
- final_report.json (Stage 14/15 summary JSON, optional but supported)
- candidates_struct.tsv (Stage 12 output: includes seq + dotbracket + mfe)
- mature_predictions.tsv (Stage 12.5 output, optional for extra columns)
- rejects.merged.tsv (merged rejects across stages, optional but supported)
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import html
import json
import os
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# Force non-GUI backend (prevents Qt/wayland problems)
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt  # noqa: E402


# -----------------------------
# Utilities
# -----------------------------

def die(msg: str, code: int = 2) -> None:
    print(f"[make-report] ERROR: {msg}", file=sys.stderr)
    raise SystemExit(code)


def read_tsv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        die(f"Missing file: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f, delimiter="\t")
        return [row for row in r]


def safe_float(x: str, default: Optional[float] = None) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return default


def safe_int(x: str, default: Optional[int] = None) -> Optional[int]:
    try:
        return int(float(x))
    except Exception:
        return default


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, s: str) -> None:
    ensure_dir(path.parent)
    path.write_text(s, encoding="utf-8")


def which(cmd: str) -> Optional[str]:
    return shutil.which(cmd)


def run_cmd(cmd: List[str], cwd: Optional[Path] = None) -> Tuple[int, str, str]:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return proc.returncode, proc.stdout, proc.stderr


def fmt_int(n: Optional[int]) -> str:
    return "NA" if n is None else f"{n:,}"


def fmt_float(x: Optional[float], nd: int = 3) -> str:
    if x is None:
        return "NA"
    return f"{x:.{nd}f}"


# -----------------------------
# Data models
# -----------------------------

@dataclass
class StructRec:
    candidate_id: str
    seq: str
    dotbracket: str
    mfe: Optional[float]
    final_label: Optional[str]
    best_rf_score: Optional[float]


# -----------------------------
# Plot helpers
# -----------------------------

def save_bar(counter: Counter, title: str, out_png: Path, xlabel: str = "", ylabel: str = "count") -> None:
    ensure_dir(out_png.parent)
    labels = list(counter.keys())
    vals = [counter[k] for k in labels]

    plt.figure(figsize=(8, 4))
    plt.bar(range(len(labels)), vals)
    plt.xticks(range(len(labels)), labels, rotation=30, ha="right")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def save_hist(values: List[float], title: str, out_png: Path, bins: int = 30, xlabel: str = "") -> None:
    ensure_dir(out_png.parent)
    plt.figure(figsize=(8, 4))
    plt.hist(values, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


# -----------------------------
# Structure rendering
# -----------------------------

def try_rnaplot_svg(seq: str, dot: str, out_svg: Path, workdir: Path) -> Tuple[bool, str]:
    """
    Generates RNAplot SVG if available.
    Uses ViennaRNA RNAplot; writes a temporary file with:
      >id
      SEQ
      DOTBRACKET (and mfe is optional)
    """
    rnaplot = which("RNAplot")
    if not rnaplot:
        return False, "RNAplot not found in PATH"

    ensure_dir(workdir)
    ensure_dir(out_svg.parent)

    # RNAplot reads from stdin: "seq\nstructure\n"
    # For SVG output: RNAplot -o svg
    cmd = [rnaplot, "-o", "svg"]

    rc, out, err = run_cmd(cmd, cwd=workdir)
    # That command without stdin is not correct; use subprocess with input below:
    proc = subprocess.run(
        cmd,
        cwd=str(workdir),
        input=f"{seq}\n{dot}\n",
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        return False, proc.stderr.strip() or "RNAplot failed"

    # RNAplot writes "rna.ss.svg" (or similar) in cwd.
    # Find the newest svg.
    svgs = sorted(workdir.glob("*.svg"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not svgs:
        return False, "RNAplot produced no SVG"

    latest = svgs[0]
    out_svg.write_bytes(latest.read_bytes())
    # Clean other intermediate files to avoid clutter
    for p in workdir.glob("*"):
        if p.is_file() and p.name != out_svg.name:
            try:
                p.unlink()
            except Exception:
                pass

    return True, ""


# -----------------------------
# HTML assembly
# -----------------------------

def html_page(title: str, body: str) -> str:
    # Minimal CSS, readable, "report-like"
    css = """
    :root { --bg:#0b0f14; --fg:#e6edf3; --muted:#9aa4af; --card:#121923; --line:#233044; --accent:#7aa2ff; }
    body { margin:0; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; background: var(--bg); color: var(--fg); }
    .wrap { max-width: 1100px; margin: 0 auto; padding: 28px 18px 60px; }
    h1 { margin: 0 0 10px; font-size: 28px; letter-spacing: 0.2px; }
    h2 { margin: 26px 0 12px; font-size: 18px; color: var(--fg); }
    .sub { color: var(--muted); margin-bottom: 18px; }
    .grid { display:grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }
    .card { background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 12px 14px; }
    .k { color: var(--muted); font-size: 12px; }
    .v { font-size: 22px; margin-top: 4px; }
    .row { display:grid; grid-template-columns: 1fr 1fr; gap: 12px; }
    img { max-width:100%; border-radius: 10px; border:1px solid var(--line); background:#0a0d12; }
    table { width:100%; border-collapse: collapse; background: var(--card); border:1px solid var(--line); border-radius: 12px; overflow:hidden; }
    th, td { padding: 8px 10px; border-bottom: 1px solid var(--line); font-size: 13px; vertical-align: top; }
    th { text-align:left; color: var(--muted); font-weight: 600; }
    tr:hover td { background: rgba(122,162,255,0.08); }
    .pill { display:inline-block; padding: 2px 8px; border-radius: 999px; border:1px solid var(--line); font-size: 12px; color: var(--muted); }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New"; font-size: 12px; color: #cfe3ff; }
    details { background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 10px 12px; }
    summary { cursor:pointer; color: var(--fg); font-weight: 600; }
    .warn { color: #ffcc66; }
    a { color: var(--accent); text-decoration: none; }
    """
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{html.escape(title)}</title>
<style>{css}</style>
</head>
<body>
<div class="wrap">
{body}
</div>
</body>
</html>
"""


def html_table(rows: List[Dict[str, str]], cols: List[str], max_rows: int = 30) -> str:
    head = "".join(f"<th>{html.escape(c)}</th>" for c in cols)
    body_rows = []
    for row in rows[:max_rows]:
        tds = []
        for c in cols:
            v = row.get(c, "")
            tds.append(f"<td>{html.escape(str(v))}</td>")
        body_rows.append("<tr>" + "".join(tds) + "</tr>")
    body = "\n".join(body_rows) if body_rows else "<tr><td colspan='999' class='warn'>No rows</td></tr>"
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


# -----------------------------
# Main report logic
# -----------------------------

def build_report(
    sample_id: str,
    outdir: Path,
    final_candidates_tsv: Path,
    final_report_json: Optional[Path],
    candidates_struct_tsv: Path,
    mature_tsv: Optional[Path],
    rejects_merged_tsv: Optional[Path],
) -> Tuple[Path, Optional[Path], Optional[Path]]:
    ensure_dir(outdir)
    assets = outdir / "assets"
    ensure_dir(assets)

    # Load inputs
    final_rows = read_tsv(final_candidates_tsv)
    struct_rows = read_tsv(candidates_struct_tsv)
    mature_rows = read_tsv(mature_tsv) if (mature_tsv and mature_tsv.exists()) else []

    report_obj = None
    if final_report_json and final_report_json.exists():
        try:
            report_obj = json.loads(final_report_json.read_text(encoding="utf-8"))
        except Exception:
            report_obj = None

    rejects_rows = read_tsv(rejects_merged_tsv) if (rejects_merged_tsv and rejects_merged_tsv.exists()) else []

    # Summaries
    # Expect columns in final_candidates.tsv: final_label, known_status, best_rf_score etc (we will be defensive)
    label_key = "final_label" if final_rows and "final_label" in final_rows[0] else (
        "known_status" if final_rows and "known_status" in final_rows[0] else None
    )
    if not label_key:
        die(f"final_candidates.tsv has no 'final_label' (or 'known_status') column: {final_candidates_tsv}")

    label_counts = Counter((r.get(label_key, "") or "NA").strip() for r in final_rows)
    total_final = len(final_rows)

    # RF score histogram if available
    rf_key = None
    for k in ("best_rf_score", "rf_score"):
        if final_rows and k in final_rows[0]:
            rf_key = k
            break
    rf_vals = []
    if rf_key:
        for r in final_rows:
            v = safe_float((r.get(rf_key, "") or "").strip(), None)
            if v is not None:
                rf_vals.append(v)

    # Reject summaries
    # Common columns in rejects: stage, reason, etc. We'll try typical names.
    reject_stage_key = None
    reject_reason_key = None
    if rejects_rows:
        for k in ("stage", "reject_stage", "STEP", "Stage", "STAGE"):
            if k in rejects_rows[0]:
                reject_stage_key = k
                break
        for k in ("reason", "reject_reason", "REASON", "Reason"):
            if k in rejects_rows[0]:
                reject_reason_key = k
                break

    stage_counts = Counter()
    reason_counts = Counter()
    if rejects_rows and reject_stage_key:
        stage_counts = Counter((r.get(reject_stage_key, "") or "NA").strip() for r in rejects_rows)
    if rejects_rows and reject_reason_key:
        reason_counts = Counter((r.get(reject_reason_key, "") or "NA").strip() for r in rejects_rows)

    # Save plots
    plot_labels_png = assets / "labels.png"
    save_bar(label_counts, f"{sample_id}: Final label counts", plot_labels_png, xlabel="final_label")

    plot_rf_png = None
    if rf_vals:
        plot_rf_png = assets / "rf_score_hist.png"
        save_hist(rf_vals, f"{sample_id}: RF score distribution", plot_rf_png, bins=30, xlabel=rf_key)

    plot_reject_stage_png = None
    if stage_counts:
        # pick top 20 stages/reasons by count
        stage_counts_top = Counter(dict(stage_counts.most_common(20)))
        plot_reject_stage_png = assets / "reject_stages.png"
        save_bar(stage_counts_top, f"{sample_id}: Reject counts by stage (top20)", plot_reject_stage_png, xlabel="stage")

    plot_reject_reason_png = None
    if reason_counts:
        reason_counts_top = Counter(dict(reason_counts.most_common(20)))
        plot_reject_reason_png = assets / "reject_reasons.png"
        save_bar(reason_counts_top, f"{sample_id}: Reject reasons (top20)", plot_reject_reason_png, xlabel="reason")

    # Build struct index: candidate_id -> (seq, dot, mfe, final_label, best_rf_score)
    struct_index: Dict[str, StructRec] = {}
    # Determine keys in candidates_struct.tsv
    # Expected: candidate_id, seq, dotbracket, mfe, final_label, best_rf_score
    for row in struct_rows:
        cid = (row.get("candidate_id") or "").strip()
        if not cid:
            continue
        seq = (row.get("seq") or "").strip()
        dot = (row.get("dotbracket") or "").strip()
        mfe = safe_float((row.get("mfe") or "").strip(), None)
        fl = (row.get("final_label") or "").strip() or None
        brf = safe_float((row.get("best_rf_score") or row.get("rf_score") or "").strip(), None)
        struct_index[cid] = StructRec(candidate_id=cid, seq=seq, dotbracket=dot, mfe=mfe, final_label=fl, best_rf_score=brf)

    # Identify top candidates for structure gallery
    # Priority: Novel-High, then Known-Atypical, then Known-Confirmed
    cid_key = "candidate_id" if final_rows and "candidate_id" in final_rows[0] else None
    if not cid_key:
        die(f"final_candidates.tsv missing candidate_id column: {final_candidates_tsv}")

    # Define score key for sorting
    sort_key = rf_key or "best_rf_score"

    def row_score(r: Dict[str, str]) -> float:
        v = safe_float((r.get(sort_key, "") or "").strip(), None)
        return -1.0 if v is None else v

    def top_by_label(wanted_label: str, n: int) -> List[Dict[str, str]]:
        sub = [r for r in final_rows if (r.get(label_key, "") or "").strip() == wanted_label]
        sub.sort(key=row_score, reverse=True)
        return sub[:n]

    top_novel = top_by_label("Novel-High", 10) if "Novel-High" in label_counts else []
    top_atyp = top_by_label("Known-Atypical", 6) if "Known-Atypical" in label_counts else []
    top_known = top_by_label("Known-Confirmed", 6) if "Known-Confirmed" in label_counts else []

    gallery_rows = top_novel + top_atyp + top_known

    # Render structure SVGs via RNAplot (if possible)
    gallery_html_blocks = []
    rnaplot_ok_any = False
    workdir = outdir / ".rnaplot_tmp"
    ensure_dir(workdir)

    for r in gallery_rows:
        cid = (r.get(cid_key, "") or "").strip()
        if not cid:
            continue
        srec = struct_index.get(cid)
        if not srec or not srec.seq or not srec.dotbracket:
            # Structure missing: still show row info
            gallery_html_blocks.append(
                f"<details><summary>{html.escape(cid)}</summary>"
                f"<div class='warn'>No structure found in candidates_struct.tsv for this candidate_id.</div>"
                f"</details>"
            )
            continue

        svg_path = assets / f"struct_{abs(hash(cid))}.svg"
        ok, err = try_rnaplot_svg(srec.seq, srec.dotbracket, svg_path, workdir)
        if ok:
            rnaplot_ok_any = True
            svg_inline = svg_path.read_text(encoding="utf-8", errors="replace")
            # Inline svg directly (keeps report self-contained)
            block = f"""
<details>
  <summary>{html.escape(cid)} <span class="pill">{html.escape(r.get(label_key,''))}</span> <span class="pill">RF {fmt_float(row_score(r), 3)}</span> <span class="pill">MFE {fmt_float(srec.mfe, 2)}</span></summary>
  <div style="margin-top:10px">{svg_inline}</div>
  <div class="mono" style="margin-top:10px; white-space:pre-wrap">{html.escape(srec.seq)}</div>
  <div class="mono" style="margin-top:6px; white-space:pre-wrap">{html.escape(srec.dotbracket)}</div>
</details>
"""
            gallery_html_blocks.append(block)
        else:
            block = f"""
<details>
  <summary>{html.escape(cid)} <span class="pill">{html.escape(r.get(label_key,''))}</span> <span class="pill">RF {fmt_float(row_score(r), 3)}</span> <span class="pill">MFE {fmt_float(srec.mfe, 2)}</span></summary>
  <div class="warn" style="margin-top:10px">RNAplot unavailable for this environment ({html.escape(err)}). Showing sequence + dotbracket only.</div>
  <div class="mono" style="margin-top:10px; white-space:pre-wrap">{html.escape(srec.seq)}</div>
  <div class="mono" style="margin-top:6px; white-space:pre-wrap">{html.escape(srec.dotbracket)}</div>
</details>
"""
            gallery_html_blocks.append(block)

    # Clean temp
    try:
        if workdir.exists():
            for p in workdir.glob("*"):
                try:
                    p.unlink()
                except Exception:
                    pass
            try:
                workdir.rmdir()
            except Exception:
                pass
    except Exception:
        pass

    # Build top table for final candidates
    # Keep it compact: show the most useful columns if present
    preferred_cols = [
        "candidate_id", "peak_id", "chrom", "strand", "peak_center0",
        "final_label", "known_db", "known_id",
        "best_rf_score", "best_pred_label",
        "mature_arm", "mature_start0", "mature_end0", "mature_seq"
    ]
    existing_cols = [c for c in preferred_cols if final_rows and c in final_rows[0]]
    if not existing_cols:
        # fallback to first 10 columns
        existing_cols = list(final_rows[0].keys())[:10] if final_rows else []

    # Sort final candidates: Novel-High first, then score
    label_rank = {"Novel-High": 0, "Known-Atypical": 1, "Known-Confirmed": 2}
    def sort_tuple(r: Dict[str, str]) -> Tuple[int, float]:
        lbl = (r.get(label_key, "") or "").strip()
        return (label_rank.get(lbl, 99), -row_score(r))

    final_sorted = sorted(final_rows, key=sort_tuple)
    final_top_table = html_table(final_sorted, existing_cols, max_rows=50)

    # Compose HTML
    now = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    k_known_conf = label_counts.get("Known-Confirmed", 0)
    k_known_atyp = label_counts.get("Known-Atypical", 0)
    k_novel = label_counts.get("Novel-High", 0)

    body = f"""
<h1>miRPV-NG sRNA-seq report — {html.escape(sample_id)}</h1>
<div class="sub">Generated: {html.escape(now)} • Source: final_candidates.tsv + candidates_struct.tsv</div>

<div class="grid">
  <div class="card"><div class="k">Final candidates</div><div class="v">{fmt_int(total_final)}</div></div>
  <div class="card"><div class="k">Novel-High</div><div class="v">{fmt_int(k_novel)}</div></div>
  <div class="card"><div class="k">Known-Confirmed</div><div class="v">{fmt_int(k_known_conf)}</div></div>
  <div class="card"><div class="k">Known-Atypical</div><div class="v">{fmt_int(k_known_atyp)}</div></div>
</div>

<h2>Summary plots</h2>
<div class="row">
  <div class="card"><div class="k">Final label counts</div><div style="margin-top:10px"><img src="assets/{plot_labels_png.name}" alt="label counts"/></div></div>
  <div class="card">
    <div class="k">RF score distribution</div>
    <div style="margin-top:10px">
      {"<img src='assets/"+plot_rf_png.name+"' alt='rf histogram'/>" if plot_rf_png else "<div class='warn'>No RF score column found.</div>"}
    </div>
  </div>
</div>

<h2>Rejects overview</h2>
<div class="row">
  <div class="card">
    <div class="k">Rejects by stage (top20)</div>
    <div style="margin-top:10px">
      {"<img src='assets/"+plot_reject_stage_png.name+"' alt='reject stages'/>" if plot_reject_stage_png else "<div class='warn'>rejects.merged.tsv not provided (or missing stage column).</div>"}
    </div>
  </div>
  <div class="card">
    <div class="k">Reject reasons (top20)</div>
    <div style="margin-top:10px">
      {"<img src='assets/"+plot_reject_reason_png.name+"' alt='reject reasons'/>" if plot_reject_reason_png else "<div class='warn'>rejects.merged.tsv not provided (or missing reason column).</div>"}
    </div>
  </div>
</div>

<h2>Top candidates (preview)</h2>
<div class="sub">Sorted with Novel-High first, then RF score. Showing top 50.</div>
{final_top_table}

<h2>Structure gallery (top Novel-High + anchors)</h2>
<div class="sub">
  {"RNAplot detected — structures rendered as SVG." if rnaplot_ok_any else "RNAplot not detected — showing dotbrackets only (install ViennaRNA to render)."}
</div>
{"".join(gallery_html_blocks) if gallery_html_blocks else "<div class='warn'>No candidates available for gallery.</div>"}

<h2>Inputs</h2>
<div class="card">
  <div class="k">Files</div>
  <div class="mono" style="margin-top:10px; white-space:pre-wrap">
final_candidates.tsv: {html.escape(str(final_candidates_tsv))}
candidates_struct.tsv: {html.escape(str(candidates_struct_tsv))}
mature.tsv: {html.escape(str(mature_tsv)) if mature_tsv else "NA"}
rejects.merged.tsv: {html.escape(str(rejects_merged_tsv)) if rejects_merged_tsv else "NA"}
final_report.json: {html.escape(str(final_report_json)) if final_report_json else "NA"}
  </div>
</div>
"""

    html_out = outdir / "report.html"
    write_text(html_out, html_page(f"miRPV-NG report — {sample_id}", body))

    # Try PDF: prefer reportlab if present; otherwise skip gracefully.
    pdf_out = outdir / "report.pdf"
    pdf_error = outdir / "pdf_error.txt"
    pdf_ok = False

    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
        from reportlab.lib.units import inch
        from reportlab.lib.utils import ImageReader

        c = canvas.Canvas(str(pdf_out), pagesize=letter)
        w, h = letter

        y = h - 0.8 * inch
        c.setFont("Helvetica-Bold", 16)
        c.drawString(0.8 * inch, y, f"miRPV-NG sRNA-seq report — {sample_id}")
        y -= 0.35 * inch
        c.setFont("Helvetica", 10)
        c.drawString(0.8 * inch, y, f"Generated: {now}")
        y -= 0.35 * inch

        c.setFont("Helvetica-Bold", 12)
        c.drawString(0.8 * inch, y, "Headline counts")
        y -= 0.22 * inch
        c.setFont("Helvetica", 10)
        c.drawString(0.9 * inch, y, f"Final candidates: {total_final}")
        y -= 0.18 * inch
        c.drawString(0.9 * inch, y, f"Novel-High: {k_novel}")
        y -= 0.18 * inch
        c.drawString(0.9 * inch, y, f"Known-Confirmed: {k_known_conf}")
        y -= 0.18 * inch
        c.drawString(0.9 * inch, y, f"Known-Atypical: {k_known_atyp}")
        y -= 0.35 * inch

        # Embed plots if exist
        def draw_img(img_path: Path, y0: float) -> float:
            if not img_path.exists():
                return y0
            img = ImageReader(str(img_path))
            iw, ih = img.getSize()
            # Fit width
            max_w = w - 1.6 * inch
            scale = max_w / float(iw)
            new_w = max_w
            new_h = float(ih) * scale
            if y0 - new_h < 0.8 * inch:
                c.showPage()
                y0 = h - 0.8 * inch
            c.drawImage(img, 0.8 * inch, y0 - new_h, width=new_w, height=new_h)
            return y0 - new_h - 0.3 * inch

        c.setFont("Helvetica-Bold", 12)
        c.drawString(0.8 * inch, y, "Summary plots")
        y -= 0.25 * inch
        y = draw_img(plot_labels_png, y)
        if plot_rf_png:
            y = draw_img(plot_rf_png, y)
        if plot_reject_stage_png:
            y = draw_img(plot_reject_stage_png, y)
        if plot_reject_reason_png:
            y = draw_img(plot_reject_reason_png, y)

        c.setFont("Helvetica", 9)
        c.drawString(0.8 * inch, 0.6 * inch, f"HTML report contains full tables and structure gallery: {html_out.name}")
        c.save()
        pdf_ok = True

    except Exception as e:
        write_text(pdf_error, f"{type(e).__name__}: {e}\n")
        try:
            if pdf_out.exists():
                pdf_out.unlink()
        except Exception:
            pass
        pdf_ok = False

    return html_out, (pdf_out if pdf_ok else None), (pdf_error if not pdf_ok else None)


def main() -> int:
    ap = argparse.ArgumentParser(prog="python -m mirpv_ng.make_report", description="Generate HTML (+ optional PDF) report for miRPV-NG sRNA-seq run")
    ap.add_argument("--sample-id", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--final-candidates-tsv", required=True)
    ap.add_argument("--final-report-json", required=False, default=None)
    ap.add_argument("--candidates-struct-tsv", required=True)
    ap.add_argument("--mature-tsv", required=False, default=None)
    ap.add_argument("--rejects-merged-tsv", required=False, default=None)

    args = ap.parse_args()
    outdir = Path(args.outdir)
    html_out, pdf_out, pdf_err = build_report(
        sample_id=args.sample_id,
        outdir=outdir,
        final_candidates_tsv=Path(args.final_candidates_tsv),
        final_report_json=Path(args.final_report_json) if args.final_report_json else None,
        candidates_struct_tsv=Path(args.candidates_struct_tsv),
        mature_tsv=Path(args.mature_tsv) if args.mature_tsv else None,
        rejects_merged_tsv=Path(args.rejects_merged_tsv) if args.rejects_merged_tsv else None,
    )

    print(f"[make-report] report.html: {html_out}")
    if pdf_out:
        print(f"[make-report] report.pdf:  {pdf_out}")
    else:
        # not fatal
        print(f"[make-report] report.pdf:  FAILED (see pdf_error.txt if present)")
        if pdf_err and pdf_err.exists():
            print(f"[make-report] pdf_error.txt: {pdf_err}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
