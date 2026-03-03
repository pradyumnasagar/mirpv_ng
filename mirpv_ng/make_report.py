#!/usr/bin/env python3
"""
miRPV-NG Report Generator (HTML + PDF) - FIXED

Fixes:
1. Interactive Render: Uses correct namespace 'fornac.FornaContainer'.
2. Loop Crash: Adds try/catch blocks so one failed graph doesn't kill the whole report.
3. Rendering: Uses reliable CDN links for Fornac 1.1.8.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure matplotlib doesn't try to use Qt under WSL/headless
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt


# -----------------------------
# IO helpers
# -----------------------------

def read_tsv(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            rows.append(row)
    return rows


def safe_float(x: object, default: Optional[float] = None) -> Optional[float]:
    try:
        if x is None:
            return default
        s = str(x).strip()
        if s == "":
            return default
        return float(s)
    except Exception:
        return default


def which(cmd: str) -> Optional[str]:
    return shutil.which(cmd)


def run_cmd(cmd: List[str], cwd: Optional[Path] = None, input_text: Optional[str] = None) -> Tuple[int, str, str]:
    p = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        input=input_text,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return p.returncode, p.stdout, p.stderr


# -----------------------------
# Plot helpers (for PDF)
# -----------------------------

def _mpl_dark():
    plt.rcParams.update({
        "figure.facecolor": "#0b1120",
        "axes.facecolor": "#0f172a",
        "axes.edgecolor": "#1f2937",
        "axes.labelcolor": "#e5e7eb",
        "xtick.color": "#cbd5e1",
        "ytick.color": "#cbd5e1",
        "text.color": "#e5e7eb",
        "grid.color": "#334155",
        "grid.alpha": 0.35,
    })


def save_bar(labels: List[str], values: List[int], title: str, out_png: Path):
    _mpl_dark()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8.2, 4.2))
    plt.grid(axis="y")
    plt.bar(range(len(labels)), values)
    plt.xticks(range(len(labels)), labels, rotation=20, ha="right")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=170)
    plt.close()


def save_hist(values: List[float], title: str, out_png: Path, bins: int = 20, xlabel: str = ""):
    _mpl_dark()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8.2, 4.2))
    plt.grid(axis="y")
    plt.hist(values, bins=bins)
    plt.title(title)
    if xlabel:
        plt.xlabel(xlabel)
    plt.tight_layout()
    plt.savefig(out_png, dpi=170)
    plt.close()


def save_scatter(x: List[float], y: List[float], title: str, out_png: Path, xlabel: str = "", ylabel: str = ""):
    _mpl_dark()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8.2, 4.2))
    plt.grid(True)
    plt.scatter(x, y, s=18)
    plt.title(title)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_png, dpi=170)
    plt.close()


# -----------------------------
# ViennaRNA structure snapshot (for PDF)
# -----------------------------

def rnaplot_to_png(seq: str, dot: str, out_png: Path, density: int = 200) -> Tuple[bool, str]:
    """
    Try: RNAplot -> *_ss.eps -> convert/gs -> PNG
    """
    if not which("RNAplot"):
        return False, "RNAplot not found"
    if not seq or not dot or len(seq) != len(dot):
        return False, "sequence/structure length mismatch or empty"

    out_png.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="mirpvng_rnaplot_") as td:
        td = Path(td)
        # RNAplot accepts: >id\nSEQ\nDOT\n
        inp = f">x\n{seq}\n{dot}\n"
        rc, out, err = run_cmd(["RNAplot"], cwd=td, input_text=inp)
        if rc != 0:
            return False, f"RNAplot failed: {err.strip() or out.strip()}"
        eps = sorted(td.glob("*_ss.eps"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not eps:
            eps = sorted(td.glob("*.eps"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not eps:
            return False, "RNAplot produced no EPS"
        eps_path = eps[0]

        # Prefer ImageMagick "convert"; fallback to ghostscript.
        if which("convert"):
            rc, out, err = run_cmd(["convert", "-density", str(density), str(eps_path), str(out_png)])
            if rc == 0 and out_png.exists():
                return True, "ok"
        
        if which("gs"):
            rc, out, err = run_cmd([
                "gs", "-dSAFER", "-dBATCH", "-dNOPAUSE",
                "-sDEVICE=png16m", f"-r{density}",
                f"-sOutputFile={out_png}", str(eps_path)
            ])
            if rc == 0 and out_png.exists():
                return True, "ok"
            return False, f"gs failed: {err.strip() or out.strip()}"

        return False, "neither convert nor gs available"


# -----------------------------
# HTML report (Fixed & Robust)
# -----------------------------

def generate_html_report(
    output_path: Path,
    sample_id: str,
    classification_counts: Dict[str, int],
    rf_scores: List[float],
    candidates_list: List[Dict[str, object]],
    extra_series: Dict[str, List[float]],
):
    js_class_labels = json.dumps(list(classification_counts.keys()))
    js_class_values = json.dumps(list(classification_counts.values()))
    js_candidates = json.dumps(candidates_list)

    # RF histogram bins (0-1)
    bins = [0] * 10
    for score in rf_scores:
        if score is None:
            continue
        val = float(score)
        b = min(int(val * 10), 9)
        bins[b] += 1
    js_rf_bins = json.dumps(bins)

    # extra charts data
    js_len = json.dumps(extra_series.get("length", []))
    js_mfe = json.dumps(extra_series.get("mfe", []))
    js_depth = json.dumps(extra_series.get("depth", []))
    js_cpm = json.dumps(extra_series.get("cpm", []))
    js_rf_vs_mfe = json.dumps(extra_series.get("rf_vs_mfe", []))

    # --- HTML template ---
    # NOTE: Switched Fornac links to the robust 1.1.8 version we verified earlier.
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>miRPV-NG Report: {sample_id}</title>
    
    <link href="https://cdn.jsdelivr.net/npm/fornac@1.1.8/app/styles/fornac.min.css" rel="stylesheet">
    
    <style>
        :root {{
            --bg-color: #0b1120;
            --card-color: #0f172a;
            --text-color: #e2e8f0;
            --muted-color: #94a3b8;
            --border-color: #1e293b;
            --accent-color: #38bdf8;
            --accent-blue: #3b82f6;
            --accent-yellow: #eab308;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-color);
            margin: 0;
            padding: 0;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 2rem;
        }}
        h1, h2 {{
            color: var(--text-color);
            margin-bottom: 1.5rem;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2rem;
        }}
        .card {{
            background-color: var(--card-color);
            border-radius: 12px;
            padding: 1.5rem;
            border: 1px solid var(--border-color);
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }}
        .card h3 {{
            margin-top: 0;
            margin-bottom: 1rem;
            font-size: 1.25rem;
        }}
        canvas {{
            max-width: 100%;
        }}
        .novel-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(380px, 1fr));
            gap: 1.5rem;
        }}
        .novel-card {{
            background-color: var(--card-color);
            border-radius: 12px;
            padding: 1.5rem;
            border: 1px solid var(--border-color);
        }}
        .novel-header {{
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 1rem;
            gap: 12px;
        }}
        .novel-id {{
            font-size: 0.9rem;
            font-weight: 600;
            color: var(--accent-color);
            word-break: break-all;
            max-width: 65%;
        }}
        .novel-metrics {{
            display: flex;
            gap: 0.5rem;
            flex-wrap: wrap;
        }}
        .badge {{
            padding: 4px 8px; 
            border-radius: 4px; 
            background: #334155; 
            font-size: 0.75rem; 
            color: #fff; 
            font-weight: 600;
        }}
        .badge-rf {{ border-left: 3px solid var(--accent-blue); }}
        .badge-mfe {{ border-left: 3px solid var(--accent-yellow); }}

        .fornac-container {{
            width: 100%;
            height: 300px;
            background: #ffffff;
            border-radius: 8px;
            overflow: hidden;
            position: relative;
        }}
        .loading-text {{
            position: absolute;
            top: 50%; left: 50%;
            transform: translate(-50%, -50%);
            color: #333;
        }}
        .section-spacer {{ margin-top: 2.5rem; }}
        .muted {{ color: var(--muted-color); }}
        .topbar {{ display:flex; justify-content:space-between; align-items:center; margin-bottom: 1.25rem; gap: 1rem; }}
        .small {{ font-size: 0.9rem; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="topbar">
            <div>
                <h1>miRPV-NG Report</h1>
                <div class="muted small">Sample: <b>{sample_id}</b> • Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}</div>
            </div>
            <div class="muted small">Interactive HTML • PDF also available</div>
        </div>

        <div class="grid">
            <div class="card">
                <h3>Classification</h3>
                <canvas id="classificationChart"></canvas>
            </div>
            <div class="card">
                <h3>RF Score Distribution</h3>
                <canvas id="rfScoreChart"></canvas>
            </div>
        </div>

        <div class="grid">
            <div class="card">
                <h3>Precursor Length Distribution</h3>
                <canvas id="lenChart"></canvas>
            </div>
            <div class="card">
                <h3>MFE Distribution</h3>
                <canvas id="mfeChart"></canvas>
            </div>
        </div>

        <div class="grid">
            <div class="card">
                <h3>RF vs MFE</h3>
                <canvas id="rfMfeChart"></canvas>
            </div>
            <div class="card">
                <h3>Depth/CPM</h3>
                <canvas id="depthChart"></canvas>
            </div>
        </div>

        <div class="section-spacer">
            <h2>Top Novel Candidates</h2>
            <div id="novelCandidates" class="novel-grid"></div>
        </div>
    </div>

<script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/3.5.17/d3.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script src="https://cdn.jsdelivr.net/npm/fornac@1.1.8/dist/scripts/fornac.min.js"></script>

<script>
const classLabels = {js_class_labels};
const classValues = {js_class_values};
const rfBins = {js_rf_bins};

const lengthVals = {js_len};
const mfeVals = {js_mfe};
const depthVals = {js_depth};
const cpmVals = {js_cpm};
const rfMfePts = {js_rf_vs_mfe};

const candidates = {js_candidates};

// --- Charts Setup ---
new Chart(document.getElementById('classificationChart'), {{
    type: 'bar',
    data: {{ labels: classLabels, datasets: [{{ data: classValues, backgroundColor: ['#22c55e', '#eab308', '#3b82f6'] }}] }},
    options: {{ plugins: {{ legend: {{ display: false }} }}, scales: {{ y: {{ beginAtZero: true, grid: {{ color: '#1e293b' }} }} }} }}
}});

new Chart(document.getElementById('rfScoreChart'), {{
    type: 'bar',
    data: {{ labels: ['0-0.1','0.1-0.2','0.2-0.3','0.3-0.4','0.4-0.5','0.5-0.6','0.6-0.7','0.7-0.8','0.8-0.9','0.9-1.0'], datasets: [{{ data: rfBins, backgroundColor: '#818cf8' }}] }},
    options: {{ plugins: {{ legend: {{ display: false }} }}, scales: {{ y: {{ beginAtZero: true, grid: {{ color: '#1e293b' }} }} }} }}
}});

// Helper for client-side histograms
function hist(values, nbins, minv, maxv){{
    const bins = new Array(nbins).fill(0);
    const step = (maxv-minv)/nbins;
    for (const v of values){{
        if (v===null || v===undefined) continue;
        const x = Number(v);
        if (!isFinite(x) || x < minv || x > maxv) continue;
        let b = Math.floor((x-minv)/step);
        if (b>=nbins) b = nbins-1;
        bins[b] += 1;
    }}
    const labels = [];
    for (let i=0;i<nbins;i++) {{
        const a = (minv + i*step).toFixed(1);
        const b = (minv + (i+1)*step).toFixed(1);
        labels.push(`${{a}}-${{b}}`);
    }}
    return {{labels, bins}};
}}

(function(){{
  // Length Histogram
  if (lengthVals && lengthVals.length) {{
    const minL = Math.min(...lengthVals), maxL = Math.max(...lengthVals);
    const h = hist(lengthVals, 18, minL, maxL);
    new Chart(document.getElementById('lenChart'), {{
      type:'bar',
      data:{{ labels:h.labels, datasets:[{{ data:h.bins, backgroundColor:'#60a5fa' }}] }},
      options:{{ plugins:{{legend:{{display:false}}}}, scales:{{y:{{beginAtZero:true, grid:{{color:'#1e293b'}}}} }} }}
    }});
  }} else {{
    document.getElementById('lenChart').parentElement.innerHTML += "<div class='muted small'>No data found.</div>";
  }}

  // MFE Histogram
  if (mfeVals && mfeVals.length) {{
    const minM = Math.min(...mfeVals), maxM = Math.max(...mfeVals);
    const h = hist(mfeVals, 18, minM, maxM);
    new Chart(document.getElementById('mfeChart'), {{
      type:'bar',
      data:{{ labels:h.labels, datasets:[{{ data:h.bins, backgroundColor:'#f59e0b' }}] }},
      options:{{ plugins:{{legend:{{display:false}}}}, scales:{{y:{{beginAtZero:true, grid:{{color:'#1e293b'}}}} }} }}
    }});
  }} else {{
    document.getElementById('mfeChart').parentElement.innerHTML += "<div class='muted small'>No data found.</div>";
  }}

  // RF vs MFE Scatter
  if (rfMfePts && rfMfePts.length) {{
    new Chart(document.getElementById('rfMfeChart'), {{
      type:'scatter',
      data:{{ datasets:[{{ data:rfMfePts, backgroundColor:'#a78bfa' }}] }},
      options:{{
        plugins:{{legend:{{display:false}}}},
        scales:{{ x:{{title:{{display:true,text:'RF'}}, min:0, max:1, grid:{{color:'#1e293b'}}}}, y:{{title:{{display:true,text:'MFE'}}, grid:{{color:'#1e293b'}}}} }}
      }}
    }});
  }} else {{
    document.getElementById('rfMfeChart').parentElement.innerHTML += "<div class='muted small'>No data found.</div>";
  }}

  // Depth/CPM
  const hasDepth = depthVals && depthVals.length;
  const hasCpm = cpmVals && cpmVals.length;
  if (hasDepth || hasCpm) {{
    if (hasDepth) {{
      const minD = Math.min(...depthVals), maxD = Math.max(...depthVals);
      const h = hist(depthVals, 16, minD, maxD);
      new Chart(document.getElementById('depthChart'), {{
        type:'bar',
        data:{{ labels:h.labels, datasets:[{{ data:h.bins, backgroundColor:'#22c55e' }}] }},
        options:{{ plugins:{{legend:{{display:false}}}}, scales:{{y:{{beginAtZero:true,grid:{{color:'#1e293b'}}}} }} }}
      }});
    }} else if (hasCpm) {{
      const minC = Math.min(...cpmVals), maxC = Math.max(...cpmVals);
      const h2 = hist(cpmVals, 16, minC, maxC);
      new Chart(document.getElementById('depthChart'), {{
        type:'bar',
        data:{{ labels:h2.labels, datasets:[{{ data:h2.bins, backgroundColor:'#14b8a6' }}] }},
        options:{{ plugins:{{legend:{{display:false}}}}, scales:{{y:{{beginAtZero:true,grid:{{color:'#1e293b'}}}} }} }}
      }});
    }}
  }} else {{
    document.getElementById('depthChart').parentElement.innerHTML += "<div class='muted small'>No depth/cpm data.</div>";
  }}
}})();

// --- Novel Candidates with ROBUST Fornac Loading ---
document.addEventListener("DOMContentLoaded", function() {{
    const novelContainer = document.getElementById('novelCandidates');
    const novelCands = candidates.filter(c => c.classification === 'Novel-High');

    novelCands.forEach((c, index) => {{
        const divId = `fornac-${{index}}`;
        const card = document.createElement('div');
        card.className = 'novel-card';
        card.innerHTML = `
            <div class="novel-header">
                <div class="novel-id">${{c.id}}</div>
                <div class="novel-metrics">
                    <span class="badge badge-rf">RF: ${{c.rf_score}}</span>
                    <span class="badge badge-mfe">MFE: ${{c.mfe}}</span>
                </div>
            </div>
            <div id="${{divId}}" class="fornac-container">
                <div class="loading-text">Initializing...</div>
            </div>
        `;
        novelContainer.appendChild(card);

        // DELAYED LOAD to prevent crash and ensure DOM readiness
        setTimeout(() => {{
            try {{
                // FIX: Check for global variable and correct namespace
                if (typeof fornac === 'undefined' || typeof fornac.FornaContainer === 'undefined') {{
                    throw new Error("Library not loaded");
                }}

                // Clear loading text
                document.getElementById(divId).innerHTML = '';

                // FIX: Use 'fornac.FornaContainer' (correct namespace)
                const fc = new fornac.FornaContainer("#" + divId, {{
                    'zoomable': true,
                    'editable': false,
                    'animation': true,
                    'labelInterval': 0
                }});
                fc.addRNA(c.dotbracket, {{ 'sequence': c.sequence }});
                
            }} catch (e) {{
                console.error("Forna Error for " + c.id, e);
                document.getElementById(divId).innerHTML = 
                    `<div style="padding:20px; color:#ef4444; font-size:0.9rem;">
                        Viz Error: ${{e.message}}
                     </div>`;
            }}
        }}, 100 + (index * 50)); // Stagger loads to prevent freezing
    }});
}});
</script>
</body>
</html>
"""
    output_path.write_text(html_content, encoding="utf-8")


# -----------------------------
# PDF report
# -----------------------------

def generate_pdf_report(
    pdf_path: Path,
    sample_id: str,
    outdir: Path,
    classification_counts: Dict[str, int],
    rf_scores: List[float],
    lengths: List[float],
    mfes: List[float],
    rf_mfe_pairs: List[Tuple[float, float]],
    top_novel: List[Dict[str, object]],
):
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
        from reportlab.lib.units import inch
        from reportlab.lib.utils import ImageReader
    except Exception as e:
        raise RuntimeError(f"reportlab is required for PDF, but import failed: {e}")

    assets = outdir / "assets"
    assets.mkdir(parents=True, exist_ok=True)

    # Make plot images
    class_png = assets / "pdf_classification.png"
    rf_png = assets / "pdf_rf_hist.png"
    len_png = assets / "pdf_len_hist.png"
    mfe_png = assets / "pdf_mfe_hist.png"
    scatter_png = assets / "pdf_rf_mfe.png"

    labels = list(classification_counts.keys())
    values = [classification_counts[k] for k in labels]
    save_bar(labels, values, f"{sample_id}: Classification", class_png)

    if rf_scores:
        save_hist(rf_scores, f"{sample_id}: RF score distribution", rf_png, bins=20, xlabel="RF")
    if lengths:
        save_hist(lengths, f"{sample_id}: precursor length distribution", len_png, bins=20, xlabel="length")
    if mfes:
        save_hist(mfes, f"{sample_id}: MFE distribution", mfe_png, bins=20, xlabel="MFE")
    if rf_mfe_pairs:
        xs = [p[0] for p in rf_mfe_pairs]
        ys = [p[1] for p in rf_mfe_pairs]
        save_scatter(xs, ys, f"{sample_id}: RF vs MFE", scatter_png, xlabel="RF", ylabel="MFE")

    # Build PDF
    c = canvas.Canvas(str(pdf_path), pagesize=letter)
    W, H = letter
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    def draw_title():
        y = H - 0.8 * inch
        c.setFont("Helvetica-Bold", 18)
        c.drawString(0.75 * inch, y, f"miRPV-NG Report — {sample_id}")
        y -= 0.3 * inch
        c.setFont("Helvetica", 10)
        c.drawString(0.75 * inch, y, f"Generated: {now}")
        y -= 0.25 * inch
        c.setFont("Helvetica-Bold", 11)
        c.drawString(0.75 * inch, y, "Headline counts")
        y -= 0.18 * inch
        c.setFont("Helvetica", 10)
        s = "   ".join([f"{k}: {classification_counts.get(k,0)}" for k in ["Known-Confirmed","Known-Atypical","Novel-High"] if k in classification_counts])
        if not s:
            s = "   ".join([f"{k}: {v}" for k,v in classification_counts.items()])
        c.drawString(0.85 * inch, y, s[:140])
        return y - 0.25 * inch

    def draw_img(img: Path, y: float) -> float:
        if not img.exists():
            return y
        ir = ImageReader(str(img))
        iw, ih = ir.getSize()
        max_w = W - 1.5 * inch
        scale = max_w / float(iw)
        new_w = max_w
        new_h = float(ih) * scale
        if y - new_h < 0.75 * inch:
            c.showPage()
            y = H - 0.8 * inch
        c.drawImage(ir, 0.75 * inch, y - new_h, width=new_w, height=new_h)
        return y - new_h - 0.22 * inch

    y = draw_title()
    c.setFont("Helvetica-Bold", 12)
    c.drawString(0.75 * inch, y, "Summary plots")
    y -= 0.25 * inch

    y = draw_img(class_png, y)
    y = draw_img(rf_png, y)
    y = draw_img(len_png, y)
    y = draw_img(mfe_png, y)
    y = draw_img(scatter_png, y)

    # Novel candidates section
    c.showPage()
    y = H - 0.8 * inch
    c.setFont("Helvetica-Bold", 16)
    c.drawString(0.75 * inch, y, f"Top Novel-High candidates — {sample_id}")
    y -= 0.3 * inch
    c.setFont("Helvetica", 9)
    c.drawString(0.75 * inch, y, "Static structures are generated with ViennaRNA RNAplot if available. Otherwise, only IDs/metrics are listed.")
    y -= 0.3 * inch

    for i, cand in enumerate(top_novel, 1):
        cid = str(cand.get("id", "NA"))
        rf = cand.get("rf_score", "NA")
        mfe = cand.get("mfe", "NA")
        seq = str(cand.get("sequence", "") or "")
        dot = str(cand.get("dotbracket", "") or "")

        c.setFont("Helvetica-Bold", 10)
        c.drawString(0.75 * inch, y, f"{i:02d}. {cid}"[:120])
        y -= 0.16 * inch
        c.setFont("Helvetica", 9)
        c.drawString(0.85 * inch, y, f"RF: {rf}    MFE: {mfe}"[:120])
        y -= 0.18 * inch

        # try make structure png
        png = assets / f"novel_{i:02d}.png"
        ok, msg = rnaplot_to_png(seq, dot, png)
        if ok and png.exists():
            y = draw_img(png, y)
        else:
            c.setFont("Helvetica-Oblique", 8)
            c.drawString(0.85 * inch, y, f"(structure image unavailable: {msg})"[:140])
            y -= 0.22 * inch

        # page break if needed
        if y < 1.2 * inch:
            c.showPage()
            y = H - 0.8 * inch

    c.setFont("Helvetica", 8)
    c.drawString(0.75 * inch, 0.6 * inch, "Open report.html for interactive structures and browsing.")
    c.save()


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate miRPV-NG HTML and PDF reports")
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--final-candidates-tsv", required=True)
    parser.add_argument("--candidates-struct-tsv", required=False, default=None,
                        help="candidates_struct.tsv (Optional if final_candidates has structure)")
    parser.add_argument("--rejects-merged-tsv", required=False, default=None)
    parser.add_argument("--top-novel", type=int, default=10)
    parser.add_argument("--no-pdf", action="store_true", help="Skip PDF generation")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    final_rows = read_tsv(Path(args.final_candidates_tsv))
    if not final_rows:
        raise SystemExit("[make-report] final_candidates.tsv is empty")

    struct_rows = []
    if args.candidates_struct_tsv:
        p = Path(args.candidates_struct_tsv)
        if p.exists():
            struct_rows = read_tsv(p)

    # Index structures by candidate_id
    struct_by_id: Dict[str, Dict[str, str]] = {}
    for r in struct_rows:
        cid = (r.get("candidate_id") or "").strip()
        if cid:
            struct_by_id[cid] = r

    # Build candidate objects for HTML (and novel list for PDF)
    classification_counts = Counter()
    rf_scores: List[float] = []
    lengths: List[float] = []
    mfes: List[float] = []
    rf_mfe_pairs: List[Tuple[float, float]] = []
    depth_vals: List[float] = []
    cpm_vals: List[float] = []

    candidates: List[Dict[str, object]] = []

    # Determine likely column names
    col_class = "final_label" if "final_label" in final_rows[0] else ("classification" if "classification" in final_rows[0] else None)
    col_rf = "best_rf_score" if "best_rf_score" in final_rows[0] else ("rf_score" if "rf_score" in final_rows[0] else None)
    col_mfe = "mfe" if "mfe" in final_rows[0] else None
    col_depth = "depth_raw" if "depth_raw" in final_rows[0] else None
    col_cpm = "cpm" if "cpm" in final_rows[0] else None

    for r in final_rows:
        cid = (r.get("candidate_id") or r.get("id") or "").strip()
        cls = (r.get(col_class) if col_class else r.get("final_label")) or "Unknown"
        cls = str(cls).strip()
        classification_counts[cls] += 1

        rf = safe_float(r.get(col_rf), None) if col_rf else None
        if rf is not None:
            rf_scores.append(rf)

        # Pull structure from final_candidates first; else from candidates_struct.tsv
        seq = (r.get("seq") or r.get("sequence") or "")
        dot = (r.get("dotbracket") or r.get("structure") or "")
        mfe = safe_float(r.get("mfe"), None)

        if (not seq or not dot) and cid in struct_by_id:
            sr = struct_by_id[cid]
            seq = seq or (sr.get("seq") or sr.get("sequence") or "")
            dot = dot or (sr.get("dotbracket") or sr.get("structure") or "")
            if mfe is None:
                mfe = safe_float(sr.get("mfe"), None)

        if seq:
            lengths.append(float(len(seq)))

        if mfe is not None:
            mfes.append(mfe)
            if rf is not None:
                rf_mfe_pairs.append((rf, mfe))

        if col_depth:
            dv = safe_float(r.get(col_depth), None)
            if dv is not None:
                depth_vals.append(dv)
        if col_cpm:
            cv = safe_float(r.get(col_cpm), None)
            if cv is not None:
                cpm_vals.append(cv)

        candidates.append({
            "id": cid,
            "classification": cls,
            "rf_score": f"{rf:.3f}" if rf is not None else "NA",
            "mfe": f"{mfe:.1f}" if mfe is not None else "NA",
            "sequence": seq,
            "dotbracket": dot
        })

    # Sort and pick top novel
    novel = [c for c in candidates if c["classification"] == "Novel-High"]
    novel_sorted = sorted(novel, key=lambda x: float(x["rf_score"]) if x["rf_score"] != "NA" else -1.0, reverse=True)
    top_novel = novel_sorted[: max(args.top_novel, 0)]

    # Generate HTML
    html_path = outdir / "report.html"
    extra_series = {
        "length": [float(x) for x in lengths],
        "mfe": [float(x) for x in mfes],
        "depth": [float(x) for x in depth_vals],
        "cpm": [float(x) for x in cpm_vals],
        "rf_vs_mfe": [{"x": p[0], "y": p[1]} for p in rf_mfe_pairs],
    }
    generate_html_report(html_path, args.sample_id, dict(classification_counts), rf_scores, candidates, extra_series)

    print(f"[make-report] report.html: {html_path}")

    # Generate PDF
    if not args.no_pdf:
        pdf_path = outdir / "report.pdf"
        generate_pdf_report(
            pdf_path=pdf_path,
            sample_id=args.sample_id,
            outdir=outdir,
            classification_counts=dict(classification_counts),
            rf_scores=rf_scores,
            lengths=lengths,
            mfes=mfes,
            rf_mfe_pairs=rf_mfe_pairs,
            top_novel=top_novel,
        )
        print(f"[make-report] report.pdf:  {pdf_path}")


if __name__ == "__main__":
    main()