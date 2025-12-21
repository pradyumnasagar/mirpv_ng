import argparse
import pandas as pd
import json
import os
import sys

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate HTML report for miRPV-NG")
    parser.add_argument("--sample-id", required=True, help="Sample ID for the run")
    parser.add_argument("--outdir", required=True, help="Output directory for the report")
    parser.add_argument("--final-candidates-tsv", required=True, help="Path to final_candidates.tsv")
    parser.add_argument("--candidates-struct-tsv", required=False, help="Path to candidates_struct.tsv (Optional if final_candidates has structure)")
    return parser.parse_args()

def generate_html_report(output_path, sample_id, classification_counts, rf_score_dist, candidates_list):
    """
    Generates the HTML report with embedded Fornac visualization.
    """
    
    # 1. Prepare Data for JavaScript
    js_class_labels = json.dumps(list(classification_counts.keys()))
    js_class_values = json.dumps(list(classification_counts.values()))
    js_candidates = json.dumps(candidates_list)
    
    # Bin RF Scores for Histogram
    bins = [0] * 10
    for score in rf_score_dist:
        try:
            val = float(score)
            if 0 <= val <= 1.0:
                idx = min(int(val * 10), 9)
                bins[idx] += 1
        except (ValueError, TypeError):
            continue
    js_rf_bins = json.dumps(bins)

    # 2. HTML Template
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>miRPV-NG Report: {sample_id}</title>
        
        <link href="https://cdn.jsdelivr.net/npm/fornac@1.1.8/app/styles/fornac.min.css" rel="stylesheet">
        
        <style>
            :root {{ --bg-dark: #0f172a; --card-bg: #1e293b; --text-main: #e2e8f0; --accent-blue: #3b82f6; --accent-yellow: #eab308; }}
            body {{ font-family: 'Segoe UI', sans-serif; background-color: var(--bg-dark); color: var(--text-main); margin: 0; padding: 20px; }}
            .container {{ max-width: 1400px; margin: 0 auto; }}
            h1 {{ border-bottom: 1px solid #334155; padding-bottom: 15px; margin-bottom: 30px; }}
            
            /* Grids */
            .dashboard-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; margin-bottom: 40px; }}
            .candidates-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(350px, 1fr)); gap: 20px; }}
            
            /* Cards */
            .card {{ background-color: var(--card-bg); border-radius: 12px; padding: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.2); }}
            .card-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px; border-bottom: 1px solid #334155; padding-bottom: 10px; }}
            .seq-id {{ font-family: monospace; color: var(--accent-blue); font-size: 0.9rem; max-width: 60%; word-break: break-all; }}
            
            /* Badges */
            .badge-container {{ display: flex; gap: 8px; }}
            .badge {{ padding: 4px 8px; border-radius: 4px; background: #334155; font-size: 0.75rem; color: #fff; font-weight: 600; }}
            .badge-rf {{ border-left: 3px solid var(--accent-blue); }}
            .badge-mfe {{ border-left: 3px solid var(--accent-yellow); }}

            /* Visualization */
            .rna-viz-container {{ width: 100%; height: 350px; background: #ffffff; border-radius: 8px; position: relative; overflow: hidden; }}
            .loading-text {{ color: #333; position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); }}
            
            /* Print Optimization (for Save to PDF) */
            @media print {{
                body {{ background-color: white !important; color: black !important; -webkit-print-color-adjust: exact; }}
                .container {{ max-width: 100%; }}
                .card {{ background-color: #f8f9fa !important; border: 1px solid #ccc; box-shadow: none; break-inside: avoid; page-break-inside: avoid; }}
                .badge {{ color: black !important; border: 1px solid #000; }}
                h1, h2, h3 {{ color: black !important; }}
                canvas {{ max-height: 300px; }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>miRPV-NG Report: {sample_id}</h1>

            <div class="dashboard-grid">
                <div class="card"><h3>Classification</h3><div style="height:250px;"><canvas id="classChart"></canvas></div></div>
                <div class="card"><h3>RF Score Distribution</h3><div style="height:250px;"><canvas id="rfChart"></canvas></div></div>
            </div>

            <h2>Top Novel Candidates</h2>
            <div id="candidates-container" class="candidates-grid"></div>
        </div>

        <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/d3/3.5.17/d3.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/fornac@1.1.8/dist/scripts/fornac.min.js"></script>

        <script>
            const classLabels = {js_class_labels};
            const classValues = {js_class_values};
            const rfBins = {js_rf_bins};
            const candidates = {js_candidates};

            document.addEventListener("DOMContentLoaded", function() {{
                // 1. Charts
                new Chart(document.getElementById('classChart'), {{
                    type: 'bar',
                    data: {{ labels: classLabels, datasets: [{{ label: 'Count', data: classValues, backgroundColor: ['#22c55e', '#eab308', '#3b82f6'] }}] }},
                    options: {{ maintainAspectRatio: false, plugins: {{ legend: {{ display: false }} }}, scales: {{ y: {{ grid: {{ color: '#334155' }} }} }} }}
                }});
                
                new Chart(document.getElementById('rfChart'), {{
                    type: 'bar',
                    data: {{ labels: ['0-0.1', '0.1-0.2', '0.2-0.3', '0.3-0.4', '0.4-0.5', '0.5-0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9', '0.9-1.0'], datasets: [{{ label: 'Freq', data: rfBins, backgroundColor: '#818cf8' }}] }},
                    options: {{ maintainAspectRatio: false, plugins: {{ legend: {{ display: false }} }}, scales: {{ y: {{ grid: {{ color: '#334155' }} }} }} }}
                }});

                // 2. RNA Visualizations
                const grid = document.getElementById('candidates-container');

                candidates.forEach((cand, index) => {{
                    const divId = "rna-viz-" + index;
                    const card = document.createElement('div');
                    card.className = 'card';
                    
                    card.innerHTML = `
                        <div class="card-header">
                            <div class="seq-id" title="${{cand.id}}">${{cand.id}}</div>
                            <div class="badge-container">
                                <span class="badge badge-rf">RF: ${{cand.rf_score}}</span>
                                <span class="badge badge-mfe">MFE: ${{cand.mfe}}</span>
                            </div>
                        </div>
                        <div id="${{divId}}" class="rna-viz-container">
                            <div class="loading-text">Initializing...</div>
                        </div>
                    `;
                    grid.appendChild(card);

                    setTimeout(() => {{
                        try {{
                            // Clear loading text
                            document.getElementById(divId).innerHTML = '';

                            var container = new fornac.FornaContainer("#" + divId, {{
                                'zoomable': true, 'editable': false, 'animation': true, 'labelInterval': 0
                            }});
                            container.addRNA(cand.structure, {{ 'sequence': cand.sequence, 'structure': cand.structure }});
                            
                        }} catch (e) {{
                            console.error("Viz Error:", e);
                            document.getElementById(divId).innerHTML = `<div style="padding:20px; color:red;">Error: ${{e.message}}</div>`;
                        }}
                    }}, 100 + (index * 100));
                }});
            }});
        </script>
    </body>
    </html>
    """
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

def main():
    args = parse_arguments()
    
    # 1. Load Data
    print(f"Loading final candidates from: {args.final_candidates_tsv}")
    try:
        df_final = pd.read_csv(args.final_candidates_tsv, sep='\t')
    except Exception as e:
        print(f"Error loading final_candidates: {e}")
        sys.exit(1)

    # 2. Determine Column Names
    # Looks for 'final_label' (your file) or 'classification' (fallback)
    class_col = next((col for col in ['final_label', 'classification', 'class'] if col in df_final.columns), None)
    # Looks for 'best_rf_score' (your file) or 'rf_score' (fallback)
    rf_col = next((col for col in ['best_rf_score', 'rf_proba', 'rf_score'] if col in df_final.columns), None)
    # Looks for 'dotbracket' (your file) or 'structure' (fallback)
    struct_col = next((col for col in ['dotbracket', 'structure'] if col in df_final.columns), None)
    # Looks for 'seq' (your file) or 'sequence' (fallback)
    seq_col = next((col for col in ['seq', 'sequence'] if col in df_final.columns), None)
    
    # ID column
    id_col = 'candidate_id' if 'candidate_id' in df_final.columns else 'id'

    # 3. Process Statistics
    classification_counts = {}
    if class_col:
        classification_counts = df_final[class_col].value_counts().to_dict()
    else:
        print("Warning: Could not find classification column (e.g., 'final_label').")

    rf_scores = []
    if rf_col:
        rf_scores = df_final[rf_col].dropna().tolist()

    # 4. Merge Logic (Robust)
    merged_df = df_final
    
    # If the struct file is provided, we merge ONLY if we are missing data
    if args.candidates_struct_tsv:
        try:
            df_struct = pd.read_csv(args.candidates_struct_tsv, sep='\t')
            
            # Check if we need to merge
            if not struct_col or not seq_col:
                print("Structure/Sequence missing in final_candidates, attempting to merge from candidates_struct...")
                # Only take columns that are NOT in df_final to avoid collisions
                cols_to_use = [c for c in df_struct.columns if c not in df_final.columns]
                # Ensure we have the ID for merging
                if id_col not in cols_to_use:
                    cols_to_use.append(id_col)
                
                merged_df = pd.merge(df_final, df_struct[cols_to_use], on=id_col, how='left')
                
                # Update column pointers if we pulled them in
                if not struct_col: struct_col = next((col for col in ['dotbracket', 'structure'] if col in merged_df.columns), None)
                if not seq_col: seq_col = next((col for col in ['seq', 'sequence'] if col in merged_df.columns), None)
            else:
                print("Note: Structure and Sequence already present in final_candidates. Skipping merge to avoid duplication.")
        except Exception as e:
            print(f"Warning: Could not read struct file ({e}). Proceeding with available data.")

    # 5. Prepare Top Candidates
    if rf_col:
        merged_df = merged_df.sort_values(by=rf_col, ascending=False)
    
    top_candidates = merged_df.head(20).to_dict('records')
    clean_candidates = []

    for cand in top_candidates:
        clean_candidates.append({
            'id': cand.get(id_col, 'Unknown'),
            'sequence': cand.get(seq_col, ''),
            'structure': cand.get(struct_col, ''),
            'rf_score': round(cand.get(rf_col, 0.0), 3),
            'mfe': round(cand.get('mfe', 0.0), 1)
        })

    # 6. Generate Report
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
        
    output_file = os.path.join(args.outdir, "report.html")
    
    generate_html_report(
        output_file, 
        args.sample_id, 
        classification_counts, 
        rf_scores, 
        clean_candidates
    )
    
    print(f"Report generated successfully: {output_file}")
    print("To save as PDF: Open the HTML file in Chrome/Edge, press Ctrl+P, and select 'Save as PDF'.")

if __name__ == "__main__":
    main()