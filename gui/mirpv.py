#!/usr/bin/env python3
"""
miRPV-NG GUI (single-file) — PySide6 wrapper that runs miRPV-NG stages via subprocess.

Fixes vs earlier version:
1) Bowtie / blocklist "index prefix" validation:
   - user selects INDEX FOLDER (dir) + BASENAME (string)
   - GUI validates folder/basename*.ebwt exists
   - CLI receives prefix: folder/basename

2) rejects.merged.tsv not selectable:
   - report step assumes: OUTROOT/09_final_report/rejects.merged.tsv

Run:
  python mirpv_ng_gui.py

Requires:
  pip/conda install pyside6
"""

from __future__ import annotations

import json
import sys
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any

from PySide6.QtCore import Qt, QProcess, QUrl, QObject, Signal
from PySide6.QtGui import QAction, QDesktopServices, QFont, QPixmap
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QLineEdit, QPushButton, QFileDialog, QComboBox, QGroupBox, QCheckBox,
    QSpinBox, QDoubleSpinBox, QTextEdit, QMessageBox, QTabWidget, QFormLayout, QProgressBar,
    QSplitter, QSizePolicy, QScrollArea, QFrame
)

APP_TITLE = "miRPV-NG"
DEFAULT_ADAPTER = "TGGAATTCTCGGGTGCCAAGG"
LOGO_FILENAME = "miRPV.png"

# =============================================================================
# STAGE PARAMETERS CONFIGURATION
# Centralized definition of all CLI flags per stage.
# Each parameter is: {"flag": str, "type": str, "default": value, "help": str}
# Types: "int", "float", "str", "bool"
# =============================================================================

STAGE_PARAMS: Dict[str, List[Dict[str, Any]]] = {
    "score-fasta": [
        {"flag": "--max-hairpin-len", "type": "int", "default": 120, "help": "Max hairpin length for direct scoring"},
        {"flag": "--max-seq-only-len", "type": "int", "default": 5000, "help": "Max sequence length for window scanning"},
        {"flag": "--window-len", "type": "int", "default": 100, "help": "Scanning window length"},
        {"flag": "--step", "type": "int", "default": 20, "help": "Scanning window step"},
        {"flag": "--tier1-min-pairs", "type": "int", "default": 18, "help": "Tier1 min base pairs"},
        {"flag": "--tier1-min-mfe", "type": "float", "default": -15.0, "help": "Tier1 min MFE threshold"},
    ],
    "fastq-to-peaks": [
        {"flag": "--max-multimaps", "type": "int", "default": 50, "help": "Max alignments per read"},
        {"flag": "--island-gap", "type": "int", "default": 50, "help": "Gap to merge nearby coverage islands"},
        {"flag": "--min-depth", "type": "int", "default": 5, "help": "Min read depth for island"},
        {"flag": "--min-cpm", "type": "float", "default": 0.5, "help": "Min CPM for island"},
        {"flag": "--smooth-w", "type": "int", "default": 3, "help": "Smoothing window size"},
        {"flag": "--peak-distance", "type": "int", "default": 35, "help": "Min distance between peaks"},
        {"flag": "--peak-micromerge", "type": "int", "default": 8, "help": "Micromerge distance for peaks"},
        {"flag": "--scipy-prominence", "type": "float", "default": None, "help": "SciPy prominence (auto if None)"},
        {"flag": "--scipy-width-min", "type": "float", "default": 2, "help": "SciPy min peak width"},
        {"flag": "--scipy-width-max", "type": "float", "default": None, "help": "SciPy max peak width (None=auto)"},
        {"flag": "--fallback-prom-frac", "type": "float", "default": 0.30, "help": "Fallback prominence fraction"},
        {"flag": "--support-window", "type": "int", "default": 15, "help": "Support window size"},
        {"flag": "--hard-frac-20-24", "type": "float", "default": 0.30, "help": "Hard fraction 20-24nt reads"},
        {"flag": "--disable-hard-length-prefilter", "type": "bool", "default": False, "help": "Disable hard length prefilter"},
        {"flag": "--anchor-unique-dominance", "type": "float", "default": 0.50, "help": "Anchor unique dominance"},
        {"flag": "--anchor-unique-prec5p", "type": "float", "default": 0.70, "help": "Anchor unique 5' precision"},
        {"flag": "--anchor-multi-dominance", "type": "float", "default": 0.60, "help": "Anchor multi dominance"},
        {"flag": "--anchor-multi-prec5p", "type": "float", "default": 0.85, "help": "Anchor multi 5' precision"},
        {"flag": "--repeat-multi-prec5p", "type": "float", "default": 0.90, "help": "Repeat multi 5' precision"},
        {"flag": "--repeat-multi-dominance", "type": "float", "default": 0.70, "help": "Repeat multi dominance"},
        {"flag": "--blocklist-mismatches", "type": "int", "default": 0, "help": "Blocklist alignment mismatches"},
        {"flag": "--blocklist-max-align", "type": "int", "default": 1, "help": "Max blocklist alignments"},
    ],
    "candidates-to-scored": [
        {"flag": "--max-hairpin-len", "type": "int", "default": 120, "help": "Max hairpin length"},
        {"flag": "--max-seq-only-len", "type": "int", "default": 5000, "help": "Max sequence-only scan length"},
        {"flag": "--window-len", "type": "int", "default": 100, "help": "Scanning window length"},
        {"flag": "--step", "type": "int", "default": 10, "help": "Scanning window step"},
        {"flag": "--tier1-min-pairs", "type": "int", "default": 18, "help": "Tier1 min base pairs"},
        {"flag": "--tier1-min-mfe", "type": "float", "default": -15.0, "help": "Tier1 min MFE"},
    ],
    "peaks-to-known": [
        {"flag": "--min-mature-overlap-bp", "type": "int", "default": 10, "help": "Min mature overlap (bp)"},
        {"flag": "--mature-query-pad", "type": "int", "default": 2, "help": "Mature query padding"},
        {"flag": "--center-tol", "type": "int", "default": 2, "help": "Peak center tolerance"},
    ],
    "peaks-to-finalists": [
        {"flag": "--known-atyp-min-rf", "type": "float", "default": 0.55, "help": "Known-atypical min RF score"},
        {"flag": "--novel-high-min-rf", "type": "float", "default": 0.65, "help": "Novel-high min RF score"},
        {"flag": "--known-atyp-min-rf-repeat", "type": "float", "default": 0.65, "help": "Known-atypical min RF (repeat)"},
        {"flag": "--novel-high-min-rf-repeat", "type": "float", "default": 0.75, "help": "Novel-high min RF (repeat)"},
    ],
    "predict-mature": [
        {"flag": "--loop-buffer", "type": "int", "default": 0, "help": "Loop buffer distance"},
        {"flag": "--fallback-loop-buffer", "type": "int", "default": 10, "help": "Fallback loop buffer"},
        {"flag": "--lengths", "type": "str", "default": "21,22,23,24", "help": "Mature lengths (comma-sep)"},
        {"flag": "--max-per-arm", "type": "int", "default": 30, "help": "Max candidates per arm"},
        {"flag": "--min-paired-context", "type": "int", "default": 6, "help": "Min paired context"},
        {"flag": "--fallback-max-per-arm", "type": "int", "default": 120, "help": "Fallback max per arm"},
        {"flag": "--fallback-min-paired-context", "type": "int", "default": 0, "help": "Fallback min paired context"},
    ],
}

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def path_text(s: str) -> str:
    return s.strip()


def must_exist_file(p: str, label: str) -> Path:
    pp = Path(p.strip())
    if not pp.exists():
        raise FileNotFoundError(f"{label} not found: {pp}")
    if pp.is_dir():
        raise FileNotFoundError(f"{label} must be a file, got directory: {pp}")
    return pp


def must_exist_dir(p: str, label: str) -> Path:
    pp = Path(p.strip())
    if not pp.exists():
        raise FileNotFoundError(f"{label} not found: {pp}")
    if not pp.is_dir():
        raise FileNotFoundError(f"{label} must be a directory, got file: {pp}")
    return pp


def validate_bowtie1_index_prefix(index_dir: Path, basename: str, label: str) -> str:
    base = basename.strip()
    if not base:
        raise ValueError(f"{label} basename is required (e.g., hg38).")
    # Bowtie1 builds .ebwt files; check any matching exists
    hits = list(index_dir.glob(f"{base}*.ebwt"))
    if not hits:
        # also allow if user entered full prefix path as basename by mistake
        alt = index_dir / base
        hits2 = list(index_dir.glob(f"{alt.name}*.ebwt"))
        raise FileNotFoundError(
            f"{label} not found: expected files like {index_dir}/{base}*.ebwt\n"
            f"Found none."
        )
    # return prefix for bowtie: "<dir>/<basename>"
    return str(index_dir / base)


@dataclass
class CmdSpec:
    name: str
    cmd: List[str]
    workdir: Optional[Path] = None


class Runner(QObject):
    started = Signal(str, list)              # stage name, cmd
    output = Signal(str)                     # line
    finished_stage = Signal(str, int)        # stage name, return code
    finished_all = Signal(bool)              # ok
    status = Signal(str)                     # status line

    def __init__(self) -> None:
        super().__init__()
        self._queue: List[CmdSpec] = []
        self._proc: Optional[QProcess] = None
        self._current: Optional[CmdSpec] = None
        self._stopping: bool = False

    def is_running(self) -> bool:
        return self._proc is not None and self._proc.state() != QProcess.NotRunning

    def stop(self) -> None:
        self._stopping = True
        if self._proc is None:
            return
        self.status.emit("Stopping…")
        try:
            self._proc.terminate()
        except Exception:
            pass

    def run_queue(self, cmds: List[CmdSpec]) -> None:
        if self.is_running():
            raise RuntimeError("Runner is already running.")
        self._stopping = False
        self._queue = list(cmds)
        self._run_next()

    def _run_next(self) -> None:
        if self._stopping:
            self._queue.clear()
            self._current = None
            self._cleanup_proc()
            self.finished_all.emit(False)
            return

        if not self._queue:
            self._current = None
            self._cleanup_proc()
            self.finished_all.emit(True)
            return

        self._current = self._queue.pop(0)
        spec = self._current
        self._proc = QProcess()
        self._proc.setProcessChannelMode(QProcess.MergedChannels)

        if spec.workdir is not None:
            self._proc.setWorkingDirectory(str(spec.workdir))

        self._proc.readyReadStandardOutput.connect(self._on_ready_read)
        self._proc.finished.connect(self._on_finished)
        self._proc.errorOccurred.connect(self._on_error)

        self.started.emit(spec.name, spec.cmd)
        self.status.emit(f"Running: {spec.name}")

        program = spec.cmd[0]
        args = spec.cmd[1:]
        self.output.emit(f"[DEBUG] Starting: {program} {' '.join(args[:3])}...\n")
        self._proc.start(program, args)

        if not self._proc.waitForStarted(5000):
            err_msg = self._proc.errorString() if self._proc else "Unknown error"
            self.output.emit(f"[ERROR] Failed to start process: {program}\n")
            self.output.emit(f"[ERROR] Reason: {err_msg}\n")
            self.finished_stage.emit(spec.name, 127)
            self._run_next()

    def _on_error(self, error: QProcess.ProcessError) -> None:
        error_names = {
            QProcess.FailedToStart: "FailedToStart",
            QProcess.Crashed: "Crashed",
            QProcess.Timedout: "Timedout",
            QProcess.WriteError: "WriteError",
            QProcess.ReadError: "ReadError",
            QProcess.UnknownError: "UnknownError",
        }
        self.output.emit(f"[ERROR] Process error: {error_names.get(error, error)}\n")

    def _on_ready_read(self) -> None:
        if self._proc is None:
            return
        data = bytes(self._proc.readAllStandardOutput()).decode(errors="replace")
        for line in data.splitlines(True):
            self.output.emit(line)

    def _on_finished(self, code: int, _status: QProcess.ExitStatus) -> None:
        spec = self._current
        if spec is None:
            return
        self.finished_stage.emit(spec.name, code)
        if self._stopping:
            self._queue.clear()
            self.finished_all.emit(False)
            self._cleanup_proc()
            return
        self._run_next()

    def _cleanup_proc(self) -> None:
        if self._proc is not None:
            try:
                self._proc.deleteLater()
            except Exception:
                pass
        self._proc = None


class FileRow(QWidget):
    def __init__(self, label: str, mode: str, placeholder: str = "") -> None:
        super().__init__()
        self.mode = mode  # "file" or "dir"
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)

        self.lbl = QLabel(label)
        self.lbl.setMinimumWidth(180)
        
        
        self.edit = QLineEdit()
        self.edit.setMinimumHeight(20)
        
        
        
        if placeholder:
            self.edit.setPlaceholderText(placeholder)
            self.edit.setMinimumWidth(420)
            self.edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            
        self.btn = QPushButton("Browse")
        self.btn.setMinimumWidth(90)
        self.btn.setMinimumHeight(44)
        self.btn.clicked.connect(self._browse)
        self.btn.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)

        lay.addWidget(self.lbl)
        lay.addWidget(self.edit, 1)
        lay.addWidget(self.btn)

    def text(self) -> str:
        return self.edit.text().strip()

    def setText(self, t: str) -> None:
        self.edit.setText(t)

    def _browse(self) -> None:
        if self.mode == "file":
            p, _ = QFileDialog.getOpenFileName(self, "Select file", self.text() or str(Path.cwd()))
            if p:
                self.setText(p)
        elif self.mode == "save":
            p, _ = QFileDialog.getSaveFileName(self, "Save as", self.text() or str(Path.cwd()))
            if p:
                self.setText(p)
        elif self.mode == "dir":
            p = QFileDialog.getExistingDirectory(self, "Select folder", self.text() or str(Path.cwd()))
            if p:
                self.setText(p)


class CollapsibleSection(QWidget):
    """A collapsible section with a toggle button and content area."""
    
    def __init__(self, title: str, parent=None) -> None:
        super().__init__(parent)
        
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(0)
        
        # Header button
        self._toggle_btn = QPushButton(f"▶ {title}")
        self._toggle_btn.setStyleSheet("""
            QPushButton {
                text-align: left;
                padding: 6px 10px;
                background-color: #3a3a3a;
                border: none;
                border-radius: 3px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #4a4a4a;
            }
        """)
        self._toggle_btn.clicked.connect(self._toggle)
        self._layout.addWidget(self._toggle_btn)
        
        # Content area
        self._content = QWidget()
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(15, 5, 5, 5)
        self._content.setVisible(False)
        self._layout.addWidget(self._content)
        
        self._title = title
        self._expanded = False
    
    def content_layout(self) -> QVBoxLayout:
        return self._content_layout
    
    def add_widget(self, widget: QWidget) -> None:
        self._content_layout.addWidget(widget)
    
    def _toggle(self) -> None:
        self._expanded = not self._expanded
        self._content.setVisible(self._expanded)
        arrow = "▼" if self._expanded else "▶"
        self._toggle_btn.setText(f"{arrow} {self._title}")
    
    def set_expanded(self, expanded: bool) -> None:
        if self._expanded != expanded:
            self._toggle()


class StageParamEditor(QWidget):
    """
    Editor widget for stage parameters based on STAGE_PARAMS configuration.
    Creates appropriate widgets (spinbox, checkbox, lineedit) for each param.
    """
    
    def __init__(self, stage_name: str, params: List[Dict[str, Any]], parent=None) -> None:
        super().__init__(parent)
        self.stage_name = stage_name
        self.params = params
        self.widgets: Dict[str, QWidget] = {}
        
        layout = QGridLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(8)
        
        row = 0
        col = 0
        max_cols = 4  # 2 params per row (label + widget each)
        
        for param in params:
            flag = param["flag"]
            ptype = param["type"]
            default = param["default"]
            help_text = param.get("help", "")
            
            # Label
            label_text = flag.replace("--", "").replace("-", " ").title()
            label = QLabel(label_text + ":")
            label.setToolTip(f"{flag}: {help_text}")
            layout.addWidget(label, row, col)
            col += 1
            
            # Widget based on type
            if ptype == "int":
                widget = QSpinBox()
                widget.setRange(-10000, 100000)
                widget.setValue(default if default is not None else 0)
                widget.setToolTip(help_text)
            elif ptype == "float":
                widget = QDoubleSpinBox()
                widget.setRange(-10000.0, 100000.0)
                widget.setDecimals(3)
                widget.setValue(default if default is not None else 0.0)
                widget.setToolTip(help_text)
            elif ptype == "bool":
                widget = QCheckBox()
                widget.setChecked(bool(default))
                widget.setToolTip(help_text)
            else:  # str
                widget = QLineEdit()
                widget.setText(str(default) if default is not None else "")
                widget.setToolTip(help_text)
            
            layout.addWidget(widget, row, col)
            self.widgets[flag] = widget
            col += 1
            
            # Move to next row after 2 params
            if col >= max_cols:
                col = 0
                row += 1
    
    def get_cli_args(self) -> List[str]:
        """Get CLI arguments based on current widget values."""
        args = []
        for param in self.params:
            flag = param["flag"]
            ptype = param["type"]
            default = param["default"]
            widget = self.widgets.get(flag)
            
            if widget is None:
                continue
            
            if ptype == "int":
                value = widget.value()
                if value != default:
                    args.extend([flag, str(value)])
            elif ptype == "float":
                value = widget.value()
                if value != default:
                    args.extend([flag, str(value)])
            elif ptype == "bool":
                if widget.isChecked() and not default:
                    args.append(flag)
            else:  # str
                value = widget.text().strip()
                if value and value != str(default):
                    args.extend([flag, value])
        
        return args


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1200, 760)

        self.runner = Runner()
        self.runner.started.connect(self._on_stage_started)
        self.runner.output.connect(self._on_output)
        self.runner.finished_stage.connect(self._on_stage_finished)
        self.runner.finished_all.connect(self._on_all_finished)
        self.runner.status.connect(self._set_status)

        self._report_html_to_open: Optional[Path] = None
        self._stages_total: int = 0
        self._stages_done: int = 0


        self._build_ui()




    def _build_ui(self) -> None:
        act_quit = QAction("Quit", self)
        act_quit.triggered.connect(self.close)
        self.menuBar().addMenu("File").addAction(act_quit)

        root = QWidget()
        self.setCentralWidget(root)
        main = QVBoxLayout(root)
        main.setContentsMargins(10, 10, 10, 10)

        top = QHBoxLayout()
        self.mode = QComboBox()
        self.mode.addItems(["sRNA-seq mode", "Sequence-only mode"])
        self.mode.currentIndexChanged.connect(self._sync_mode)

        self.btn_run = QPushButton("Run selected stages")
        self.btn_run.clicked.connect(self._run_clicked)
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop_clicked)
        
         


        top.addWidget(QLabel("Mode:"))
        top.addWidget(self.mode)
        
        
        top.addStretch(1)
         # --- START LOGO  ---
        # Try to find the logo file (prefer package asset; fallback to local file)
        logo_path = None
        try:
            from importlib.resources import files as rfiles
            logo_path = rfiles("mirpv_ng.assets").joinpath(LOGO_FILENAME)
        except Exception:
            pass
        if logo_path is None or (hasattr(logo_path, "exists") and not logo_path.exists()):
            # Fallback: look next to this GUI file
            logo_path = Path(__file__).parent / LOGO_FILENAME
        
        lbl_logo = QLabel()
        lbl_logo.setAlignment(Qt.AlignCenter)
        lbl_logo.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        
        

        if logo_path.exists():
            # Load image
            pix = QPixmap(str(logo_path))
            # Scale it (Width=300, Height=120) to fit nicely
            if not pix.isNull():
                lbl_logo.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
                scaled = pix.scaled(120, 50, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                pix = pix.scaled(180, 60, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                lbl_logo.setPixmap(pix)

            else:
                lbl_logo.setText(APP_TITLE)
        else:
            # Fallback if image missing
            lbl_logo.setText(APP_TITLE)
            lbl_logo.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            #lbl_logo.setStyleSheet("font-size: 24px; font-weight: bold; color: #38bdf8; margin: 15px;")
        
        
        top.addWidget(lbl_logo, 0, Qt.AlignCenter)
# --- END LOGO  ---
        top.addStretch(1)
        
        
        
        
#        top.addStretch(1)
        
        top.addWidget(self.btn_run)
        top.addWidget(self.btn_stop)

        splitter = QSplitter(Qt.Horizontal)

        # LEFT
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)

        self.tabs = QTabWidget()

        # ---------- sRNA-seq TAB ----------
        srna = QWidget()
        srna_v = QVBoxLayout(srna)

        grp_run = QGroupBox("Run naming / output")
        frm_run = QFormLayout(grp_run)
        self.sample_id = QLineEdit("TEST1")
        self.out_root = FileRow("Output root folder", "dir", "e.g., results/TEST1_run")
        self.out_root.setText(str(Path("results") / "TEST1_gui_run"))
        frm_run.addRow("Sample ID", self.sample_id)
        frm_run.addRow(self.out_root)

        grp_inputs = QGroupBox("Inputs (sRNA-seq)")
        grid = QGridLayout(grp_inputs)

        self.fastq = FileRow("FASTQ(.gz)", "file", "reads.fastq.gz")

        # Bowtie genome index: folder + basename
        self.bowtie_index_dir = FileRow("Bowtie1 genome index folder", "dir", "refs/bowtie1_hg38/")
        self.bowtie_index_base = QLineEdit("hg38")

        # Blocklist index: folder + basename
        self.blocklist_index_dir = FileRow("Blocklist index folder", "dir", "refs/indexes/blocklist/")
        self.blocklist_index_base = QLineEdit("rfam_trna")
        self.blocklist_name = QLineEdit("rfam")
        self.blocklist_enable = QCheckBox("Enable blocklist")
        self.blocklist_enable.setChecked(True)

        self.genome_fa = FileRow("Genome FASTA", "file", "refs/hg38/hg38.fa")
        self.repeat_bed = FileRow("Repeat BED (repClass)", "file", "refs/repeats/hg38/hg38.rmsk.repClass.bed")

        self.mirgenedb_gff = FileRow("MirGeneDB GFF", "file", "refs/known/hsa/hsa_mirgene.gff")
        self.mirbase_gff = FileRow("miRBase GFF3", "file", "refs/known/hsa/hsa_mirbase.gff3")

        self.rf_model = FileRow("Pre-miRNA RF model", "file", "models/hsa_premirna_rf_extended_tier2_v6.pkl")
        self.mature_model = FileRow("Mature XGBRanker model", "file", "models/hsa_mature_xgbrank_v3.pkl")

        r = 0
        grid.addWidget(self.fastq, r, 0, 1, 2); r += 1
        grid.addWidget(self.bowtie_index_dir, r, 0, 1, 2); r += 1
        grid.addWidget(QLabel("Bowtie genome index basename"), r, 0)
        grid.addWidget(self.bowtie_index_base, r, 1); r += 1

        grid.addWidget(self.blocklist_enable, r, 0, 1, 2); r += 1
        grid.addWidget(self.blocklist_index_dir, r, 0, 1, 2); r += 1
        grid.addWidget(QLabel("Blocklist index basename"), r, 0)
        grid.addWidget(self.blocklist_index_base, r, 1); r += 1
        grid.addWidget(QLabel("Blocklist name"), r, 0)
        grid.addWidget(self.blocklist_name, r, 1); r += 1

        grid.addWidget(self.genome_fa, r, 0, 1, 2); r += 1
        grid.addWidget(self.repeat_bed, r, 0, 1, 2); r += 1
        grid.addWidget(self.mirgenedb_gff, r, 0, 1, 2); r += 1
        grid.addWidget(self.mirbase_gff, r, 0, 1, 2); r += 1
        grid.addWidget(self.rf_model, r, 0, 1, 2); r += 1
        grid.addWidget(self.mature_model, r, 0, 1, 2); r += 1

# --- Advanced toggle + collapsible params ---
        adv_bar = QWidget()
        adv_bar_l = QHBoxLayout(adv_bar)
        adv_bar_l.setContentsMargins(0, 0, 0, 0)

        self.adv_toggle = QCheckBox("Show advanced parameters")
        self.adv_toggle.setChecked(False)

        adv_bar_l.addWidget(self.adv_toggle)
        adv_bar_l.addStretch(1)

        grp_params = QGroupBox("Advanced parameters (sRNA-seq)")
        grp_params.setVisible(False)  # hidden by default
        prm = QGridLayout(grp_params)

        # connect toggle
        self.adv_toggle.toggled.connect(grp_params.setVisible)


        self.threads = QSpinBox(); self.threads.setRange(1, 256); self.threads.setValue(16)
        self.adapter = QLineEdit(DEFAULT_ADAPTER)
        self.use_scipy = QCheckBox("Use SciPy peak calling"); self.use_scipy.setChecked(True)
        self.smooth_w = QSpinBox(); self.smooth_w.setRange(0, 99); self.smooth_w.setValue(3)
        self.peak_distance = QSpinBox(); self.peak_distance.setRange(1, 500); self.peak_distance.setValue(35)
        self.scipy_width_min = QSpinBox(); self.scipy_width_min.setRange(1, 50); self.scipy_width_min.setValue(2)
        self.pad1 = QSpinBox(); self.pad1.setRange(10, 1000); self.pad1.setValue(70)
        self.pad2 = QSpinBox(); self.pad2.setRange(10, 1000); self.pad2.setValue(100)
        self.species = QLineEdit("hsa")
        self.feature_set = QComboBox(); self.feature_set.addItems(["extended", "core36"])
        self.tier2 = QCheckBox("Tier2 (soft-gated)"); self.tier2.setChecked(True)

        row = 0
        prm.addWidget(QLabel("Threads"), row, 0); prm.addWidget(self.threads, row, 1)
        prm.addWidget(QLabel("Species"), row, 2); prm.addWidget(self.species, row, 3); row += 1
        prm.addWidget(QLabel("Adapter"), row, 0); prm.addWidget(self.adapter, row, 1, 1, 3); row += 1
        prm.addWidget(self.use_scipy, row, 0, 1, 2)
        prm.addWidget(QLabel("Smooth w"), row, 2); prm.addWidget(self.smooth_w, row, 3); row += 1
        prm.addWidget(QLabel("Peak distance"), row, 0); prm.addWidget(self.peak_distance, row, 1)
        prm.addWidget(QLabel("SciPy width min"), row, 2); prm.addWidget(self.scipy_width_min, row, 3); row += 1
        prm.addWidget(QLabel("Pads"), row, 0)
        pad_box = QWidget(); pad_l = QHBoxLayout(pad_box); pad_l.setContentsMargins(0, 0, 0, 0)
        pad_l.addWidget(self.pad1); pad_l.addWidget(QLabel(",")); pad_l.addWidget(self.pad2); pad_l.addStretch(1)
        prm.addWidget(pad_box, row, 1)
        prm.addWidget(QLabel("Feature set"), row, 2); prm.addWidget(self.feature_set, row, 3); row += 1
        prm.addWidget(self.tier2, row, 0, 1, 2); row += 1
        
        # Parallelism controls (visible in advanced mode)
        prm.addWidget(QLabel("--- Parallelism ---"), row, 0, 1, 4); row += 1
        
        self.parallel_jobs = QSpinBox()
        self.parallel_jobs.setRange(1, 128)
        self.parallel_jobs.setValue(1)
        self.parallel_jobs.setToolTip("Number of parallel workers (1 = sequential)")
        
        self.parallel_backend = QComboBox()
        self.parallel_backend.addItems(["process", "thread"])
        self.parallel_backend.setToolTip("Parallelization backend")
        
        self.parallel_tmpdir = QLineEdit()
        self.parallel_tmpdir.setPlaceholderText("(system default)")
        self.parallel_tmpdir.setToolTip("Temporary directory for parallel workers")
        
        prm.addWidget(QLabel("Jobs"), row, 0); prm.addWidget(self.parallel_jobs, row, 1)
        prm.addWidget(QLabel("Backend"), row, 2); prm.addWidget(self.parallel_backend, row, 3); row += 1
        prm.addWidget(QLabel("Tmpdir"), row, 0); prm.addWidget(self.parallel_tmpdir, row, 1, 1, 3); row += 1

        grp_stages = QGroupBox("Stages (sRNA-seq)")
        st = QGridLayout(grp_stages)

        self.st_fastq_to_peaks = QCheckBox("01 fastq-to-peaks (Stage 1–8)")
        self.st_candidates_to_scored = QCheckBox("02 candidates-to-scored (Stage 9)")
        self.st_scored_to_peaks = QCheckBox("03 scored-to-peaks (Stage 9.5)")
        self.st_peaks_to_known = QCheckBox("04 peaks-to-known (Stage 10)")
        self.st_peaks_to_finalists = QCheckBox("05 peaks-to-finalists (Stage 11)")
        self.st_finalists_to_struct = QCheckBox("06 finalists-to-struct (Stage 12)")
        self.st_predict_mature = QCheckBox("07 predict-mature (mature XGBRanker)")
        self.st_final_candidates = QCheckBox("08 final-candidates (Stage 13 merge)")
        self.st_make_report = QCheckBox("09 make-report (HTML+PDF)")

        for cb in [
            self.st_fastq_to_peaks,
            self.st_candidates_to_scored,
            self.st_scored_to_peaks,
            self.st_peaks_to_known,
            self.st_peaks_to_finalists,
            self.st_finalists_to_struct,
            self.st_predict_mature,
            self.st_final_candidates,
            self.st_make_report,
        ]:
            cb.setChecked(True)

        st.addWidget(self.st_fastq_to_peaks, 0, 0, 1, 2)
        st.addWidget(self.st_candidates_to_scored, 1, 0, 1, 2)
        st.addWidget(self.st_scored_to_peaks, 2, 0, 1, 2)
        st.addWidget(self.st_peaks_to_known, 3, 0, 1, 2)
        st.addWidget(self.st_peaks_to_finalists, 4, 0, 1, 2)
        st.addWidget(self.st_finalists_to_struct, 5, 0, 1, 2)
        st.addWidget(self.st_predict_mature, 6, 0, 1, 2)
        st.addWidget(self.st_final_candidates, 7, 0, 1, 2)
        st.addWidget(self.st_make_report, 8, 0, 1, 2)

        srna_v.addWidget(grp_run)
        srna_v.addWidget(grp_inputs)

        srna_v.addWidget(adv_bar)      
        srna_v.addWidget(grp_params) 

        srna_v.addWidget(grp_stages)

        # --- Stage-specific Advanced Parameters (collapsible) ---
        # These are hidden when Advanced mode is OFF, visible when ON
        self.adv_stage_params_container = QWidget()
        adv_stage_layout = QVBoxLayout(self.adv_stage_params_container)
        adv_stage_layout.setContentsMargins(0, 5, 0, 0)
        adv_stage_layout.setSpacing(2)
        
        # Create collapsible sections for each stage with params
        self.stage_param_editors: Dict[str, StageParamEditor] = {}
        
        stage_display_names = {
            "fastq-to-peaks": "Stage 1-8: fastq-to-peaks",
            "candidates-to-scored": "Stage 9: candidates-to-scored",
            "peaks-to-known": "Stage 10: peaks-to-known",
            "peaks-to-finalists": "Stage 11: peaks-to-finalists",
            "predict-mature": "Stage 7: predict-mature",
        }
        
        for stage_name, params in STAGE_PARAMS.items():
            display_name = stage_display_names.get(stage_name, stage_name)
            section = CollapsibleSection(display_name)
            editor = StageParamEditor(stage_name, params)
            section.add_widget(editor)
            adv_stage_layout.addWidget(section)
            self.stage_param_editors[stage_name] = editor
        
        self.adv_stage_params_container.setVisible(False)  # Hidden by default
        
        # Wire advanced toggle to show/hide stage params
        self.adv_toggle.toggled.connect(self.adv_stage_params_container.setVisible)
        
        srna_v.addWidget(self.adv_stage_params_container)

        srna_v.addStretch(1)

        self.tabs.addTab(srna, "sRNA-seq")

        # ---------- SEQUENCE-ONLY TAB ----------
        seq = QWidget()
        seq_v = QVBoxLayout(seq)
        
        seq_run = QGroupBox("Sequence-only: score-fasta (+optional mature prediction)")
        seq_frm = QFormLayout(seq_run)

        self.seq_fasta = FileRow("Input FASTA", "file", "candidates.fa")
        self.seq_model = FileRow("RF model", "file", "models/hsa_premirna_rf_extended_tier2_v6.pkl")
        
        # Output Selection (Folder + Filename)
        self.seq_out_dir = FileRow("Output Folder", "dir", str(Path.cwd()))
        self.seq_out_name = QLineEdit("out.scored.tsv")
      
        # Parameters
        self.seq_species = QLineEdit("hsa")
        self.seq_feature_set = QComboBox(); self.seq_feature_set.addItems(["extended", "core36"])
        self.seq_tier2 = QCheckBox("Enable tier2 filters")
        self.seq_tier2.setChecked(True)
        
        
        # Mature Prediction (Added/Verified)
        self.seq_do_mature = QCheckBox("Also run predict-mature miRNA")
        self.seq_mature_model = FileRow("Mature model", "file", "models/hsa_mature_xgbrank_v3.pkl")
        self.seq_mature_name = QLineEdit("mature_predictions.tsv")
        
        self.seq_do_mature.stateChanged.connect(self._sync_seq_mature_enabled)
        
        
        
        # Layout for Sequence tab
        
        seq_frm.addRow(self.seq_fasta)
        seq_frm.addRow(self.seq_model)
        
        # Separator line
        line1 = QWidget(); line1.setFixedHeight(1); line1.setStyleSheet("background-color: #cccccc;")
        seq_frm.addRow(line1)     
        
        
        
        seq_frm.addRow(self.seq_out_dir)
        seq_frm.addRow("Output Filename", self.seq_out_name)
        seq_frm.addRow("Species", self.seq_species)
        seq_frm.addRow("Feature set", self.seq_feature_set)
        seq_frm.addRow(self.seq_tier2)


        # Separator line
        line2 = QWidget(); line2.setFixedHeight(1); line2.setStyleSheet("background-color: #cccccc;")
        seq_frm.addRow(line2)

        seq_frm.addRow(self.seq_do_mature)
        seq_frm.addRow(self.seq_mature_model)
        seq_frm.addRow("Mature Filename", self.seq_mature_name)

        # --- Sequence-only Advanced Parameters (collapsible) ---
        seq_adv_toggle = QCheckBox("Show advanced parameters")
        seq_adv_toggle.setChecked(False)

        self.seq_adv_container = QWidget()
        seq_adv_layout = QVBoxLayout(self.seq_adv_container)
        seq_adv_layout.setContentsMargins(0, 5, 0, 0)
        seq_adv_layout.setSpacing(2)

        if "score-fasta" in STAGE_PARAMS:
            section = CollapsibleSection("score-fasta parameters")
            editor = StageParamEditor("score-fasta", STAGE_PARAMS["score-fasta"])
            section.add_widget(editor)
            seq_adv_layout.addWidget(section)
            self.stage_param_editors["score-fasta"] = editor

        # Parallelism for seq-only mode
        seq_par_section = CollapsibleSection("Parallelism")
        par_w = QWidget()
        par_l = QGridLayout(par_w)
        par_l.setContentsMargins(5, 5, 5, 5)

        self.seq_threads = QSpinBox()
        self.seq_threads.setRange(1, 256)
        self.seq_threads.setValue(1)
        self.seq_threads.setToolTip("Number of parallel workers")

        self.seq_backend = QComboBox()
        self.seq_backend.addItems(["process", "thread", "external"])
        self.seq_backend.setToolTip("Parallelization backend")

        par_l.addWidget(QLabel("Threads"), 0, 0)
        par_l.addWidget(self.seq_threads, 0, 1)
        par_l.addWidget(QLabel("Backend"), 0, 2)
        par_l.addWidget(self.seq_backend, 0, 3)

        seq_par_section.add_widget(par_w)
        seq_adv_layout.addWidget(seq_par_section)

        self.seq_adv_container.setVisible(False)
        seq_adv_toggle.toggled.connect(self.seq_adv_container.setVisible)

        seq_v.addWidget(seq_run)
        seq_v.addWidget(seq_adv_toggle)
        seq_v.addWidget(self.seq_adv_container)
        seq_v.addStretch(1)
        self.tabs.addTab(seq, "Sequence-only")

        left_layout.addWidget(self.tabs)

        # RIGHT: log
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)

        self.status = QLabel("Ready.")
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        mono = QFont("Monospace"); mono.setStyleHint(QFont.TypeWriter)
        self.log.setFont(mono)

        # --- Command Preview ---
        preview_label = QLabel("Command Preview:")
        preview_label.setStyleSheet("font-weight: bold; margin-top: 6px;")
        self.cmd_preview = QTextEdit()
        self.cmd_preview.setReadOnly(True)
        self.cmd_preview.setMaximumHeight(80)
        self.cmd_preview.setFont(QFont("Monospace"))
        self.cmd_preview.setPlaceholderText("Click 'Preview' to see the command...")

        btnrow = QHBoxLayout()
        self.btn_clear = QPushButton("Clear log")
        self.btn_clear.clicked.connect(lambda: self.log.clear())
        self.btn_preview = QPushButton("Preview")
        self.btn_preview.setToolTip("Show the CLI command that will be run")
        self.btn_preview.clicked.connect(self._update_preview)
        self.btn_export_config = QPushButton("Export Config")
        self.btn_export_config.setToolTip("Save current settings to a JSON config file")
        self.btn_export_config.clicked.connect(self._export_config)
        self.btn_open_report = QPushButton("Open report.html")
        self.btn_open_report.setEnabled(False)
        self.btn_open_report.clicked.connect(self._open_report_html)
        btnrow.addWidget(self.btn_clear)
        btnrow.addWidget(self.btn_preview)
        btnrow.addWidget(self.btn_export_config)
        btnrow.addStretch(1)
        btnrow.addWidget(self.btn_open_report)

        right_layout.addWidget(self.progress)
        right_layout.addWidget(self.status)
        right_layout.addWidget(preview_label)
        right_layout.addWidget(self.cmd_preview)
        right_layout.addLayout(btnrow)
        right_layout.addWidget(self.log, 1)

        splitter.addWidget(left)
        splitter.setStretchFactor(1, 3)
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)

        main.addLayout(top)
        main.addWidget(splitter, 1)

        self._sync_mode()
        self._sync_seq_mature_enabled()


    # ------------------- sync helpers -------------------

    def _sync_mode(self) -> None:
        # 0: sRNA-seq, 1: sequence-only
        self.tabs.setCurrentIndex(self.mode.currentIndex())

    def _sync_seq_mature_enabled(self) -> None:
        en = self.seq_do_mature.isChecked()
        self.seq_mature_model.setEnabled(en)
        self.seq_mature_name.setEnabled(en)
    # ------------------- runner hooks -------------------

    def _append(self, text: str) -> None:
        """Append text to the log widget."""
        self.log.moveCursor(self.log.textCursor().End)
        self.log.insertPlainText(text)
        self.log.moveCursor(self.log.textCursor().End)






    # ------------------- button actions -------------------

    def _stop_clicked(self) -> None:
        self.runner.stop()
        self.btn_stop.setEnabled(False)

    def _run_clicked(self) -> None:
        if self.runner.is_running():
            QMessageBox.warning(self, "Running", "A run is already in progress.")
            return

        try:
            self.btn_open_report.setEnabled(False)
            self._report_html_to_open = None
            cmds = self._build_commands()
        except Exception as e:
            QMessageBox.critical(self, "Invalid configuration", str(e))
            return

        if not cmds:
            QMessageBox.information(self, "Nothing to do", "No stages selected.")
            return

        self.log.append("[GUI] Starting run…\n")
        self.log.append(f"[GUI] Total stages: {len(cmds)}\n")
        self._stages_total = len(cmds)
        self._stages_done = 0
        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.progress.setValue(0)

        self.runner.run_queue(cmds)

    def _open_report_html(self) -> None:
        if not self._report_html_to_open:
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._report_html_to_open)))



    def _build_commands(self) -> List[CmdSpec]:
        cmds: List[CmdSpec] = []

        # ==========================================
        # MODE 1: SEQUENCE-ONLY
        # ==========================================
        if self.mode.currentIndex() == 1:
            fasta = must_exist_file(self.seq_fasta.text(), "Input FASTA")
            model = must_exist_file(self.seq_model.text(), "RF model")

            # Output Paths
            out_dir = Path(self.seq_out_dir.text() or ".")
            out_name = self.seq_out_name.text().strip() or "out.scored.tsv"
            out_tsv = out_dir / out_name
            ensure_dir(out_tsv.parent)

            # 1. Score Fasta Command
            cmd = [sys.executable, "-m", "mirpv_ng.cli",
                "score-fasta",
                "--model", str(model),
                "--fasta", str(fasta),
                "--out", str(out_tsv),
                "--species", (self.seq_species.text().strip() or "hsa"),
                "--feature-set", self.seq_feature_set.currentText(),
            ]
            if self.seq_tier2.isChecked():
                cmd.append("--tier2")

            # Parallelism
            threads = self.seq_threads.value()
            if threads > 1:
                cmd.extend(["--threads", str(threads)])
                cmd.extend(["--backend", self.seq_backend.currentText()])

            # Advanced stage params for score-fasta
            if "score-fasta" in self.stage_param_editors:
                cmd.extend(self.stage_param_editors["score-fasta"].get_cli_args())

            cmds.append(CmdSpec(name="score-fasta", cmd=cmd))

            # 2. Mature Prediction Command (Optional)
            if self.seq_do_mature.isChecked():
                m_model = must_exist_file(self.seq_mature_model.text(), "Mature model")
                m_name = self.seq_mature_name.text().strip() or "mature_predictions.tsv"
                m_out = out_dir / m_name
                
                cmd2 = [sys.executable, "-m", "mirpv_ng.cli",
                    "predict-mature",
                    "--mature-model", str(m_model),
                    "--fasta", str(fasta),
                    "--out", str(m_out),
                ]
                # Add predict-mature advanced params
                if "predict-mature" in self.stage_param_editors:
                    cmd2.extend(self.stage_param_editors["predict-mature"].get_cli_args())
                cmds.append(CmdSpec(name="predict-mature", cmd=cmd2))

            return cmds


        # sRNA-seq mode
        sample = self.sample_id.text().strip()
        if not sample:
            raise ValueError("Sample ID is required.")
        outroot = Path(self.out_root.text())
        ensure_dir(outroot)

        fastq = must_exist_file(self.fastq.text(), "FASTQ")
        genome_fa = must_exist_file(self.genome_fa.text(), "Genome FASTA")
        repeat_bed = must_exist_file(self.repeat_bed.text(), "Repeat BED")
        mirgenedb_gff = must_exist_file(self.mirgenedb_gff.text(), "MirGeneDB GFF")
        mirbase_gff = must_exist_file(self.mirbase_gff.text(), "miRBase GFF3")
        rf_model = must_exist_file(self.rf_model.text(), "Pre-miRNA RF model")
        mature_model = must_exist_file(self.mature_model.text(), "Mature model")

        # Bowtie genome index prefix = dir + base
        bowtie_dir = must_exist_dir(self.bowtie_index_dir.text(), "Bowtie genome index folder")
        bowtie_base = self.bowtie_index_base.text().strip()
        bowtie_prefix = validate_bowtie1_index_prefix(bowtie_dir, bowtie_base, "Bowtie genome index")

        # Blocklist index prefix = dir + base (optional)
        blocklist_prefix: Optional[str] = None
        if self.blocklist_enable.isChecked():
            blk_dir = must_exist_dir(self.blocklist_index_dir.text(), "Blocklist index folder")
            blk_base = self.blocklist_index_base.text().strip()
            blocklist_prefix = validate_bowtie1_index_prefix(blk_dir, blk_base, "Blocklist index")
        blocklist_name = self.blocklist_name.text().strip() or "rfam"

        threads = int(self.threads.value())
        adapter = self.adapter.text().strip()
        if not adapter:
            raise ValueError("Adapter is required.")

        pads = [int(self.pad1.value()), int(self.pad2.value())]
        if pads[0] == pads[1]:
            raise ValueError("Pads must be two different integers (e.g., 70 and 100).")

        use_scipy = self.use_scipy.isChecked()
        smooth_w = int(self.smooth_w.value())
        peak_dist = int(self.peak_distance.value())
        scipy_width_min = int(self.scipy_width_min.value())

        species = self.species.text().strip() or "hsa"
        feature_set = self.feature_set.currentText()
        tier2 = self.tier2.isChecked()

        # stage folders (kept consistent with your ladder)
        s01 = outroot / "01_fastq_to_peaks"
        s02 = outroot / "02_candidates_to_scored"
        s03 = outroot / "03_scored_to_peaks"
        s04 = outroot / "04_peaks_to_known"
        s05 = outroot / "05_peaks_to_finalists"
        s06 = outroot / "06_finalists_to_struct"
        s07 = outroot / "07_mature_prediction"
        s08 = outroot / "08_final_candidates"
        s09_final = outroot / "09_final_report"
        s10_report = outroot / "10_report"

        candidates_tsv = s01 / "candidates.tsv"
        candidates_fa = s01 / "candidates.fa"
        candidates_scored_tsv = s02 / "candidates.scored.tsv"
        peaks_scored_tsv = s03 / "peaks.scored.tsv"
        peaks_known_tsv = s04 / "peaks.known.tsv"
        strict_finalists_tsv = s05 / "strict_finalists.tsv"
        candidates_struct_tsv = s06 / "candidates_struct.tsv"
        candidates_struct_fa = s06 / "candidates_struct.fa"
        mature_predictions_tsv = s07 / "mature_predictions.tsv"
        final_candidates_tsv = s08 / "final_candidates.tsv"
        rejects_merged_tsv = s09_final / "rejects.merged.tsv"  # assumed
        report_html = s10_report / "report.html"

        cmds: List[CmdSpec] = []

        if self.st_fastq_to_peaks.isChecked():
            ensure_dir(s01)
            cmd = [
                sys.executable, "-m", "mirpv_ng.cli", "fastq-to-peaks",
                "--fastq", str(fastq),
                "--sample-id", sample,
                "--outdir", str(s01),
                "--bowtie-index", bowtie_prefix,
                "--threads", str(threads),
                "--adapter", adapter,
                "--repeat-bed", str(repeat_bed),
                "--genome-fasta", str(genome_fa),
                "--pads", str(pads[0]), str(pads[1]),
            ]
            if blocklist_prefix is not None:
                cmd += ["--blocklist-index", blocklist_prefix, "--blocklist-name", blocklist_name]
            if use_scipy:
                cmd += [
                    "--use-scipy",
                    "--smooth-w", str(smooth_w),
                    "--peak-distance", str(peak_dist),
                    "--scipy-width-min", str(scipy_width_min),
                ]
            # Add advanced stage-specific params if Advanced mode is ON
            if self.adv_toggle.isChecked() and "fastq-to-peaks" in self.stage_param_editors:
                cmd.extend(self.stage_param_editors["fastq-to-peaks"].get_cli_args())
            cmds.append(CmdSpec("01_fastq_to_peaks", cmd))

        if self.st_candidates_to_scored.isChecked():
            ensure_dir(s02)
            cmd = [
                sys.executable, "-m", "mirpv_ng.cli", "candidates-to-scored",
                "--candidates-tsv", str(candidates_tsv),
                "--candidates-fa", str(candidates_fa),
                "--model", str(rf_model),
                "--outdir", str(s02),
                "--sample-id", sample,
                "--species", species,
                "--feature-set", feature_set,
            ]
            if tier2:
                cmd += ["--tier2"]
            # Add advanced stage-specific params if Advanced mode is ON
            if self.adv_toggle.isChecked() and "candidates-to-scored" in self.stage_param_editors:
                cmd.extend(self.stage_param_editors["candidates-to-scored"].get_cli_args())
            cmds.append(CmdSpec("02_candidates_to_scored", cmd))

        if self.st_scored_to_peaks.isChecked():
            ensure_dir(s03)
            cmd = [
                sys.executable, "-m", "mirpv_ng.cli", "scored-to-peaks",
                "--scored-tsv", str(candidates_scored_tsv),
                "--outdir", str(s03),
            ]
            cmds.append(CmdSpec("03_scored_to_peaks", cmd))

        if self.st_peaks_to_known.isChecked():
            ensure_dir(s04)
            cmd = [
                sys.executable, "-m", "mirpv_ng.cli", "peaks-to-known",
                "--peaks-tsv", str(peaks_scored_tsv),
                "--outdir", str(s04),
                "--sample-id", sample,
                "--mirgenedb-gff", str(mirgenedb_gff),
                "--mirbase-gff", str(mirbase_gff),
            ]
            # Add advanced stage-specific params if Advanced mode is ON
            if self.adv_toggle.isChecked() and "peaks-to-known" in self.stage_param_editors:
                cmd.extend(self.stage_param_editors["peaks-to-known"].get_cli_args())
            cmds.append(CmdSpec("04_peaks_to_known", cmd))

        if self.st_peaks_to_finalists.isChecked():
            ensure_dir(s05)
            cmd = [
                sys.executable, "-m", "mirpv_ng.cli", "peaks-to-finalists",
                "--peaks-scored-tsv", str(peaks_scored_tsv),
                "--peaks-known-tsv", str(peaks_known_tsv),
                "--outdir", str(s05),
                "--sample-id", sample,
            ]
            # Add advanced stage-specific params if Advanced mode is ON
            if self.adv_toggle.isChecked() and "peaks-to-finalists" in self.stage_param_editors:
                cmd.extend(self.stage_param_editors["peaks-to-finalists"].get_cli_args())
            cmds.append(CmdSpec("05_peaks_to_finalists", cmd))

        if self.st_finalists_to_struct.isChecked():
            ensure_dir(s06)
            cmd = [
                sys.executable, "-m", "mirpv_ng.cli", "finalists-to-struct",
                "--strict-finalists-tsv", str(strict_finalists_tsv),
                "--candidates-fa", str(candidates_fa),
                "--outdir", str(s06),
                "--sample-id", sample,
                "--threads", str(threads),
            ]
            cmds.append(CmdSpec("06_finalists_to_struct", cmd))

        if self.st_predict_mature.isChecked():
            ensure_dir(s07)
            cmd = [
                sys.executable, "-m", "mirpv_ng.cli", "predict-mature",
                "--mature-model", str(mature_model),
                "--fasta", str(candidates_struct_fa),
                "--out", str(mature_predictions_tsv),
            ]
            # Add advanced stage-specific params if Advanced mode is ON
            if self.adv_toggle.isChecked() and "predict-mature" in self.stage_param_editors:
                cmd.extend(self.stage_param_editors["predict-mature"].get_cli_args())
            cmds.append(CmdSpec("07_predict_mature", cmd))

        if self.st_final_candidates.isChecked():
            ensure_dir(s08)
            cmd = [
                sys.executable, "-m", "mirpv_ng.cli", "final-candidates",
                "--candidates-struct-tsv", str(candidates_struct_tsv),
                "--mature-tsv", str(mature_predictions_tsv),
                "--outdir", str(s08),
                "--sample-id", sample,
            ]
            cmds.append(CmdSpec("08_final_candidates", cmd))

        if self.st_make_report.isChecked():
            ensure_dir(s10_report)
            # assumes rejects.merged.tsv exists at OUTROOT/09_final_report/
            cmd = [
                sys.executable, "-m", "mirpv_ng.make_report",
                "--sample-id", sample,
                "--outdir", str(s10_report),
                "--final-candidates-tsv", str(final_candidates_tsv),
                "--candidates-struct-tsv", str(candidates_struct_tsv),
                "--rejects-merged-tsv", str(rejects_merged_tsv),
            ]
            cmds.append(CmdSpec("09_make_report", cmd))
            self._report_html_to_open = report_html

        return cmds

    def _on_stage_started(self, name: str, cmd: list) -> None:
        self._append(f"\n=== [{name}] ===\n")
        self._append(f"$ {' '.join(map(str, cmd))}\n")

    def _on_output(self, line: str) -> None:
        self._append(line)

    def _on_stage_finished(self, name: str, rc: int) -> None:
        self._stages_done += 1
        pct = int((self._stages_done / max(1, self._stages_total)) * 100)
        self.progress.setValue(pct)
        self._append(f"\n[GUI] Stage finished: {name} (rc={rc})\n")
        if rc != 0:
            self._append("[GUI] Non-zero exit code. Stopping remaining stages.\n")
            self.runner.stop()

    def _on_all_finished(self, ok: bool) -> None:
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        if ok:
            self._set_status("Done.")
            if self._report_html_to_open and self._report_html_to_open.exists():
                self.btn_open_report.setEnabled(True)
                self._append(f"[GUI] Report ready: {self._report_html_to_open}\n")
            else:
                self.btn_open_report.setEnabled(False)
        else:
            self._set_status("Stopped / failed.")
            self.btn_open_report.setEnabled(False)

    def _set_status(self, s: str) -> None:
        self.status.setText(s)

    def _update_preview(self) -> None:
        """Show the exact CLI command(s) that would be run."""
        try:
            cmds = self._build_commands()
            lines = []
            for c in cmds:
                lines.append(f"# {c.name}")
                lines.append(" \\ \n    ".join(c.cmd))
                lines.append("")
            self.cmd_preview.setPlainText("\n".join(lines) if lines else "(no stages selected)")
        except Exception as e:
            self.cmd_preview.setPlainText(f"Error: {e}")

    def _export_config(self) -> None:
        """Export current GUI settings to a JSON config file."""
        config: Dict[str, Any] = {}
        if self.mode.currentIndex() == 1:
            # Sequence-only mode
            config["mode"] = "sequence-only"
            config["fasta"] = self.seq_fasta.text()
            config["model"] = self.seq_model.text()
            config["out_dir"] = self.seq_out_dir.text()
            config["out_name"] = self.seq_out_name.text()
            config["species"] = self.seq_species.text()
            config["feature_set"] = self.seq_feature_set.currentText()
            config["tier2"] = self.seq_tier2.isChecked()
            config["threads"] = self.seq_threads.value()
            config["backend"] = self.seq_backend.currentText()
            config["predict_mature"] = self.seq_do_mature.isChecked()
            if self.seq_do_mature.isChecked():
                config["mature_model"] = self.seq_mature_model.text()
                config["mature_out_name"] = self.seq_mature_name.text()
            # Advanced params
            if "score-fasta" in self.stage_param_editors:
                config["score_fasta_params"] = self.stage_param_editors["score-fasta"].get_cli_args()
        else:
            # sRNA-seq mode
            config["mode"] = "srna-seq"
            config["sample_id"] = self.sample_id.text()
            config["out_root"] = self.out_root.text()
            config["fastq"] = self.fastq.text()
            config["bowtie_index_dir"] = self.bowtie_index_dir.text()
            config["bowtie_index_base"] = self.bowtie_index_base.text()
            config["genome_fasta"] = self.genome_fa.text()
            config["repeat_bed"] = self.repeat_bed.text()
            config["mirgenedb_gff"] = self.mirgenedb_gff.text()
            config["mirbase_gff"] = self.mirbase_gff.text()
            config["rf_model"] = self.rf_model.text()
            config["mature_model"] = self.mature_model.text()
            config["threads"] = self.threads.value()
            config["adapter"] = self.adapter.text()
            config["species"] = self.species.text()
            config["feature_set"] = self.feature_set.currentText()
            config["tier2"] = self.tier2.isChecked()
            config["use_scipy"] = self.use_scipy.isChecked()
            config["pads"] = [self.pad1.value(), self.pad2.value()]
            # Stage selections
            config["stages"] = {
                "fastq_to_peaks": self.st_fastq_to_peaks.isChecked(),
                "candidates_to_scored": self.st_candidates_to_scored.isChecked(),
                "scored_to_peaks": self.st_scored_to_peaks.isChecked(),
                "peaks_to_known": self.st_peaks_to_known.isChecked(),
                "peaks_to_finalists": self.st_peaks_to_finalists.isChecked(),
                "finalists_to_struct": self.st_finalists_to_struct.isChecked(),
                "predict_mature": self.st_predict_mature.isChecked(),
                "final_candidates": self.st_final_candidates.isChecked(),
                "make_report": self.st_make_report.isChecked(),
            }
            # Advanced params per stage
            for stage_name, editor in self.stage_param_editors.items():
                args = editor.get_cli_args()
                if args:
                    config.setdefault("advanced_params", {})[stage_name] = args

        path, _ = QFileDialog.getSaveFileName(
            self, "Export Config", "mirpv_config.json", "JSON files (*.json)"
        )
        if path:
            with open(path, "w") as f:
                json.dump(config, f, indent=2)
            self._append(f"[GUI] Config exported to: {path}\n")

    def _open_report_html(self) -> None:
        if self._report_html_to_open is None:
            return
        if not self._report_html_to_open.exists():
            QMessageBox.information(self, "Not found", f"Report not found:\n{self._report_html_to_open}")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._report_html_to_open.resolve())))


def main() -> int:
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
