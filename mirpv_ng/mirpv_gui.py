import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
import subprocess
import threading
import sys
import os
from pathlib import Path

# --- Configuration ---
ctk.set_appearance_mode("Dark")  # Modes: "System" (standard), "Dark", "Light"
ctk.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

class MiRPVLauncher(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Window Setup
        self.title("miRPV-NG Launcher")
        self.geometry("900x700")

        # Layout Configuration
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- Sidebar (Controls) ---
        self.sidebar_frame = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)

        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="miRPV-NG", font=ctk.CTkFont(size=24, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        self.sidebar_button_run = ctk.CTkButton(self.sidebar_frame, text="RUN REPORT", fg_color="#22c55e", hover_color="#16a34a", command=self.start_process)
        self.sidebar_button_run.grid(row=1, column=0, padx=20, pady=10)

        self.status_label = ctk.CTkLabel(self.sidebar_frame, text="Status: Ready", text_color="gray")
        self.status_label.grid(row=2, column=0, padx=20, pady=10)

        # --- Main Area ---
        self.main_frame = ctk.CTkScrollableFrame(self, label_text="Configuration")
        self.main_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")

        # 1. General Settings
        self.add_section_header("General Settings", row=0)
        self.entry_sample_id = self.add_input_field("Sample ID:", "TEST1", row=1)
        self.entry_outdir = self.add_browse_field("Output Directory:", row=2, mode="dir")

        # 2. Input Files
        self.add_section_header("Input Data", row=3)
        self.entry_final_tsv = self.add_browse_field("Final Candidates TSV:", row=4, mode="file")
        self.entry_struct_tsv = self.add_browse_field("Structure TSV (Optional):", row=5, mode="file")
        self.entry_rejects_tsv = self.add_browse_field("Rejects TSV (Optional):", row=6, mode="file")

        # 3. Parameters
        self.add_section_header("Parameters", row=7)
        self.entry_top_novel = self.add_input_field("Top Novel Count:", "10", row=8)
        self.chk_pdf = ctk.CTkCheckBox(self.main_frame, text="Generate PDF Report")
        self.chk_pdf.select()
        self.chk_pdf.grid(row=9, column=0, columnspan=2, pady=10, padx=10, sticky="w")

        # --- Console Output ---
        self.console_frame = ctk.CTkFrame(self, height=150)
        self.console_frame.grid(row=1, column=1, padx=20, pady=(0, 20), sticky="nsew")
        self.console_label = ctk.CTkLabel(self.console_frame, text="Execution Log", anchor="w")
        self.console_label.pack(fill="x", padx=10, pady=5)
        
        self.console_text = ctk.CTkTextbox(self.console_frame, font=("Consolas", 12))
        self.console_text.pack(fill="both", expand=True, padx=10, pady=(0, 10))

    # --- UI Helpers ---
    def add_section_header(self, text, row):
        label = ctk.CTkLabel(self.main_frame, text=text, font=ctk.CTkFont(size=16, weight="bold"), anchor="w")
        label.grid(row=row, column=0, columnspan=3, pady=(20, 10), padx=10, sticky="w")

    def add_input_field(self, label_text, default_val, row):
        label = ctk.CTkLabel(self.main_frame, text=label_text, anchor="w")
        label.grid(row=row, column=0, padx=10, pady=5, sticky="w")
        entry = ctk.CTkEntry(self.main_frame, width=300)
        entry.insert(0, default_val)
        entry.grid(row=row, column=1, padx=10, pady=5, sticky="ew")
        return entry

    def add_browse_field(self, label_text, row, mode="file"):
        label = ctk.CTkLabel(self.main_frame, text=label_text, anchor="w")
        label.grid(row=row, column=0, padx=10, pady=5, sticky="w")
        
        entry = ctk.CTkEntry(self.main_frame, width=300)
        entry.grid(row=row, column=1, padx=10, pady=5, sticky="ew")
        
        btn = ctk.CTkButton(self.main_frame, text="Browse", width=60, 
                            command=lambda: self.browse_action(entry, mode))
        btn.grid(row=row, column=2, padx=10, pady=5)
        return entry

    def browse_action(self, entry_widget, mode):
        if mode == "file":
            filename = filedialog.askopenfilename(filetypes=[("TSV Files", "*.tsv"), ("All Files", "*.*")])
            if filename:
                entry_widget.delete(0, "end")
                entry_widget.insert(0, filename)
        else:
            dirname = filedialog.askdirectory()
            if dirname:
                entry_widget.delete(0, "end")
                entry_widget.insert(0, dirname)

    def log(self, message):
        self.console_text.insert("end", message + "\n")
        self.console_text.see("end")

    # --- Execution Logic ---
    def start_process(self):
        # 1. Gather Inputs
        sample_id = self.entry_sample_id.get()
        outdir = self.entry_outdir.get()
        final_tsv = self.entry_final_tsv.get()
        struct_tsv = self.entry_struct_tsv.get()
        rejects_tsv = self.entry_rejects_tsv.get()
        top_novel = self.entry_top_novel.get()
        
        # 2. Validation
        if not sample_id or not outdir or not final_tsv:
            messagebox.showerror("Missing Inputs", "Please provide Sample ID, Output Dir, and Final TSV.")
            return

        # 3. Build Command
        # Assumes running from the parent folder of 'mirpv_ng'
        cmd = [
            sys.executable, "-m", "mirpv_ng.make_report",
            "--sample-id", sample_id,
            "--outdir", outdir,
            "--final-candidates-tsv", final_tsv,
            "--top-novel", top_novel
        ]

        if struct_tsv:
            cmd.extend(["--candidates-struct-tsv", struct_tsv])
        if rejects_tsv:
            cmd.extend(["--rejects-merged-tsv", rejects_tsv])
        if not self.chk_pdf.get():
            cmd.append("--no-pdf")

        # 4. Run in Thread (to avoid freezing UI)
        self.sidebar_button_run.configure(state="disabled", text="Running...")
        self.status_label.configure(text="Status: Processing...", text_color="#eab308")
        self.console_text.delete("1.0", "end")
        
        thread = threading.Thread(target=self.run_command, args=(cmd,))
        thread.start()

    def run_command(self, cmd):
        self.log(f"CMD: {' '.join(cmd)}\n" + "-"*40)
        
        try:
            # Run process and capture output in real-time
            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT, 
                text=True, 
                bufsize=1, 
                universal_newlines=True
            )

            for line in process.stdout:
                self.console_text.insert("end", line)
                self.console_text.see("end")

            process.wait()

            if process.returncode == 0:
                self.status_label.configure(text="Status: Success", text_color="#22c55e")
                messagebox.showinfo("Success", "Report generated successfully!")
            else:
                self.status_label.configure(text="Status: Failed", text_color="#ef4444")
                messagebox.showerror("Error", "The script failed. Check log for details.")

        except Exception as e:
            self.log(f"\nCRITICAL ERROR: {str(e)}")
            self.status_label.configure(text="Status: Error", text_color="#ef4444")
        
        finally:
            self.sidebar_button_run.configure(state="normal", text="RUN REPORT")

if __name__ == "__main__":
    app = MiRPVLauncher()
    app.mainloop()
