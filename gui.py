"""
Interfaz gráfica para Practica RNA - MLPs en PyTorch con FIFA 2021.
Ejecutar con: python gui.py
"""

import io
import os
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk


# ---------------------------------------------------------------------------
# Redireccionador de stdout/stderr al widget de texto
# ---------------------------------------------------------------------------
class _TextRedirector(io.TextIOBase):
    def __init__(self, widget: scrolledtext.ScrolledText) -> None:
        super().__init__()
        self._widget = widget

    def write(self, s: str) -> int:
        self._widget.after(0, self._append, s)
        return len(s)

    def _append(self, s: str) -> None:
        self._widget.configure(state="normal")
        self._widget.insert(tk.END, s)
        self._widget.see(tk.END)
        self._widget.configure(state="disabled")

    def flush(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Ventana principal
# ---------------------------------------------------------------------------
class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Practica RNA – Entrenador FIFA 2021")
        self.resizable(True, True)
        self.minsize(780, 600)
        self._build_ui()
        self._training_thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Construcción de la UI
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        # ── Frame superior: parámetros ──────────────────────────────────
        params_frame = ttk.LabelFrame(self, text="Parámetros", padding=10)
        params_frame.pack(fill="x", padx=10, pady=(10, 0))

        # CSV path
        ttk.Label(params_frame, text="Archivo CSV:").grid(
            row=0, column=0, sticky="w", padx=4, pady=3
        )
        self._csv_var = tk.StringVar(value="fifa2021_training.csv")
        ttk.Entry(params_frame, textvariable=self._csv_var, width=45).grid(
            row=0, column=1, sticky="ew", padx=4, pady=3
        )
        ttk.Button(params_frame, text="Examinar…", command=self._browse_csv).grid(
            row=0, column=2, padx=4, pady=3
        )

        # Numeros
        num_fields = [
            ("Batch size:", "batch_size", "32"),
            ("Épocas base:", "base_epochs", "500"),
            ("Épocas competencia:", "comp_epochs", "300"),
            ("Semilla:", "seed", "42"),
            ("TensorBoard log cada:", "tb_log_every", "5"),
        ]
        self._num_vars: dict[str, tk.StringVar] = {}
        for i, (label, key, default) in enumerate(num_fields, start=1):
            ttk.Label(params_frame, text=label).grid(
                row=i, column=0, sticky="w", padx=4, pady=3
            )
            var = tk.StringVar(value=default)
            self._num_vars[key] = var
            ttk.Entry(params_frame, textvariable=var, width=12).grid(
                row=i, column=1, sticky="w", padx=4, pady=3
            )

        # Device
        row_device = len(num_fields) + 1
        ttk.Label(params_frame, text="Dispositivo:").grid(
            row=row_device, column=0, sticky="w", padx=4, pady=3
        )
        self._device_var = tk.StringVar(value="auto")
        device_cb = ttk.Combobox(
            params_frame,
            textvariable=self._device_var,
            values=["auto", "cpu", "cuda"],
            state="readonly",
            width=10,
        )
        device_cb.grid(row=row_device, column=1, sticky="w", padx=4, pady=3)

        # Competencia checkbox
        self._run_comp_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            params_frame,
            text="Entrenar también modelos de competencia",
            variable=self._run_comp_var,
        ).grid(row=row_device + 1, column=0, columnspan=2, sticky="w", padx=4, pady=3)

        # Directorio runs
        ttk.Label(params_frame, text="Directorio runs:").grid(
            row=row_device + 2, column=0, sticky="w", padx=4, pady=3
        )
        self._runs_dir_var = tk.StringVar(value="runs")
        ttk.Entry(params_frame, textvariable=self._runs_dir_var, width=45).grid(
            row=row_device + 2, column=1, sticky="ew", padx=4, pady=3
        )
        ttk.Button(
            params_frame, text="Examinar…", command=self._browse_runs_dir
        ).grid(row=row_device + 2, column=2, padx=4, pady=3)

        params_frame.columnconfigure(1, weight=1)

        # ── Frame botones ───────────────────────────────────────────────
        btn_frame = ttk.Frame(self, padding=(10, 6))
        btn_frame.pack(fill="x", padx=10)

        self._start_btn = ttk.Button(
            btn_frame, text="▶  Iniciar entrenamiento", command=self._start_training
        )
        self._start_btn.pack(side="left", padx=(0, 6))

        self._stop_btn = ttk.Button(
            btn_frame,
            text="⏹  Detener",
            command=self._stop_training,
            state="disabled",
        )
        self._stop_btn.pack(side="left", padx=(0, 6))

        ttk.Button(
            btn_frame, text="🔭  Abrir TensorBoard", command=self._open_tensorboard
        ).pack(side="left", padx=(0, 6))

        ttk.Button(btn_frame, text="🗑  Limpiar log", command=self._clear_log).pack(
            side="right"
        )

        # ── Barra de progreso ───────────────────────────────────────────
        self._progress = ttk.Progressbar(self, mode="indeterminate")
        self._progress.pack(fill="x", padx=10, pady=(0, 4))

        # ── Área de log ─────────────────────────────────────────────────
        log_frame = ttk.LabelFrame(self, text="Salida del entrenamiento", padding=6)
        log_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        self._log = scrolledtext.ScrolledText(
            log_frame,
            state="disabled",
            font=("Consolas", 9),
            background="#1e1e1e",
            foreground="#d4d4d4",
            insertbackground="white",
            wrap="word",
        )
        self._log.pack(fill="both", expand=True)

        # ── Barra de estado ─────────────────────────────────────────────
        self._status_var = tk.StringVar(value="Listo.")
        ttk.Label(self, textvariable=self._status_var, anchor="w").pack(
            fill="x", padx=10, pady=(0, 4)
        )

    # ------------------------------------------------------------------
    # Helpers de navegación de archivos
    # ------------------------------------------------------------------
    def _browse_csv(self) -> None:
        path = filedialog.askopenfilename(
            title="Seleccionar CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if path:
            self._csv_var.set(path)

    def _browse_runs_dir(self) -> None:
        path = filedialog.askdirectory(title="Seleccionar directorio de runs")
        if path:
            self._runs_dir_var.set(path)

    # ------------------------------------------------------------------
    # Validación de entrada
    # ------------------------------------------------------------------
    def _validate_inputs(self) -> list[str] | None:
        """Devuelve la lista de argumentos CLI o None si hay un error."""
        errors: list[str] = []

        csv_path = self._csv_var.get().strip()
        if not os.path.isfile(csv_path):
            errors.append(f"No se encontró el CSV: {csv_path}")

        int_fields = {
            "batch_size": ("Batch size", 1, 4096),
            "base_epochs": ("Épocas base", 1, 100_000),
            "comp_epochs": ("Épocas competencia", 1, 100_000),
            "seed": ("Semilla", 0, 2**32 - 1),
            "tb_log_every": ("TensorBoard log cada", 1, 10_000),
        }
        int_values: dict[str, int] = {}
        for key, (name, lo, hi) in int_fields.items():
            raw = self._num_vars[key].get().strip()
            try:
                val = int(raw)
                if not (lo <= val <= hi):
                    errors.append(f"{name} debe estar entre {lo} y {hi}.")
                else:
                    int_values[key] = val
            except ValueError:
                errors.append(f"{name} debe ser un entero válido (recibido: '{raw}').")

        if errors:
            messagebox.showerror("Parámetros incorrectos", "\n".join(errors))
            return None

        args = [
            sys.executable, "main.py",
            "--csv-path", csv_path,
            "--batch-size", str(int_values["batch_size"]),
            "--base-epochs", str(int_values["base_epochs"]),
            "--competition-epochs", str(int_values["comp_epochs"]),
            "--seed", str(int_values["seed"]),
            "--tb-log-every", str(int_values["tb_log_every"]),
            "--device", self._device_var.get(),
            "--runs-dir", self._runs_dir_var.get().strip(),
        ]
        if self._run_comp_var.get():
            args.append("--run-competition")

        return args

    # ------------------------------------------------------------------
    # Control de entrenamiento
    # ------------------------------------------------------------------
    def _start_training(self) -> None:
        if self._training_thread and self._training_thread.is_alive():
            messagebox.showinfo("En progreso", "Ya hay un entrenamiento en curso.")
            return

        args = self._validate_inputs()
        if args is None:
            return

        self._start_btn.configure(state="disabled")
        self._stop_btn.configure(state="normal")
        self._progress.start(12)
        self._status_var.set("Entrenando…")
        self._log_line(f"$ {' '.join(args)}\n")

        self._proc: subprocess.Popen | None = None

        def run() -> None:
            try:
                self._proc = subprocess.Popen(
                    args,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    cwd=os.path.dirname(os.path.abspath(__file__)),
                )
                assert self._proc.stdout is not None
                for line in self._proc.stdout:
                    self._log_line(line)
                self._proc.wait()
                code = self._proc.returncode
                self._log_line(
                    f"\n[Proceso terminado con código {code}]\n"
                )
                self.after(0, self._on_training_done, code == 0)
            except Exception as exc:
                self._log_line(f"\n[Error: {exc}]\n")
                self.after(0, self._on_training_done, False)

        self._training_thread = threading.Thread(target=run, daemon=True)
        self._training_thread.start()

    def _stop_training(self) -> None:
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            self._log_line("\n[Entrenamiento interrumpido por el usuario]\n")
            self._on_training_done(success=False)

    def _on_training_done(self, success: bool) -> None:
        self._progress.stop()
        self._start_btn.configure(state="normal")
        self._stop_btn.configure(state="disabled")
        self._status_var.set(
            "✔ Entrenamiento completado." if success else "✘ Entrenamiento detenido/error."
        )

    # ------------------------------------------------------------------
    # TensorBoard
    # ------------------------------------------------------------------
    def _open_tensorboard(self) -> None:
        runs_dir = self._runs_dir_var.get().strip() or "runs"
        self._log_line(f"\n[Abriendo TensorBoard en: {runs_dir}]\n")
        try:
            subprocess.Popen(
                [sys.executable, "-m", "tensorboard.main", "--logdir", runs_dir],
                creationflags=subprocess.CREATE_NEW_CONSOLE
                if sys.platform == "win32"
                else 0,
            )
            self._status_var.set(
                f"TensorBoard iniciado. Abre http://localhost:6006"
            )
        except Exception as exc:
            messagebox.showerror("Error", f"No se pudo abrir TensorBoard:\n{exc}")

    # ------------------------------------------------------------------
    # Utilidades de log
    # ------------------------------------------------------------------
    def _log_line(self, text: str) -> None:
        def _do() -> None:
            self._log.configure(state="normal")
            self._log.insert(tk.END, text)
            self._log.see(tk.END)
            self._log.configure(state="disabled")

        self.after(0, _do)

    def _clear_log(self) -> None:
        self._log.configure(state="normal")
        self._log.delete("1.0", tk.END)
        self._log.configure(state="disabled")


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app = App()
    app.mainloop()

