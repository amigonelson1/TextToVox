import os
import subprocess
from pathlib import Path
from tkinter import filedialog
from typing import Optional, Callable
import customtkinter as ctk

from src.core.config_manager import AppConfig


class ControlPanel(ctk.CTkFrame):
    """Panel de configuración acústica (velocidad, pausas) y opciones de exportación."""

    def __init__(
        self,
        master,
        config: AppConfig,
        on_config_changed: Optional[Callable[[], None]] = None,
        **kwargs
    ):
        super().__init__(master, **kwargs)
        self.config = config
        self.on_config_changed = on_config_changed

        self._build_ui()

    def _build_ui(self):
        # Título
        ctk.CTkLabel(
            self,
            text="⚙️ Ajustes de Locución y Exportación",
            font=ctk.CTkFont(size=13, weight="bold")
        ).pack(anchor="w", padx=10, pady=(4, 2))

        # 1. Velocidad
        speed_header = ctk.CTkFrame(self, fg_color="transparent")
        speed_header.pack(fill="x", padx=10, pady=(2, 0))

        ctk.CTkLabel(
            speed_header,
            text="Velocidad de habla:",
            font=ctk.CTkFont(size=12)
        ).pack(side="left")

        self.lbl_speed_val = ctk.CTkLabel(
            speed_header,
            text=f"{self.config.speed:.2f}x",
            font=ctk.CTkFont(size=12, weight="bold")
        )
        self.lbl_speed_val.pack(side="left", padx=5)

        self.btn_reset_speed = ctk.CTkButton(
            speed_header,
            text="1.0x",
            width=45,
            height=20,
            font=ctk.CTkFont(size=11),
            fg_color="gray30",
            hover_color="gray25",
            command=self._reset_speed
        )
        self.btn_reset_speed.pack(side="right")

        self.slider_speed = ctk.CTkSlider(
            self,
            from_=0.5,
            to=2.0,
            number_of_steps=30,
            command=self._on_speed_change
        )
        self.slider_speed.set(self.config.speed)
        self.slider_speed.pack(fill="x", padx=10, pady=(2, 6))

        # 2. Pausas acústicas (oraciones y párrafos)
        self.slider_sent_pause, self.lbl_sent_pause = self._add_pause_slider(
            title="Pausa entre oraciones:",
            initial_ms=self.config.sentence_pause_ms,
            from_=50,
            to=800,
            steps=15,
            command=self._on_sent_pause_change,
        )
        self.slider_para_pause, self.lbl_para_pause = self._add_pause_slider(
            title="Pausa entre párrafos:",
            initial_ms=self.config.paragraph_pause_ms,
            from_=150,
            to=2000,
            steps=37,
            command=self._on_para_pause_change,
        )

        # 3. Formato de exportación
        ctk.CTkLabel(
            self,
            text="Formato de audio a generar:",
            font=ctk.CTkFont(size=12)
        ).pack(anchor="w", padx=10, pady=(4, 2))

        self.format_seg = ctk.CTkSegmentedButton(
            self,
            values=["MP3", "WAV", "Ambos (WAV + MP3)"],
            command=self._on_format_change
        )
        self.format_seg.set(self.config.export_format)
        self.format_seg.pack(fill="x", padx=10, pady=(0, 6))

        # 4. Carpeta de destino
        dest_header = ctk.CTkFrame(self, fg_color="transparent")
        dest_header.pack(fill="x", padx=10, pady=(4, 2))

        ctk.CTkLabel(
            dest_header,
            text="Carpeta de destino:",
            font=ctk.CTkFont(size=12)
        ).pack(side="left")

        self.btn_open_out_dir = ctk.CTkButton(
            dest_header,
            text="Abrir salida",
            width=80,
            height=20,
            font=ctk.CTkFont(size=11),
            fg_color="gray30",
            hover_color="gray25",
            command=self._open_output_dir
        )
        self.btn_open_out_dir.pack(side="right")

        dir_picker_frame = ctk.CTkFrame(self, fg_color="transparent")
        dir_picker_frame.pack(fill="x", padx=10, pady=(0, 6))

        self.entry_dir = ctk.CTkEntry(dir_picker_frame, height=28)
        self.entry_dir.insert(0, str(Path(self.config.output_dir).resolve()))
        self.entry_dir.pack(side="left", fill="x", expand=True, padx=(0, 5))

        self.btn_browse = ctk.CTkButton(
            dir_picker_frame,
            text="Examinar...",
            width=85,
            height=28,
            command=self._browse_dir
        )
        self.btn_browse.pack(side="right")

    def _on_speed_change(self, val: float):
        self.config.speed = round(val, 2)
        self.lbl_speed_val.configure(text=f"{self.config.speed:.2f}x")
        self._notify()

    def _reset_speed(self):
        self.config.speed = 1.0
        self.slider_speed.set(1.0)
        self.lbl_speed_val.configure(text="1.00x")
        self._notify()

    def _add_pause_slider(
        self,
        title: str,
        initial_ms: int,
        from_: int,
        to: int,
        steps: int,
        command: Callable[[float], None],
    ):
        frame = ctk.CTkFrame(self, fg_color="transparent")
        frame.pack(fill="x", padx=10, pady=(2, 0))

        ctk.CTkLabel(
            frame,
            text=title,
            font=ctk.CTkFont(size=12)
        ).pack(side="left")

        clamped = max(from_, min(to, int(initial_ms)))
        val_lbl = ctk.CTkLabel(
            frame,
            text=f"{clamped} ms",
            font=ctk.CTkFont(size=12, weight="bold")
        )
        val_lbl.pack(side="left", padx=5)

        slider = ctk.CTkSlider(
            self,
            from_=from_,
            to=to,
            number_of_steps=steps,
            command=command
        )
        slider.set(clamped)
        slider.pack(fill="x", padx=10, pady=(1, 4))
        return slider, val_lbl

    def _on_sent_pause_change(self, val: float):
        self.config.sentence_pause_ms = int(round(val))
        self.lbl_sent_pause.configure(text=f"{self.config.sentence_pause_ms} ms")
        self._notify()

    def _on_para_pause_change(self, val: float):
        self.config.paragraph_pause_ms = int(round(val))
        self.lbl_para_pause.configure(text=f"{self.config.paragraph_pause_ms} ms")
        self._notify()

    def _on_format_change(self, val: str):
        self.config.export_format = val
        self._notify()

    def _browse_dir(self):
        chosen = filedialog.askdirectory(
            title="Seleccionar carpeta de destino",
            initialdir=self.config.output_dir
        )
        if chosen:
            self.config.output_dir = chosen
            self.entry_dir.delete(0, "end")
            self.entry_dir.insert(0, chosen)
            self._notify()

    def _open_output_dir(self):
        out_p = Path(self.config.output_dir).resolve()
        out_p.mkdir(parents=True, exist_ok=True)
        try:
            os.startfile(str(out_p))
        except Exception:
            subprocess.run(["explorer", str(out_p)])

    def get_output_dir(self) -> Path:
        raw = self.entry_dir.get().strip()
        if raw:
            return Path(raw)
        return Path("output")

    def get_export_format(self) -> str:
        return self.format_seg.get()

    def get_speed(self) -> float:
        return float(self.slider_speed.get())

    def get_sentence_pause(self) -> int:
        return int(round(self.slider_sent_pause.get()))

    def get_paragraph_pause(self) -> int:
        return int(round(self.slider_para_pause.get()))

    def _notify(self):
        if self.on_config_changed:
            self.on_config_changed()

