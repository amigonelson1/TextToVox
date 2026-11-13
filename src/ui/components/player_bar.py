import os
from pathlib import Path
from typing import Optional
import customtkinter as ctk

from src.core.audio.player import AudioPlayer


class PlayerBar(ctk.CTkFrame):
    """Barra inferior con reproductor de audio, slider interactivo y accesos directos a archivos generados."""

    def __init__(self, master, player: AudioPlayer, **kwargs):
        super().__init__(master, **kwargs)
        self.player = player
        self.player.on_finished = self._on_playback_finished

        self.last_wav_path: Optional[Path] = None
        self.last_mp3_path: Optional[Path] = None
        self._is_scrubbing: bool = False

        self._build_ui()
        self._poll_playback_status()

    def _build_ui(self):
        # Fila superior: Controles y Tiempo
        top_row = ctk.CTkFrame(self, fg_color="transparent")
        top_row.pack(fill="x", padx=10, pady=(6, 2))

        # Botón Play/Pausa
        self.btn_play = ctk.CTkButton(
            top_row,
            text="▶ Reproducir",
            width=110,
            height=32,
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="#2aa84e",
            hover_color="#20823c",
            command=self._toggle_play
        )
        self.btn_play.pack(side="left", padx=(0, 6))

        # Botón Detener
        self.btn_stop = ctk.CTkButton(
            top_row,
            text="⏹ Detener",
            width=85,
            height=32,
            fg_color="gray35",
            hover_color="gray25",
            command=self._stop
        )
        self.btn_stop.pack(side="left", padx=(0, 10))

        # Slider de posición temporal
        self.timeline_slider = ctk.CTkSlider(
            top_row,
            from_=0.0,
            to=1.0,
            command=self._on_slider_move
        )
        self.timeline_slider.set(0.0)
        self.timeline_slider.pack(side="left", fill="x", expand=True, padx=(0, 10))
        self.timeline_slider.bind("<ButtonPress-1>", self._on_slider_press)
        self.timeline_slider.bind("<ButtonRelease-1>", self._on_slider_release)

        # Etiqueta de tiempo
        self.lbl_time = ctk.CTkLabel(
            top_row,
            text="00:00 / 00:00",
            font=ctk.CTkFont(size=12, weight="bold"),
            width=90
        )
        self.lbl_time.pack(side="left", padx=(0, 10))

        # Botones para abrir los archivos exportados
        self.btn_open_wav = ctk.CTkButton(
            top_row,
            text="🎵 Abrir WAV",
            width=90,
            height=28,
            fg_color="gray30",
            hover_color="gray25",
            state="disabled",
            command=self._open_wav
        )
        self.btn_open_wav.pack(side="right", padx=(4, 0))

        self.btn_open_mp3 = ctk.CTkButton(
            top_row,
            text="🎧 Abrir MP3",
            width=90,
            height=28,
            fg_color="gray30",
            hover_color="gray25",
            state="disabled",
            command=self._open_mp3
        )
        self.btn_open_mp3.pack(side="right")

    def start_playback(self) -> bool:
        """Inicia la reproducción del audio cargado y sincroniza el botón Play/Pausa."""
        if self.player.get_duration() <= 0:
            return False
        try:
            if not self.player.is_playing():
                self.player.play()
            self.btn_play.configure(text="⏸ Pausar", fg_color="#e08214")
            return True
        except Exception:
            return False

    def _toggle_play(self):
        if self.player.is_playing():
            self.player.pause()
            self.btn_play.configure(text="▶ Continuar", fg_color="#2aa84e")
        else:
            self.player.play()
            self.btn_play.configure(text="⏸ Pausar", fg_color="#e08214")

    def _stop(self):
        self.player.stop()
        self.btn_play.configure(text="▶ Reproducir", fg_color="#2aa84e")
        self.timeline_slider.set(0.0)
        self._update_time_label(0.0, self.player.get_duration())

    def _on_slider_press(self, event):
        self._is_scrubbing = True

    def _on_slider_release(self, event):
        self._is_scrubbing = False
        duration = self.player.get_duration()
        if duration > 0:
            target_sec = self.timeline_slider.get() * duration
            self.player.seek(target_sec)

    def _on_slider_move(self, val: float):
        if self._is_scrubbing:
            duration = self.player.get_duration()
            cur_sec = val * duration
            self._update_time_label(cur_sec, duration)

    def _on_playback_finished(self):
        # Callback ejecutado en hilo del player
        self.after(0, self._handle_finished_on_ui)

    def _handle_finished_on_ui(self):
        self.btn_play.configure(text="▶ Reproducir", fg_color="#2aa84e")
        self.timeline_slider.set(0.0)
        self._update_time_label(0.0, self.player.get_duration())

    def _poll_playback_status(self):
        if self.player.is_playing() and not self._is_scrubbing:
            pos = self.player.get_position()
            duration = self.player.get_duration()
            if duration > 0:
                self.timeline_slider.set(pos / duration)
                self._update_time_label(pos, duration)
        self.after(150, self._poll_playback_status)

    def _update_time_label(self, cur_sec: float, total_sec: float):
        c_min, c_sec = divmod(int(cur_sec), 60)
        t_min, t_sec = divmod(int(total_sec), 60)
        self.lbl_time.configure(text=f"{c_min:02d}:{c_sec:02d} / {t_min:02d}:{t_sec:02d}")

    def on_new_audio_loaded(self, wav_path: Optional[Path], mp3_path: Optional[Path]):
        self.last_wav_path = wav_path
        self.last_mp3_path = mp3_path

        duration = self.player.get_duration()
        self.timeline_slider.set(0.0)
        self._update_time_label(0.0, duration)
        self.btn_play.configure(state="normal", text="▶ Reproducir", fg_color="#2aa84e")

        if wav_path and wav_path.exists():
            self.btn_open_wav.configure(state="normal")
        else:
            self.btn_open_wav.configure(state="disabled")

        if mp3_path and mp3_path.exists():
            self.btn_open_mp3.configure(state="normal")
        else:
            self.btn_open_mp3.configure(state="disabled")

    def _open_wav(self):
        if self.last_wav_path and self.last_wav_path.exists():
            try:
                os.startfile(str(self.last_wav_path.resolve()))
            except Exception:
                pass

    def _open_mp3(self):
        if self.last_mp3_path and self.last_mp3_path.exists():
            try:
                os.startfile(str(self.last_mp3_path.resolve()))
            except Exception:
                pass

