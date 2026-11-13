import datetime
import threading
from pathlib import Path
from tkinter import filedialog
from typing import Callable, List, Optional
import customtkinter as ctk
import numpy as np
import sounddevice as sd
import soundfile as sf

from src.core.audio.audio_voice_changer import AudioFileInfo, probe_audio_file
from src.core.cloning.profile_manager import ClonedVoiceProfile
from src.core.config_manager import AppConfig
from src.core.voice_manager.voice_info import VoiceInfo


class AudioConverterPanel(ctk.CTkFrame):
    """
    Panel de conversión de voz a voz (Speech-to-Speech):
    Permite cargar un audio propio o grabar desde micrófono, escuchar el original
    y transformarlo con la voz seleccionada del catálogo y el sintetizador vocal.
    """

    def __init__(
        self,
        master,
        config: AppConfig,
        tmp_dir: Path = Path("models/cloning/cache/_tmp"),
        on_play_original: Optional[Callable[[Path], None]] = None,
        on_preview_convert: Optional[Callable[[], None]] = None,
        on_full_convert: Optional[Callable[[], None]] = None,
        on_config_changed: Optional[Callable[[], None]] = None,
        **kwargs
    ):
        super().__init__(master, **kwargs)
        self.config = config
        self.tmp_dir = Path(tmp_dir)
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self.on_play_original = on_play_original
        self.on_preview_convert = on_preview_convert
        self.on_full_convert = on_full_convert
        self.on_config_changed = on_config_changed

        self.loaded_info: Optional[AudioFileInfo] = None
        self._busy: bool = False

        # Estado de grabadora de micrófono
        self._is_recording: bool = False
        self._rec_frames: List[np.ndarray] = []
        self._rec_lock = threading.Lock()
        self._rec_stream: Optional[sd.InputStream] = None
        self._rec_sr: int = 22050
        self._rec_start_time: Optional[datetime.datetime] = None

        self._build_ui()

    def _build_ui(self):
        # Encabezado principal
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=12, pady=(8, 4))

        ctk.CTkLabel(
            header,
            text="🎭 Cambiar Voz a un Audio (Voz a Voz)",
            font=ctk.CTkFont(size=16, weight="bold"),
        ).pack(anchor="w")

        ctk.CTkLabel(
            header,
            text=(
                "Carga una grabación tuya (.wav, .mp3, .ogg, .flac) o graba con tu micrófono. "
                "La aplicación conservará tus palabras, ritmo y entonación cambiándolos a la voz "
                "elegida en el Catálogo de Voces."
            ),
            font=ctk.CTkFont(size=12),
            text_color="gray65",
            justify="left",
            wraplength=560,
        ).pack(anchor="w", pady=(2, 0))

        # Paso 1: Audio de entrada
        input_card = ctk.CTkFrame(self, fg_color=("gray88", "gray18"), corner_radius=8)
        input_card.pack(fill="x", padx=12, pady=(8, 6))

        ctk.CTkLabel(
            input_card,
            text="1. Tu audio de origen",
            font=ctk.CTkFont(size=13, weight="bold"),
        ).pack(anchor="w", padx=12, pady=(10, 4))

        btn_row = ctk.CTkFrame(input_card, fg_color="transparent")
        btn_row.pack(fill="x", padx=12, pady=(2, 6))

        self.btn_load_audio = ctk.CTkButton(
            btn_row,
            text="📂 Cargar Audio...",
            height=34,
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#1f6aa5",
            hover_color="#144870",
            command=self._choose_audio_file,
        )
        self.btn_load_audio.pack(side="left", padx=(0, 8))

        self.btn_record = ctk.CTkButton(
            btn_row,
            text="🎤 Grabar Mic",
            height=34,
            width=145,
            fg_color="gray32",
            hover_color="#a32929",
            command=self._toggle_recording,
        )
        self.btn_record.pack(side="left", padx=(0, 8))

        self.btn_play_original = ctk.CTkButton(
            btn_row,
            text="▶ Escuchar Original",
            height=34,
            fg_color="#2aa84e",
            hover_color="#20823c",
            state="disabled",
            command=self._play_original,
        )
        self.btn_play_original.pack(side="left")

        self.lbl_file_info = ctk.CTkLabel(
            input_card,
            text="Ningún audio cargado aún. Sube un archivo o graba con el micrófono.",
            font=ctk.CTkFont(size=12),
            text_color="gray65",
            justify="left",
            wraplength=540,
        )
        self.lbl_file_info.pack(anchor="w", padx=12, pady=(2, 10))

        # Paso 2: Voz destino e intensidad
        target_card = ctk.CTkFrame(self, fg_color=("gray88", "gray18"), corner_radius=8)
        target_card.pack(fill="x", padx=12, pady=(4, 6))

        ctk.CTkLabel(
            target_card,
            text="2. Voz destino y naturalidad",
            font=ctk.CTkFont(size=13, weight="bold"),
        ).pack(anchor="w", padx=12, pady=(10, 4))

        self.lbl_target_voice = ctk.CTkLabel(
            target_card,
            text="🎯 Voz destino: Seleccionada en 'Catálogo de Voces'",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#4fa3e3",
            justify="left",
            wraplength=540,
        )
        self.lbl_target_voice.pack(anchor="w", padx=12, pady=(0, 2))

        ctk.CTkLabel(
            target_card,
            text=(
                "Cambia la voz en el selector 'Catálogo de Voces' de la derecha. "
                "Si deseas afinar tono grave/agudo o ecualización, combina con el Sintetizador Vocal."
            ),
            font=ctk.CTkFont(size=11),
            text_color="gray60",
            justify="left",
            wraplength=540,
        ).pack(anchor="w", padx=12, pady=(0, 8))

        tau_header = ctk.CTkFrame(target_card, fg_color="transparent")
        tau_header.pack(fill="x", padx=12, pady=(0, 2))

        ctk.CTkLabel(
            tau_header,
            text="Fuerza de conversión hacia la voz destino (tau):",
            font=ctk.CTkFont(size=12),
        ).pack(side="left")

        init_tau = max(0.15, min(0.75, float(self.config.voice_change_tau)))
        self.lbl_tau_val = ctk.CTkLabel(
            tau_header,
            text=f"{init_tau:.2f}",
            font=ctk.CTkFont(size=12, weight="bold"),
        )
        self.lbl_tau_val.pack(side="right")

        self.slider_tau = ctk.CTkSlider(
            target_card,
            from_=0.15,
            to=0.75,
            number_of_steps=12,
            command=self._on_tau_changed,
        )
        self.slider_tau.set(init_tau)
        self.slider_tau.pack(fill="x", padx=12, pady=(0, 2))

        ctk.CTkLabel(
            target_card,
            text="Menor = más natural/orgánico · Mayor = timbre más marcado (recomendado 0.40 – 0.55)",
            font=ctk.CTkFont(size=10),
            text_color="gray60",
        ).pack(anchor="w", padx=12, pady=(0, 10))

        # Paso 3: Acciones de conversión
        action_card = ctk.CTkFrame(self, fg_color=("gray88", "gray18"), corner_radius=8)
        action_card.pack(fill="x", padx=12, pady=(4, 8))

        ctk.CTkLabel(
            action_card,
            text="3. Transformar y escuchar resultado",
            font=ctk.CTkFont(size=13, weight="bold"),
        ).pack(anchor="w", padx=12, pady=(10, 6))

        conv_row = ctk.CTkFrame(action_card, fg_color="transparent")
        conv_row.pack(fill="x", padx=12, pady=(0, 8))

        self.btn_preview_convert = ctk.CTkButton(
            conv_row,
            text="🎧 Previsualizar Cambio (8 s, sin guardar)",
            height=38,
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#2a7f62",
            hover_color="#1f614b",
            state="disabled",
            command=self._trigger_preview,
        )
        self.btn_preview_convert.pack(side="left", fill="x", expand=True, padx=(0, 8))

        self.btn_full_convert = ctk.CTkButton(
            conv_row,
            text="🎭 Convertir Audio y Guardar",
            height=38,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#1f6aa5",
            hover_color="#144870",
            state="disabled",
            command=self._trigger_full_convert,
        )
        self.btn_full_convert.pack(side="right", fill="x", expand=True)

        ctk.CTkLabel(
            action_card,
            text="El resultado usa el formato (WAV/MP3) y carpeta configurados en la columna derecha.",
            font=ctk.CTkFont(size=11),
            text_color="gray60",
        ).pack(anchor="w", padx=12, pady=(0, 10))

    def _choose_audio_file(self) -> None:
        chosen = filedialog.askopenfilename(
            title="Seleccionar archivo de audio para cambiar su voz",
            filetypes=[
                ("Audio soportado", "*.wav *.mp3 *.ogg *.flac"),
                ("Todos los archivos", "*.*"),
            ],
        )
        if chosen:
            self.load_audio_file(Path(chosen))

    def load_audio_file(self, path: Path) -> bool:
        try:
            info = probe_audio_file(path)
            self.loaded_info = info
            self.lbl_file_info.configure(
                text=(
                    f"✅ {info.filename}  •  Duración: {info.duration_sec:.1f} s  •  "
                    f"{info.sample_rate} Hz  •  {'Mono' if info.channels == 1 else f'{info.channels} canales'}"
                ),
                text_color=("gray20", "gray85"),
            )
            self._sync_action_buttons()
            return True
        except Exception as exc:
            self.loaded_info = None
            self.lbl_file_info.configure(text=f"⚠️ No se pudo cargar el audio: {exc}", text_color="#e06c55")
            self._sync_action_buttons()
            return False

    def _toggle_recording(self) -> None:
        if self._is_recording:
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self) -> None:
        if self._busy or self._is_recording:
            return
        with self._rec_lock:
            self._rec_frames = []

        def _callback(indata, frames, time_info, status):
            if self._is_recording:
                with self._rec_lock:
                    self._rec_frames.append(indata[:, 0].copy())

        try:
            self._is_recording = True
            self._rec_start_time = datetime.datetime.now()
            self._rec_stream = sd.InputStream(
                samplerate=self._rec_sr,
                channels=1,
                dtype="float32",
                callback=_callback,
            )
            self._rec_stream.start()
            self.btn_load_audio.configure(state="disabled")
            self.btn_record.configure(text="⏹ Detener (00:00)", fg_color="#c93b2b", hover_color="#9e291c")
            self.lbl_file_info.configure(
                text="🔴 Grabando desde micrófono... Pulsa 'Detener' al terminar.",
                text_color="#e08214",
            )
            self._sync_action_buttons()
            self._tick_recording_timer()
        except Exception as exc:
            self._is_recording = False
            self._rec_stream = None
            self.btn_load_audio.configure(state="normal")
            self.lbl_file_info.configure(
                text=f"⚠️ No se pudo abrir el micrófono ({exc}). Carga un archivo de audio.",
                text_color="#e06c55",
            )

    def _tick_recording_timer(self) -> None:
        if not self._is_recording or not self._rec_start_time:
            return
        elapsed = int((datetime.datetime.now() - self._rec_start_time).total_seconds())
        mins, secs = divmod(elapsed, 60)
        self.btn_record.configure(text=f"⏹ Detener ({mins:02d}:{secs:02d})")
        self.after(300, self._tick_recording_timer)

    def _stop_recording(self) -> None:
        if not self._is_recording:
            return
        self._is_recording = False
        if self._rec_stream is not None:
            try:
                self._rec_stream.stop()
                self._rec_stream.close()
            except Exception:
                pass
            self._rec_stream = None

        self.btn_record.configure(text="🎤 Grabar Mic", fg_color="gray32", hover_color="#a32929")
        self.btn_load_audio.configure(state="normal")

        with self._rec_lock:
            frames = list(self._rec_frames)
            self._rec_frames = []

        if not frames:
            self.lbl_file_info.configure(text="⚠️ No se capturó audio del micrófono.", text_color="#e06c55")
            return

        samples = np.concatenate(frames)
        if len(samples) < int(0.4 * self._rec_sr):
            self.lbl_file_info.configure(text="⚠️ La grabación fue demasiado corta (< 0.4 s).", text_color="#e06c55")
            return

        peak = float(np.max(np.abs(samples)))
        if peak > 0:
            samples = np.clip(samples / max(peak, 1e-6) * 0.90, -0.98, 0.98)

        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_wav = self.tmp_dir / f"mic_{stamp}.wav"
        sf.write(str(out_wav), (samples * 32767.0).astype(np.int16), self._rec_sr, subtype="PCM_16")
        self.load_audio_file(out_wav)

    def cleanup_recorder(self) -> None:
        self._is_recording = False
        if self._rec_stream is not None:
            try:
                self._rec_stream.stop()
                self._rec_stream.close()
            except Exception:
                pass
            self._rec_stream = None

    def _on_tau_changed(self, val: float) -> None:
        tau = round(float(val), 2)
        self.config.voice_change_tau = tau
        self.lbl_tau_val.configure(text=f"{tau:.2f}")
        if self.on_config_changed:
            self.on_config_changed()

    def get_tau(self) -> float:
        return round(float(self.slider_tau.get()), 2)

    def get_loaded_audio_path(self) -> Optional[Path]:
        if self.loaded_info and self.loaded_info.path.is_file():
            return self.loaded_info.path
        return None

    def update_target_voice_label(
        self,
        voice: Optional[VoiceInfo],
        profile: Optional[ClonedVoiceProfile] = None,
    ) -> None:
        if voice is None:
            self.lbl_target_voice.configure(text="🎯 Voz destino: Ninguna seleccionada")
            return

        if voice.engine_type in ("openvoice", "xtts"):
            if profile and profile.exists:
                desc = f"🧬 Voz Clonada: '{profile.name}' (muestra personalizada)"
            else:
                desc = f"🧬 {voice.name} (sube una muestra de referencia a la derecha)"
        elif not voice.is_installed:
            desc = f"☁️ {voice.name} (requiere descarga previa en Catálogo de Voces)"
        else:
            desc = f"{voice.name} • {voice.gender} • {voice.language_name}"

        self.lbl_target_voice.configure(text=f"🎯 Voz destino actual: {desc}")

    def set_busy(self, busy: bool) -> None:
        self._busy = busy
        self.btn_load_audio.configure(state="disabled" if busy else "normal")
        self.btn_record.configure(state="disabled" if (busy or self._is_recording) else "normal")
        if not busy and self._is_recording:
            self.btn_record.configure(state="normal")
        self._sync_action_buttons()

    def _sync_action_buttons(self) -> None:
        ready = (self.loaded_info is not None) and (not self._busy) and (not self._is_recording)
        state = "normal" if ready else "disabled"
        self.btn_play_original.configure(state=state)
        self.btn_preview_convert.configure(state=state)
        self.btn_full_convert.configure(state=state)

    def _play_original(self) -> None:
        path = self.get_loaded_audio_path()
        if path and self.on_play_original:
            self.on_play_original(path)

    def _trigger_preview(self) -> None:
        if self.on_preview_convert:
            self.on_preview_convert()

    def _trigger_full_convert(self) -> None:
        if self.on_full_convert:
            self.on_full_convert()

