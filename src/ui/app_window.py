import datetime
import threading
import time
import queue
from pathlib import Path
from typing import Optional
import customtkinter as ctk
import soundfile as sf

from src.core.config_manager import AppConfig
from src.core.cloning import ClonedProfileManager, ClonedVoiceProfile
from src.core.voice_manager import VoiceCatalog, VoiceInfo
from src.core.engine import (
    PiperEngine,
    EdgeTTSEngine,
    CloneWorkerClient,
    OpenVoiceEngine,
    XTTSv2Engine,
)
from src.core.text_processing import (
    normalize_spanish_text,
    SmartTextChunker,
    build_preview_text,
)
import numpy as np

from src.core.audio import (
    AudioStitcher,
    AudioExporter,
    AudioPlayer,
    AudioVoiceChanger,
    LiveConfig,
    LiveVoiceSession,
    VoicePresetManager,
    VoiceSynthProcessor,
    VoiceSynthSettings,
    load_mono_resampled,
)
from src.ui.components.editor_panel import EditorPanel
from src.ui.components.audio_converter_panel import AudioConverterPanel
from src.ui.components.live_panel import LivePanel
from src.ui.components.voice_panel import VoicePanel
from src.ui.components.control_panel import ControlPanel
from src.ui.components.synth_panel import SynthPanel
from src.ui.components.player_bar import PlayerBar


class TexToVoxApp(ctk.CTk):
    """Ventana principal de la aplicación texToVox con arquitectura multi-hilo segura basada en Queue."""

    def __init__(self):
        super().__init__()

        # Configuración de apariencia
        self.config = AppConfig.load()
        ctk.set_appearance_mode(self.config.theme_mode)
        ctk.set_default_color_theme(self.config.color_theme)

        # Propiedades de la ventana
        self.title("texToVox - Sintetizador de Voz Neural de Alta Fidelidad")
        self.geometry("1120x760")
        self.minsize(920, 620)

        # Servicios del núcleo
        self.models_dir = Path("models")
        self.catalog = VoiceCatalog(self.models_dir)
        self.profile_manager = ClonedProfileManager(self.models_dir / "cloning" / "profiles")
        self.piper_engine = PiperEngine()
        self.edge_engine = EdgeTTSEngine()
        self.clone_worker = CloneWorkerClient()
        self.openvoice_engine = OpenVoiceEngine(self.clone_worker, self.models_dir / "cloning" / "cache")
        self.xtts_engine = XTTSv2Engine(self.clone_worker, self.models_dir / "cloning" / "cache")
        self.player = AudioPlayer()
        self.preset_manager = VoicePresetManager()
        self.chunker = SmartTextChunker(max_chars=250)
        self.clone_chunker = SmartTextChunker(max_chars=210)

        # Cola de comunicación entre hilos (Thread-safe)
        self._msg_queue = queue.Queue()

        # Control de estado de generación y último audio crudo para re-sintetizar
        self._is_generating: bool = False
        self._cancel_requested: bool = False
        self._loaded_voice_key: Optional[str] = None
        self._last_raw_audio: Optional[np.ndarray] = None
        self._last_raw_sr: int = 24000
        self._last_voice_key_used: str = "voz"
        self._last_voice_name_used: str = "Voz"
        self._last_is_preview: bool = False
        self._live_session: Optional[LiveVoiceSession] = None
        self._live_starting: bool = False

        self._build_layout()
        self._poll_msg_queue()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_layout(self):
        # 1. Barra Superior (Header)
        self.header_frame = ctk.CTkFrame(self, height=48, corner_radius=0)
        self.header_frame.pack(fill="x", side="top")

        self.logo_lbl = ctk.CTkLabel(
            self.header_frame,
            text="🔊 texToVox",
            font=ctk.CTkFont(size=20, weight="bold")
        )
        self.logo_lbl.pack(side="left", padx=(16, 8), pady=8)

        self.badge_offline = ctk.CTkLabel(
            self.header_frame,
            text="💎 Voces Nativas Elegantes (Colombia, México, Argentina) | 📁 Modelos Offline",
            font=ctk.CTkFont(size=12),
            text_color="#2aa84e"
        )
        self.badge_offline.pack(side="left", padx=4)

        # Selector de Tema (Oscuro / Claro)
        self.theme_switch = ctk.CTkSwitch(
            self.header_frame,
            text="Modo Oscuro",
            command=self._toggle_theme
        )
        if self.config.theme_mode == "dark":
            self.theme_switch.select()
        else:
            self.theme_switch.deselect()
        self.theme_switch.pack(side="right", padx=16)

        # 2. Barra Inferior (Reproductor de Audio)
        self.player_bar = PlayerBar(self, player=self.player, corner_radius=0)
        self.player_bar.pack(fill="x", side="bottom")

        # 3. Contenedor Central dividido en 2 columnas
        self.main_container = ctk.CTkFrame(self, fg_color="transparent")
        self.main_container.pack(fill="both", expand=True, padx=12, pady=10)

        # Columna Izquierda: Pestañas principales (Texto a Voz / Cambiar Voz a Audio / En Vivo)
        self.main_tabs = ctk.CTkTabview(self.main_container, anchor="nw")
        self.main_tabs.pack(side="left", fill="both", expand=True, padx=(0, 6))

        tab_tts = self.main_tabs.add("📝 Texto a Voz")
        tab_vc = self.main_tabs.add("🎭 Cambiar Voz a Audio")
        tab_live = self.main_tabs.add("🎙️ En Vivo (Reuniones / Juegos)")

        self.editor_panel = EditorPanel(tab_tts)
        self.editor_panel.pack(fill="both", expand=True)

        self.audio_converter_panel = AudioConverterPanel(
            tab_vc,
            config=self.config,
            tmp_dir=self.models_dir / "cloning" / "cache" / "_tmp",
            on_play_original=self._on_preview_reference,
            on_preview_convert=lambda: self._start_audio_conversion(is_preview=True),
            on_full_convert=lambda: self._start_audio_conversion(is_preview=False),
            on_config_changed=self._on_config_changed,
        )
        self.audio_converter_panel.pack(fill="both", expand=True)

        self.live_panel = LivePanel(
            tab_live,
            config=self.config,
            on_start_requested=self._start_live_session,
            on_stop_requested=self._stop_live_session,
            on_record_toggle=self._toggle_live_recording,
            on_recalibrate_requested=self._recalibrate_live_mic,
            on_runtime_changed=self._sync_live_runtime,
            on_config_changed=self._on_config_changed,
        )
        self.live_panel.pack(fill="both", expand=True)

        # Columna Derecha: Controles y Voces (Ancho fijo)
        self.sidebar_frame = ctk.CTkScrollableFrame(self.main_container, width=430)
        self.sidebar_frame.pack(side="right", fill="both", padx=(6, 0))

        # Panel de Voces
        voices = self.catalog.get_all_voices()
        self.voice_panel = VoicePanel(
            self.sidebar_frame,
            voices=voices,
            config=self.config,
            profile_manager=self.profile_manager,
            on_voice_changed=self._on_voice_changed,
            on_download_requested=self._on_download_requested,
            on_preview_reference=self._on_preview_reference,
            on_preview_voice=self._start_preview,
            on_config_changed=self._on_config_changed,
            models_dir=self.models_dir
        )
        self.voice_panel.pack(fill="x", pady=(0, 4))
        self.audio_converter_panel.update_target_voice_label(
            self.voice_panel.current_voice,
            self.voice_panel.get_selected_profile(),
        )
        self.live_panel.update_target_voice_label(
            self.voice_panel.current_voice,
            self.voice_panel.get_selected_profile(),
        )

        # Panel de Controles Acústicos y Exportación (Pegado inmediatamente al selector)
        self.control_panel = ControlPanel(
            self.sidebar_frame,
            config=self.config,
            on_config_changed=self._on_config_changed
        )
        self.control_panel.pack(fill="x", pady=(0, 6))

        # Panel del Sintetizador Vocal (Tono, Formantes, Expresividad, EQ y Preajustes)
        self.synth_panel = SynthPanel(
            self.sidebar_frame,
            config=self.config,
            preset_manager=self.preset_manager,
            on_config_changed=self._on_config_changed,
            on_reapply_requested=self._reapply_synth_to_last_audio
        )
        self.synth_panel.pack(fill="x", pady=(0, 6))

        # Panel de Acción / Generación
        self.action_card = ctk.CTkFrame(self.sidebar_frame, fg_color=("gray85", "gray20"))
        self.action_card.pack(fill="x", pady=(0, 8))

        self.action_btns_frame = ctk.CTkFrame(self.action_card, fg_color="transparent")
        self.action_btns_frame.pack(fill="x", padx=10, pady=(10, 5))

        self.btn_preview = ctk.CTkButton(
            self.action_btns_frame,
            text="🎧 Previsualizar Voz (Sin guardar)",
            font=ctk.CTkFont(size=13, weight="bold"),
            height=34,
            fg_color="#2a7f62",
            hover_color="#1f614b",
            command=self._start_preview
        )
        self.btn_preview.pack(fill="x", pady=(0, 5))

        self.btn_generate = ctk.CTkButton(
            self.action_btns_frame,
            text="⚡ Generar Audio",
            font=ctk.CTkFont(size=15, weight="bold"),
            height=42,
            fg_color="#1f6aa5",
            hover_color="#144870",
            command=self._start_generation
        )
        self.btn_generate.pack(fill="x")

        self.btn_cancel = ctk.CTkButton(
            self.action_card,
            text="✖ Cancelar Generación",
            height=28,
            fg_color="#c93b2b",
            hover_color="#9e291c",
            command=self._cancel_generation
        )
        # Inicialmente oculto

        self.progress_bar = ctk.CTkProgressBar(self.action_card)
        self.progress_bar.set(0)
        self.progress_bar.pack(fill="x", padx=10, pady=(5, 2))

        self.lbl_status = ctk.CTkLabel(
            self.action_card,
            text="Listo para generar.",
            font=ctk.CTkFont(size=12),
            text_color="gray60",
            wraplength=390,
            justify="center",
        )
        self.lbl_status.pack(padx=10, pady=(0, 10))

    def _poll_msg_queue(self):
        """Consume eventos de la cola en el hilo principal de la interfaz gráfica."""
        try:
            while True:
                msg_type, args = self._msg_queue.get_nowait()
                if msg_type == "status":
                    percent, text_msg = args
                    self._apply_status(percent, text_msg)
                elif msg_type == "completed":
                    wav_out, mp3_out, text_msg, raw_audio, sample_rate, voice_key, voice_name, is_preview, auto_play = args
                    if raw_audio is not None:
                        self._last_raw_audio = raw_audio
                        self._last_raw_sr = sample_rate
                        self._last_voice_key_used = voice_key
                        self._last_voice_name_used = voice_name
                        self._last_is_preview = bool(is_preview)
                    self.player_bar.on_new_audio_loaded(wav_out, mp3_out)
                    self._finish_generation_ui(success=True, msg=text_msg)
                    if auto_play:
                        self.player_bar.start_playback()
                elif msg_type == "error":
                    text_msg = args
                    self._finish_generation_ui(success=False, msg=text_msg)
                elif msg_type == "download_progress":
                    percent, text_msg = args
                    self.voice_panel.update_download_progress(percent, text_msg)
                elif msg_type == "refresh_voices":
                    select_key = args
                    voices = self.catalog.get_all_voices()
                    self.voice_panel.refresh_voices(voices, select_key=select_key)
                    self.audio_converter_panel.update_target_voice_label(
                        self.voice_panel.current_voice,
                        self.voice_panel.get_selected_profile(),
                    )
                    self.live_panel.update_target_voice_label(
                        self.voice_panel.current_voice,
                        self.voice_panel.get_selected_profile(),
                    )
                elif msg_type == "live_telemetry":
                    self.live_panel.apply_telemetry(args)
                elif msg_type == "live_started":
                    self._live_starting = False
                    self.live_panel.set_running_state(True)
                    self.audio_converter_panel.set_busy(True)
                    self.lbl_status.configure(text=args)
                elif msg_type == "live_failed":
                    self._live_starting = False
                    self._live_session = None
                    self.live_panel.set_running_state(False)
                    self.audio_converter_panel.set_busy(self._is_generating)
                    self.lbl_status.configure(text=f"⚠️ {args}")
        except queue.Empty:
            pass

        self.after(40, self._poll_msg_queue)

    def _on_voice_changed(self, voice: VoiceInfo):
        self.config.last_voice_key = voice.key
        self.config.save()
        profile = self.voice_panel.get_selected_profile()
        self.audio_converter_panel.update_target_voice_label(voice, profile)
        if hasattr(self, "live_panel"):
            self.live_panel.update_target_voice_label(voice, profile)
        if self._live_session and self._live_session.is_running and self._live_session.config.mode == "neural":
            self._switch_live_target_async()

    def _on_config_changed(self):
        self.config.save()
        if hasattr(self, "audio_converter_panel") and hasattr(self, "voice_panel"):
            profile = self.voice_panel.get_selected_profile()
            self.audio_converter_panel.update_target_voice_label(self.voice_panel.current_voice, profile)
            if hasattr(self, "live_panel"):
                self.live_panel.update_target_voice_label(self.voice_panel.current_voice, profile)
        self._sync_live_runtime()
        if self._live_session and self._live_session.is_running and self._live_session.config.mode == "neural":
            self._switch_live_target_async()

    def _on_preview_reference(self, audio_path: Path):
        """Reproduce en el reproductor integrado la muestra de referencia o audio cargado."""
        try:
            data, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
            if data.ndim > 1:
                data = np.mean(data, axis=1)
            pcm = np.clip(data * 32767.0, -32768.0, 32767.0).astype(np.int16)
            self.player.stop()
            self.player.load(pcm, int(sr))
            self.player_bar.on_new_audio_loaded(audio_path if audio_path.suffix.lower() == ".wav" else None, None)
            self.player_bar.start_playback()
            self.lbl_status.configure(text=f"🔊 Reproduciendo audio: {audio_path.name}")
        except Exception as exc:
            self.lbl_status.configure(text=f"⚠️ No se pudo reproducir el audio: {exc}")

    def _on_download_requested(self, voice: VoiceInfo):
        def _worker():
            try:
                def _prog(percent, msg):
                    self._msg_queue.put(("download_progress", (percent, msg)))
                self.catalog.download_voice(voice, progress_callback=_prog)
                self._msg_queue.put(("refresh_voices", voice.key))
            except Exception as e:
                err_msg = f"Error en descarga: {e}"
                self._msg_queue.put(("download_progress", (0, err_msg)))

        threading.Thread(target=_worker, daemon=True).start()

    def _validate_voice_and_cloning(
        self,
    ) -> Optional[tuple[VoiceInfo, Optional[ClonedVoiceProfile], Optional[VoiceInfo], float]]:
        selected_voice = self.voice_panel.current_voice
        if not selected_voice or not selected_voice.is_installed:
            self.lbl_status.configure(text="⚠️ Descarga primero la voz seleccionada.")
            return None

        selected_profile: Optional[ClonedVoiceProfile] = None
        selected_base_voice: Optional[VoiceInfo] = None
        openvoice_tau = 0.35

        if selected_voice.engine_type in ("openvoice", "xtts"):
            if not CloneWorkerClient.is_environment_ready():
                self.lbl_status.configure(
                    text="⚠️ Falta el entorno de clonación (.venv_clone). Ejecuta instalar_clonacion.bat."
                )
                return None
            selected_profile = self.voice_panel.get_selected_profile()
            if not selected_profile or not selected_profile.exists:
                self.lbl_status.configure(
                    text="⚠️ Sube primero un audio de referencia (6-20 s) para clonar."
                )
                return None
            if selected_voice.engine_type == "openvoice":
                selected_base_voice = self.voice_panel.get_selected_base_voice()
                if not selected_base_voice:
                    self.lbl_status.configure(text="⚠️ No hay una voz base latina disponible.")
                    return None
                openvoice_tau = self.voice_panel.get_openvoice_tau()

        return selected_voice, selected_profile, selected_base_voice, openvoice_tau

    def _start_preview(self):
        """Sintetiza un fragmento corto en memoria con velocidad, pausas y sintetizador actuales sin guardar archivos."""
        if self._is_generating:
            return
        if self._is_live_neural_busy():
            self.lbl_status.configure(text="⏹ Detén primero 'En Vivo (Neural)' antes de usar el sintetizador pesado.")
            return

        validated = self._validate_voice_and_cloning()
        if not validated:
            return
        selected_voice, selected_profile, selected_base_voice, openvoice_tau = validated

        preview_text = build_preview_text(self.editor_panel.get_text())
        self._launch_synthesis(
            raw_text=preview_text,
            selected_voice=selected_voice,
            selected_profile=selected_profile,
            selected_base_voice=selected_base_voice,
            openvoice_tau=openvoice_tau,
            is_preview=True,
        )

    def _start_generation(self):
        if self._is_generating:
            return
        if self._is_live_neural_busy():
            self.lbl_status.configure(text="⏹ Detén primero 'En Vivo (Neural)' antes de sintetizar audio largo.")
            return

        text = self.editor_panel.get_text().strip()
        if not text:
            self.lbl_status.configure(text="⚠️ Escribe o pega texto antes de generar.")
            return

        validated = self._validate_voice_and_cloning()
        if not validated:
            return
        selected_voice, selected_profile, selected_base_voice, openvoice_tau = validated

        self._launch_synthesis(
            raw_text=text,
            selected_voice=selected_voice,
            selected_profile=selected_profile,
            selected_base_voice=selected_base_voice,
            openvoice_tau=openvoice_tau,
            is_preview=False,
        )

    def _enter_busy_ui(
        self,
        cancel_text: str,
        status_msg: str,
        swap_cancel: bool = True,
        progress: float = 0.0,
    ) -> None:
        self._is_generating = True
        self._cancel_requested = False
        self.synth_panel.set_reapply_enabled(False)
        self.voice_panel.set_preview_enabled(False)
        self.audio_converter_panel.set_busy(True)
        self.live_panel.set_busy_external(True)
        self.btn_preview.configure(state="disabled")
        self.btn_generate.configure(state="disabled")

        if swap_cancel:
            self.action_btns_frame.pack_forget()
            self.btn_cancel.configure(text=cancel_text)
            self.btn_cancel.pack(fill="x", padx=10, pady=(10, 5), before=self.progress_bar)

        self.progress_bar.set(progress)
        self.lbl_status.configure(text=status_msg)

    def _launch_synthesis(
        self,
        raw_text: str,
        selected_voice: VoiceInfo,
        selected_profile: Optional[ClonedVoiceProfile],
        selected_base_voice: Optional[VoiceInfo],
        openvoice_tau: float,
        is_preview: bool,
    ):
        self._enter_busy_ui(
            cancel_text="✖ Cancelar Previsualización" if is_preview else "✖ Cancelar Generación",
            status_msg="Preparando previsualización (sin guardar)..." if is_preview else "Iniciando síntesis...",
            swap_cancel=True,
            progress=0.0,
        )

        # Capturar parámetros de la interfaz en el hilo principal
        speed = self.control_panel.get_speed()
        sent_pause = self.control_panel.get_sentence_pause()
        para_pause = self.control_panel.get_paragraph_pause()
        out_dir = self.control_panel.get_output_dir()
        fmt = self.control_panel.get_export_format()
        synth_settings = self.synth_panel.get_settings()

        worker_thread = threading.Thread(
            target=self._generation_worker,
            args=(
                raw_text,
                selected_voice,
                speed,
                sent_pause,
                para_pause,
                out_dir,
                fmt,
                selected_profile,
                selected_base_voice,
                openvoice_tau,
                synth_settings,
                is_preview,
            ),
            daemon=True
        )
        worker_thread.start()

    def _resolve_openvoice_conversion_target(
        self,
    ) -> Optional[tuple[Path, VoiceInfo, Optional[Path], str, str]]:
        """Valida entorno OpenVoice v2 y devuelve (ov_model_dir, target_voice, profile_audio, voice_key, display)."""
        if not CloneWorkerClient.is_environment_ready():
            self.lbl_status.configure(
                text="⚠️ Falta el entorno de conversión (.venv_clone). Ejecuta instalar_clonacion.bat."
            )
            return None

        ov_info = next((v for v in self.catalog.get_all_voices() if v.key == "clone-openvoice-v2"), None)
        if not ov_info or not ov_info.is_installed or not ov_info.model_path:
            self.lbl_status.configure(
                text="⚠️ Descarga primero 'Clon Híbrido LatAm (OpenVoice v2)' en el Catálogo de Voces."
            )
            return None

        target_voice = self.voice_panel.current_voice
        if not target_voice:
            self.lbl_status.configure(text="⚠️ Selecciona una voz destino en el Catálogo de Voces.")
            return None

        target_profile_audio: Optional[Path] = None
        if target_voice.engine_type in ("openvoice", "xtts"):
            profile = self.voice_panel.get_selected_profile()
            if not profile or not profile.exists:
                self.lbl_status.configure(
                    text="⚠️ Sube primero una muestra de referencia para usar esa voz clonada como destino."
                )
                return None
            target_profile_audio = Path(profile.audio_path)
            voice_display = f"{target_voice.name} - {profile.name}"
            voice_key = f"vc_{profile.profile_id}"
        else:
            if not target_voice.is_installed:
                self.lbl_status.configure(text="⚠️ Descarga primero la voz destino seleccionada.")
                return None
            voice_display = target_voice.name
            voice_key = f"vc_{target_voice.key}"

        return ov_info.model_path, target_voice, target_profile_audio, voice_key, voice_display

    def _start_audio_conversion(self, is_preview: bool = False):
        """Convierte el audio cargado por el usuario al timbre de la voz destino elegida."""
        if self._is_generating:
            return
        if self._live_session and self._live_session.is_running:
            self.lbl_status.configure(text="⏹ Detén primero la sesión 'En Vivo' para convertir archivos.")
            return

        source_audio = self.audio_converter_panel.get_loaded_audio_path()
        if source_audio is None:
            self.lbl_status.configure(text="⚠️ Carga primero un audio o graba desde el micrófono.")
            return

        resolved = self._resolve_openvoice_conversion_target()
        if not resolved:
            return
        ov_model_dir, target_voice, target_profile_audio, voice_key, voice_display = resolved

        self._enter_busy_ui(
            cancel_text="✖ Cancelar Vista Previa" if is_preview else "✖ Cancelar Conversión",
            status_msg=(
                f"Previsualizando cambio de voz ({voice_display})..."
                if is_preview
                else f"Convirtiendo audio hacia {voice_display}..."
            ),
            swap_cancel=True,
            progress=0.04,
        )

        tau = self.audio_converter_panel.get_tau()
        out_dir = self.control_panel.get_output_dir()
        fmt = self.control_panel.get_export_format()
        synth_settings = self.synth_panel.get_settings()

        threading.Thread(
            target=self._audio_conversion_worker,
            args=(
                source_audio,
                ov_model_dir,
                target_voice,
                target_profile_audio,
                voice_key,
                voice_display,
                tau,
                out_dir,
                fmt,
                synth_settings,
                is_preview,
            ),
            daemon=True,
        ).start()

    def _cancel_generation(self):
        if self._is_generating:
            self._cancel_requested = True
            self.clone_worker.abort_current()
            self.lbl_status.configure(text="Cancelando generación...")

    def _generation_worker(
        self,
        raw_text: str,
        voice: VoiceInfo,
        speed: float,
        sent_pause: int,
        para_pause: int,
        out_dir: Path,
        fmt: str,
        cloned_profile: Optional[ClonedVoiceProfile] = None,
        base_voice: Optional[VoiceInfo] = None,
        openvoice_tau: float = 0.35,
        synth_settings: Optional[VoiceSynthSettings] = None,
        is_preview: bool = False,
    ):
        t_start = time.time()
        try:
            # 1. Seleccionar motor según el tipo de voz
            if voice.engine_type == "edge":
                engine = self.edge_engine
                self.edge_engine.set_voice(voice.voice_identifier)
                self._msg_queue.put(("status", (0.05, f"Conectando con voz nativa ({voice.name})...")))
            elif voice.engine_type == "openvoice":
                engine = self.openvoice_engine
                self._msg_queue.put(("status", (0.05, "Cargando OpenVoice v2 en CPU (primera vez puede tardar unos segundos)...")))
                if not self.openvoice_engine.is_loaded():
                    self.openvoice_engine.load_model(voice.model_path, voice.config_path)
                assert cloned_profile is not None and base_voice is not None
                self.openvoice_engine.configure_cloning(
                    base_voice=base_voice,
                    reference_audio=Path(cloned_profile.audio_path),
                    tau=openvoice_tau,
                )
            elif voice.engine_type == "xtts":
                engine = self.xtts_engine
                self._msg_queue.put(("status", (0.05, "Cargando Coqui XTTS-v2 en CPU (puede tardar 10-30 s)...")))
                if not self.xtts_engine.is_loaded():
                    self.xtts_engine.load_model(voice.model_path, voice.config_path)
                assert cloned_profile is not None
                self.xtts_engine.set_reference_audio(Path(cloned_profile.audio_path), language="es")
            else:
                engine = self.piper_engine
                if self._loaded_voice_key != voice.key or not self.piper_engine.is_loaded():
                    self._msg_queue.put(("status", (0.05, f"Cargando motor Piper ({voice.name})...")))
                    self.piper_engine.load_model(voice.model_path, voice.config_path)
                    self._loaded_voice_key = voice.key

            if self._cancel_requested:
                self._msg_queue.put(("error", "Generación cancelada."))
                return

            # 2. Normalizar y fragmentar
            self._msg_queue.put(("status", (0.10, "Normalizando texto y entonación...")))
            normalized_text = normalize_spanish_text(raw_text)

            active_chunker = self.clone_chunker if voice.engine_type == "xtts" else self.chunker
            chunks = active_chunker.split(normalized_text)
            total_chunks = len(chunks)
            if total_chunks == 0:
                self._msg_queue.put(("error", "El texto no contiene oraciones válidas."))
                return

            chunk_results = []
            for i, chunk in enumerate(chunks):
                if self._cancel_requested:
                    self._msg_queue.put(("error", "Generación cancelada."))
                    return

                pct = 0.10 + (i / total_chunks) * 0.75
                self._msg_queue.put(("status", (pct, f"Sintetizando fragmento {i + 1} de {total_chunks}...")))

                audio_chunks = engine.synthesize(
                    text=chunk.text,
                    speed=speed
                )
                chunk_results.append((audio_chunks, chunk.is_paragraph_end))

            if self._cancel_requested:
                self._msg_queue.put(("error", "Generación cancelada."))
                return

            # 3. Concatenar y dar pausas acústicas
            self._msg_queue.put(("status", (0.88, "Ensamblando audio con silencios naturales...")))
            stitcher = AudioStitcher(
                sentence_pause_ms=sent_pause,
                paragraph_pause_ms=para_pause,
                normalize_peaks=True
            )
            sample_rate = engine.get_sample_rate()
            raw_audio = stitcher.stitch(chunk_results, sample_rate)

            # 4. Aplicar sintetizador vocal, exportar (solo si no es previsualización) y cargar en reproductor
            active_synth = synth_settings or VoiceSynthSettings()
            _, wav_out, mp3_out, duration_sec = self._process_export_and_load(
                raw_audio=raw_audio,
                sample_rate=sample_rate,
                voice_key=voice.key,
                out_dir=out_dir,
                fmt=fmt,
                synth_settings=active_synth,
                save_to_disk=not is_preview,
            )

            total_time = time.time() - t_start
            voice_display = f"{voice.name} - {cloned_profile.name}" if cloned_profile else voice.name
            preset_note = f" [{active_synth.preset_name}]" if not active_synth.is_neutral() else ""
            if is_preview:
                msg = (
                    f"🎧 Vista previa ({duration_sec:.1f} s • {voice_display}{preset_note} • "
                    f"{speed:.2f}x • pausa párrafos {para_pause} ms) sin guardar en disco."
                )
            else:
                msg = f"¡Éxito! Generados {duration_sec:.1f} s de audio en {total_time:.2f} s ({voice_display}{preset_note})."

            self._msg_queue.put(
                ("completed", (wav_out, mp3_out, msg, raw_audio, sample_rate, voice.key, voice_display, is_preview, is_preview))
            )

        except Exception as e:
            err = f"Error durante la generación: {e}"
            self._msg_queue.put(("error", err))

    def _process_export_and_load(
        self,
        raw_audio: np.ndarray,
        sample_rate: int,
        voice_key: str,
        out_dir: Path,
        fmt: str,
        synth_settings: VoiceSynthSettings,
        save_to_disk: bool = True,
    ) -> tuple[np.ndarray, Optional[Path], Optional[Path], float]:
        """Aplica el sintetizador vocal, exporta WAV/MP3 (si save_to_disk) y carga en el reproductor."""
        if not synth_settings.is_neutral():
            self._msg_queue.put(("status", (0.91, f"Aplicando sintetizador ({synth_settings.preset_name})...")))
        final_audio = VoiceSynthProcessor.process(raw_audio, sample_rate, synth_settings)

        wav_out = None
        mp3_out = None

        if save_to_disk:
            self._msg_queue.put(("status", (0.95, "Guardando archivos de audio...")))
            out_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = f"locucion_{voice_key}_{timestamp}"
            if (out_dir / f"{base_filename}.wav").exists() or (out_dir / f"{base_filename}.mp3").exists():
                millis = int((time.time() % 1) * 1000)
                base_filename = f"{base_filename}_{millis:03d}"

            if "WAV" in fmt or "Ambos" in fmt:
                wav_out = out_dir / f"{base_filename}.wav"
                AudioExporter.export_wav(final_audio, sample_rate, wav_out)

            if "MP3" in fmt or "Ambos" in fmt:
                mp3_out = out_dir / f"{base_filename}.mp3"
                AudioExporter.export_mp3(final_audio, sample_rate, mp3_out)

        self.player.stop()
        self.player.load(final_audio, sample_rate)
        duration_sec = len(final_audio) / sample_rate
        return final_audio, wav_out, mp3_out, duration_sec

    def _audio_conversion_worker(
        self,
        source_audio: Path,
        ov_model_dir: Path,
        target_voice: VoiceInfo,
        target_profile_audio: Optional[Path],
        voice_key: str,
        voice_display: str,
        tau: float,
        out_dir: Path,
        fmt: str,
        synth_settings: VoiceSynthSettings,
        is_preview: bool = False,
    ):
        t_start = time.time()
        try:
            raw_audio, sample_rate = AudioVoiceChanger.convert_file(
                source_audio=source_audio,
                openvoice_engine=self.openvoice_engine,
                openvoice_model_dir=ov_model_dir,
                target_voice=target_voice if target_profile_audio is None else None,
                target_profile_audio=target_profile_audio,
                tau=tau,
                max_seconds=8.5 if is_preview else None,
                progress_callback=lambda p, m: self._msg_queue.put(("status", (p, m))),
                cancel_check=lambda: self._cancel_requested,
            )

            if self._cancel_requested:
                self._msg_queue.put(("error", "Conversión cancelada."))
                return

            _, wav_out, mp3_out, duration_sec = self._process_export_and_load(
                raw_audio=raw_audio,
                sample_rate=sample_rate,
                voice_key=voice_key,
                out_dir=out_dir,
                fmt=fmt,
                synth_settings=synth_settings,
                save_to_disk=not is_preview,
            )

            dt = time.time() - t_start
            preset_note = f" [{synth_settings.preset_name}]" if not synth_settings.is_neutral() else ""
            if is_preview:
                msg = f"🎧 Vista previa voz cambiada ({duration_sec:.1f} s → {voice_display}{preset_note}) sin guardar en disco."
            else:
                msg = f"🎭 Audio transformado a '{voice_display}'{preset_note} ({duration_sec:.1f} s en {dt:.2f} s)."

            self._msg_queue.put(
                ("completed", (wav_out, mp3_out, msg, raw_audio, sample_rate, voice_key, voice_display, is_preview, is_preview))
            )
        except InterruptedError:
            self._msg_queue.put(("error", "Conversión cancelada."))
        except Exception as exc:
            self._msg_queue.put(("error", f"Error al cambiar la voz del audio: {exc}"))

    def _reapply_synth_to_last_audio(self):
        """Reaplica el sintetizador vocal al último audio crudo sin regenerar el TTS neuronal."""
        if self._is_generating or self._last_raw_audio is None:
            return

        self._enter_busy_ui(
            cancel_text="✖ Cancelar",
            status_msg="Reaplicando sintetizador vocal...",
            swap_cancel=False,
            progress=0.45,
        )

        raw_audio = self._last_raw_audio
        sample_rate = self._last_raw_sr
        voice_key = self._last_voice_key_used
        voice_name = self._last_voice_name_used
        is_preview = self._last_is_preview
        out_dir = self.control_panel.get_output_dir()
        fmt = self.control_panel.get_export_format()
        synth_settings = self.synth_panel.get_settings()

        threading.Thread(
            target=self._reapply_synth_worker,
            args=(raw_audio, sample_rate, voice_key, voice_name, out_dir, fmt, synth_settings, is_preview),
            daemon=True,
        ).start()

    def _reapply_synth_worker(
        self,
        raw_audio: np.ndarray,
        sample_rate: int,
        voice_key: str,
        voice_name: str,
        out_dir: Path,
        fmt: str,
        synth_settings: VoiceSynthSettings,
        is_preview: bool = False,
    ):
        t_start = time.time()
        try:
            _, wav_out, mp3_out, duration_sec = self._process_export_and_load(
                raw_audio=raw_audio,
                sample_rate=sample_rate,
                voice_key=voice_key,
                out_dir=out_dir,
                fmt=fmt,
                synth_settings=synth_settings,
                save_to_disk=not is_preview,
            )
            dt = time.time() - t_start
            suffix = " (vista previa sin guardar)" if is_preview else ""
            msg = f"🎛️ '{synth_settings.preset_name}' aplicado a {duration_sec:.1f} s ({voice_name}) en {dt:.2f} s{suffix}."
            self._msg_queue.put(("completed", (wav_out, mp3_out, msg, None, sample_rate, voice_key, voice_name, is_preview, is_preview)))
        except Exception as exc:
            self._msg_queue.put(("error", f"Error aplicando sintetizador: {exc}"))

    def _apply_status(self, percent: float, msg: str):
        self.progress_bar.set(percent)
        self.lbl_status.configure(text=msg)

    def _finish_generation_ui(self, success: bool, msg: str):
        self._is_generating = False
        self.btn_cancel.pack_forget()
        if not self.action_btns_frame.winfo_ismapped():
            self.action_btns_frame.pack(fill="x", padx=10, pady=(10, 5), before=self.progress_bar)
        self.btn_preview.configure(state="normal")
        self.btn_generate.configure(state="normal")
        self.voice_panel.set_preview_enabled(True)
        self.audio_converter_panel.set_busy(bool(self._live_session and self._live_session.is_running))
        self.live_panel.set_busy_external(False)
        self.synth_panel.set_reapply_enabled(self._last_raw_audio is not None)
        self.progress_bar.set(1.0 if success else 0.0)
        self.lbl_status.configure(text=msg)

    def _is_live_neural_busy(self) -> bool:
        return bool(
            (self._live_starting and self.live_panel.get_mode() == "neural")
            or (
                self._live_session is not None
                and self._live_session.is_running
                and self._live_session.config.mode == "neural"
            )
        )

    def _start_live_session(self, _ignored_cfg: LiveConfig) -> None:
        if self._live_starting or (self._live_session and self._live_session.is_running):
            return
        if self._is_generating:
            self.lbl_status.configure(text="⏳ Espera a que termine o cancela la generación actual antes de activar En Vivo.")
            return

        synth_settings = self.synth_panel.get_settings()
        cfg = self.live_panel.build_live_config(synth_settings=synth_settings)
        if cfg.input_device is None or cfg.output_device is None:
            self.lbl_status.configure(text="⚠️ Selecciona un micrófono de entrada y un dispositivo de salida válidos.")
            return

        if cfg.mode == "dsp":
            self._live_starting = True
            self.live_panel.set_preparing("⚡ Iniciando cambiador de voz ultrarrápido...")

            def _dsp_worker():
                try:
                    session = LiveVoiceSession(
                        config=cfg,
                        openvoice_engine=None,
                        on_telemetry=lambda tel: self._msg_queue.put(("live_telemetry", tel)),
                    )
                    if not self._live_starting:
                        return
                    session.start()
                    if not self._live_starting:
                        session.stop()
                        return
                    self._live_session = session
                    self._msg_queue.put((
                        "live_started",
                        f"🎙️ En Vivo ACTIVO (⚡ Ultrarrápido → {cfg.output_device.name}).",
                    ))
                except Exception as exc:
                    self._msg_queue.put(("live_failed", f"Error al iniciar audio en vivo: {exc}"))

            threading.Thread(target=_dsp_worker, daemon=True, name="LiveDSPStarter").start()
            return

        resolved = self._resolve_openvoice_conversion_target()
        if not resolved:
            return
        ov_model_dir, target_voice, target_profile_audio, _, voice_display = resolved
        cfg.target_label = voice_display

        self._live_starting = True
        self.live_panel.set_preparing(f"🧠 Cargando OpenVoice v2 y timbre destino ({voice_display})...")
        self.lbl_status.configure(text=f"🧠 Preparando voz en vivo hacia '{voice_display}'...")

        def _neural_worker():
            try:
                if not self.openvoice_engine.is_loaded():
                    self.openvoice_engine.load_model(ov_model_dir)
                if target_profile_audio is not None:
                    tgt_se = self.openvoice_engine.ensure_audio_embedding(
                        target_profile_audio, prefix="ref", min_duration_sec=2.0
                    )
                else:
                    tgt_se = self.openvoice_engine.ensure_catalog_voice_embedding(target_voice)
                cfg.target_se_path = tgt_se

                session = LiveVoiceSession(
                    config=cfg,
                    openvoice_engine=self.openvoice_engine,
                    on_telemetry=lambda tel: self._msg_queue.put(("live_telemetry", tel)),
                )
                if not self._live_starting:
                    return
                session.start()
                if not self._live_starting:
                    session.stop()
                    return
                self._live_session = session
                self._msg_queue.put((
                    "live_started",
                    f"🎙️ En Vivo Neural ACTIVO ({voice_display} → {cfg.output_device.name}).",
                ))
            except Exception as exc:
                self._msg_queue.put(("live_failed", f"Error en Neural En Vivo: {exc}"))

        threading.Thread(target=_neural_worker, daemon=True, name="LiveNeuralStarter").start()

    def _stop_live_session(self) -> None:
        self._live_starting = False
        session = self._live_session
        self._live_session = None
        if session is not None:
            if session.is_recording:
                self._export_live_recording(session)
            session.stop()
        self.live_panel.set_running_state(False)
        self.audio_converter_panel.set_busy(self._is_generating)
        self.lbl_status.configure(text="⏹ Sesión En Vivo detenida.")

    def _toggle_live_recording(self) -> None:
        session = self._live_session
        if session is None or not session.is_running:
            return
        if not session.is_recording:
            session.start_recording()
            self.live_panel.set_recording_state(True)
            self.lbl_status.configure(text="🔴 Grabando audio transformado en vivo...")
        else:
            self._export_live_recording(session)

    def _export_live_recording(self, session: LiveVoiceSession) -> None:
        try:
            out_dir = self.control_panel.get_output_dir()
            fmt = self.control_panel.get_export_format()
            wav_out, mp3_out, dur_sec = session.stop_recording_and_export(out_dir, fmt)
            self.live_panel.set_recording_state(False)
            if dur_sec <= 0.05 or (wav_out is None and mp3_out is None):
                self.lbl_status.configure(text="Grabación vacía (muy corta).")
                return
            load_path = wav_out or mp3_out
            if load_path is not None:
                arr_f32, sr = load_mono_resampled(load_path, target_sr=24000)
                pcm = np.clip(arr_f32 * 32767.0, -32768.0, 32767.0).astype(np.int16)
                self.player.stop()
                self.player.load(pcm, sr)
                self.player_bar.on_new_audio_loaded(wav_out, mp3_out)
                self._last_raw_audio = pcm
                self._last_raw_sr = sr
                self.synth_panel.set_reapply_enabled(True)
            self.lbl_status.configure(
                text=f"💾 Grabación En Vivo guardada ({dur_sec:.1f} s): {(wav_out or mp3_out).name}"
            )
        except Exception as exc:
            self.live_panel.set_recording_state(False)
            self.lbl_status.configure(text=f"⚠️ Error al guardar grabación en vivo: {exc}")

    def _recalibrate_live_mic(self) -> None:
        if self._live_session is not None and self._live_session.is_running:
            self._live_session.request_recalibration()
            self.lbl_status.configure(text="🎯 Recalibrando timbre del micrófono: habla 1 segundo...")

    def _sync_live_runtime(self) -> None:
        session = self._live_session
        if session is None or not session.is_running:
            return
        session.update_synth_settings(self.synth_panel.get_settings())
        cfg_now = self.live_panel.build_live_config(self.synth_panel.get_settings())
        session.update_runtime_params(
            input_gain_db=cfg_now.input_gain_db,
            noise_gate_db=cfg_now.noise_gate_db,
            monitor_enabled=cfg_now.monitor_enabled,
            monitor_volume=cfg_now.monitor_volume,
            tau=cfg_now.tau,
        )

    def _switch_live_target_async(self) -> None:
        resolved = self._resolve_openvoice_conversion_target()
        session = self._live_session
        if not resolved or session is None or not session.is_running:
            return
        _, target_voice, target_profile_audio, _, voice_display = resolved

        def _worker():
            try:
                if target_profile_audio is not None:
                    tgt_se = self.openvoice_engine.ensure_audio_embedding(
                        target_profile_audio, prefix="ref", min_duration_sec=2.0
                    )
                else:
                    tgt_se = self.openvoice_engine.ensure_catalog_voice_embedding(target_voice)
                if self._live_session is session and session.is_running:
                    session.update_target_se(tgt_se, target_label=voice_display)
            except Exception:
                pass

        threading.Thread(target=_worker, daemon=True).start()

    def _toggle_theme(self):
        if self.theme_switch.get():
            ctk.set_appearance_mode("dark")
            self.config.theme_mode = "dark"
        else:
            ctk.set_appearance_mode("light")
            self.config.theme_mode = "light"
        self.config.save()

    def _on_close(self):
        self._stop_live_session()
        self.live_panel.shutdown()
        self.audio_converter_panel.cleanup_recorder()
        self.player.stop()
        self.clone_worker.unload_and_shutdown()
        self.config.save()
        self.destroy()
