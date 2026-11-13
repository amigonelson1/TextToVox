import os
import subprocess
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import List, Optional, Callable
import customtkinter as ctk

from src.core.cloning.profile_manager import ClonedProfileManager, ClonedVoiceProfile
from src.core.config_manager import AppConfig
from src.core.voice_manager.voice_info import VoiceInfo


class VoicePanel(ctk.CTkFrame):
    """Panel para selección, inspección, descarga y configuración de clonación de voces."""

    def __init__(
        self,
        master,
        voices: List[VoiceInfo],
        config: AppConfig,
        profile_manager: ClonedProfileManager,
        on_voice_changed: Optional[Callable[[VoiceInfo], None]] = None,
        on_download_requested: Optional[Callable[[VoiceInfo], None]] = None,
        on_preview_reference: Optional[Callable[[Path], None]] = None,
        on_preview_voice: Optional[Callable[[], None]] = None,
        on_config_changed: Optional[Callable[[], None]] = None,
        models_dir: Path = Path("models"),
        **kwargs
    ):
        super().__init__(master, **kwargs)
        self.voices = voices
        self.config = config
        self.profile_manager = profile_manager
        self.on_voice_changed = on_voice_changed
        self.on_download_requested = on_download_requested
        self.on_preview_reference = on_preview_reference
        self.on_preview_voice = on_preview_voice
        self.on_config_changed = on_config_changed
        self.models_dir = models_dir

        self.current_voice: Optional[VoiceInfo] = None
        self.profiles: List[ClonedVoiceProfile] = []
        self.base_voices: List[VoiceInfo] = []

        self._build_ui()
        self.refresh_voices(self.voices, select_key=self.config.last_voice_key)

    def _build_ui(self):
        # Título
        header_frame = ctk.CTkFrame(self, fg_color="transparent")
        header_frame.pack(fill="x", padx=10, pady=(6, 2))

        ctk.CTkLabel(
            header_frame,
            text="🎙️ Catálogo de Voces",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(side="left")

        # Botón para abrir la carpeta de modelos en el Explorador de Windows
        self.btn_open_folder = ctk.CTkButton(
            header_frame,
            text="📁 Carpeta Modelos",
            width=115,
            height=24,
            font=ctk.CTkFont(size=11),
            fg_color="gray30",
            hover_color="gray25",
            command=self._open_models_folder
        )
        self.btn_open_folder.pack(side="right")

        # Selector de Voz principal + Botón rápido de previsualización
        voice_row = ctk.CTkFrame(self, fg_color="transparent")
        voice_row.pack(fill="x", padx=10, pady=(2, 3))

        self.combo_voices = ctk.CTkOptionMenu(
            voice_row,
            values=["Cargando voces..."],
            command=self._on_combo_select,
            dynamic_resizing=False,
            height=30
        )
        self.combo_voices.pack(side="left", fill="x", expand=True, padx=(0, 5))

        self.btn_quick_preview = ctk.CTkButton(
            voice_row,
            text="▶ Probar",
            width=76,
            height=30,
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="#2a7f62",
            hover_color="#1f614b",
            command=self._on_quick_preview_click
        )
        self.btn_quick_preview.pack(side="right")

        # Información compacta de la voz seleccionada
        self.info_card = ctk.CTkFrame(self, fg_color=("gray85", "gray20"), corner_radius=6)
        self.info_card.pack(fill="x", padx=10, pady=(0, 2))

        self.lbl_details = ctk.CTkLabel(
            self.info_card,
            text="",
            font=ctk.CTkFont(size=11),
            justify="left",
            wraplength=380
        )
        self.lbl_details.pack(anchor="w", padx=8, pady=3)

        # Selector de Locutor (para modelos con múltiples speakers)
        self.speaker_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.lbl_speaker = ctk.CTkLabel(
            self.speaker_frame,
            text="Locutor:",
            font=ctk.CTkFont(size=12)
        )
        self.lbl_speaker.pack(side="left", padx=(0, 5))

        self.combo_speakers = ctk.CTkOptionMenu(
            self.speaker_frame,
            values=["Predeterminado"],
            height=26,
            width=150
        )
        self.combo_speakers.pack(side="left", fill="x", expand=True)

        # Panel contextual para Clonación de Voz (visible solo en OpenVoice / XTTS)
        self.clone_frame = ctk.CTkFrame(self, fg_color=("gray88", "gray18"), corner_radius=6)

        ctk.CTkLabel(
            self.clone_frame,
            text="🧬 Voz de Referencia a Clonar (6-20 s limpios)",
            font=ctk.CTkFont(size=12, weight="bold")
        ).pack(anchor="w", padx=8, pady=(6, 2))

        ref_row = ctk.CTkFrame(self.clone_frame, fg_color="transparent")
        ref_row.pack(fill="x", padx=8, pady=(0, 4))

        self.combo_profiles = ctk.CTkOptionMenu(
            ref_row,
            values=["Sin muestras aún"],
            command=self._on_profile_select,
            dynamic_resizing=False,
            height=28
        )
        self.combo_profiles.pack(side="left", fill="x", expand=True, padx=(0, 4))

        self.btn_upload_ref = ctk.CTkButton(
            ref_row,
            text="📂 Subir",
            width=70,
            height=28,
            command=self._upload_reference_audio
        )
        self.btn_upload_ref.pack(side="left", padx=(0, 3))

        self.btn_preview_ref = ctk.CTkButton(
            ref_row,
            text="▶",
            width=30,
            height=28,
            fg_color="#2aa84e",
            hover_color="#20823c",
            command=self._preview_selected_profile
        )
        self.btn_preview_ref.pack(side="left", padx=(0, 3))

        self.btn_delete_ref = ctk.CTkButton(
            ref_row,
            text="🗑",
            width=30,
            height=28,
            fg_color="gray35",
            hover_color="#9e291c",
            command=self._delete_selected_profile
        )
        self.btn_delete_ref.pack(side="left")

        self.lbl_ref_status = ctk.CTkLabel(
            self.clone_frame,
            text="Sube un .wav, .mp3, .ogg o .flac con voz limpia (sin música).",
            font=ctk.CTkFont(size=10),
            text_color="gray60",
            justify="left",
            wraplength=375
        )
        self.lbl_ref_status.pack(anchor="w", padx=8, pady=(0, 4))

        # Opciones exclusivas de OpenVoice híbrido (voz base latina + tau)
        self.openvoice_opts_frame = ctk.CTkFrame(self.clone_frame, fg_color="transparent")
        ctk.CTkLabel(
            self.openvoice_opts_frame,
            text="Acento / Voz Base Latina:",
            font=ctk.CTkFont(size=11)
        ).pack(anchor="w", pady=(2, 1))

        self.combo_base_voice = ctk.CTkOptionMenu(
            self.openvoice_opts_frame,
            values=["Cargando voces base..."],
            command=self._on_base_voice_select,
            dynamic_resizing=False,
            height=26
        )
        self.combo_base_voice.pack(fill="x", pady=(0, 4))

        tau_row = ctk.CTkFrame(self.openvoice_opts_frame, fg_color="transparent")
        tau_row.pack(fill="x", pady=(0, 1))
        ctk.CTkLabel(
            tau_row,
            text="Variación / Expresividad (tau):",
            font=ctk.CTkFont(size=11)
        ).pack(side="left")
        self.lbl_tau_val = ctk.CTkLabel(
            tau_row,
            text=f"{self.config.openvoice_tau:.2f}",
            font=ctk.CTkFont(size=11, weight="bold")
        )
        self.lbl_tau_val.pack(side="right")

        self.slider_tau = ctk.CTkSlider(
            self.openvoice_opts_frame,
            from_=0.15,
            to=0.70,
            number_of_steps=11,
            command=self._on_tau_change
        )
        self.slider_tau.set(self.config.openvoice_tau)
        self.slider_tau.pack(fill="x", pady=(0, 6))

        # Zona de Descarga (cuando el modelo no está instalado)
        self.download_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.download_frame.pack(fill="x", padx=10, pady=(4, 6))

        self.btn_download = ctk.CTkButton(
            self.download_frame,
            text="⬇️ Descargar Voz Seleccionada",
            fg_color="#1f6aa5",
            hover_color="#144870",
            height=30,
            command=self._on_download_click
        )
        self.btn_download.pack(fill="x")

        self.progress_download = ctk.CTkProgressBar(self.download_frame)
        self.progress_download.set(0)
        self.lbl_download_status = ctk.CTkLabel(
            self.download_frame,
            text="",
            font=ctk.CTkFont(size=11),
            text_color="gray60"
        )

    def refresh_voices(self, voices: List[VoiceInfo], select_key: Optional[str] = None):
        self.voices = voices
        if not self.voices:
            self.combo_voices.configure(values=["No se encontraron voces"])
            return

        display_options = [v.display_name for v in self.voices]
        self.combo_voices.configure(values=display_options)

        # Actualizar lista de voces base válidas para OpenVoice (Edge o Piper instaladas)
        self.base_voices = [
            v for v in self.voices
            if v.engine_type == "edge" or (v.engine_type == "piper" and v.is_installed)
        ]
        if self.base_voices:
            base_labels = [v.display_name for v in self.base_voices]
            self.combo_base_voice.configure(values=base_labels)
            saved_base = next(
                (v for v in self.base_voices if v.key == self.config.cloned_base_voice_key),
                self.base_voices[0]
            )
            self.combo_base_voice.set(saved_base.display_name)
        else:
            self.combo_base_voice.configure(values=["Sin voces base disponibles"])

        self._refresh_profiles()

        # Seleccionar la deseada o la primera instalada
        chosen_voice = None
        if select_key:
            chosen_voice = next((v for v in self.voices if v.key == select_key), None)

        if not chosen_voice:
            # Preferir primera instalada
            chosen_voice = next((v for v in self.voices if v.is_installed), self.voices[0])

        self.combo_voices.set(chosen_voice.display_name)
        self._update_selected_voice(chosen_voice)

    def _refresh_profiles(self, select_profile_id: Optional[str] = None):
        self.profiles = self.profile_manager.load_profiles()
        target_id = select_profile_id or self.config.cloned_profile_id

        if not self.profiles:
            self.combo_profiles.configure(values=["Sin muestras guardadas"])
            self.combo_profiles.set("Sin muestras guardadas")
            self.btn_preview_ref.configure(state="disabled")
            self.btn_delete_ref.configure(state="disabled")
            self.lbl_ref_status.configure(
                text="⚠️ Sube un audio de 6-20 s (.wav, .mp3, .ogg, .flac) con voz limpia."
            )
            return

        labels = [p.name for p in self.profiles]
        self.combo_profiles.configure(values=labels)
        chosen = next((p for p in self.profiles if p.profile_id == target_id), self.profiles[0])
        self.combo_profiles.set(chosen.name)
        self.config.cloned_profile_id = chosen.profile_id
        self.btn_preview_ref.configure(state="normal")
        self.btn_delete_ref.configure(state="normal")
        self.lbl_ref_status.configure(text=f"Muestra activa: {Path(chosen.audio_path).name}")

    def _on_profile_select(self, selected_name: str):
        chosen = next((p for p in self.profiles if p.name == selected_name), None)
        if chosen:
            self.config.cloned_profile_id = chosen.profile_id
            self.lbl_ref_status.configure(text=f"Muestra activa: {Path(chosen.audio_path).name}")
            if self.on_config_changed:
                self.on_config_changed()

    def _upload_reference_audio(self):
        chosen_file = filedialog.askopenfilename(
            title="Seleccionar muestra de voz a clonar (6-20 segundos)",
            filetypes=[
                ("Audio soportado", "*.wav *.mp3 *.ogg *.flac"),
                ("Todos los archivos", "*.*")
            ]
        )
        if not chosen_file:
            return
        try:
            profile = self.profile_manager.import_profile(Path(chosen_file))
            self._refresh_profiles(select_profile_id=profile.profile_id)
            if self.on_config_changed:
                self.on_config_changed()
        except Exception as exc:
            messagebox.showerror("Error al cargar muestra", str(exc))

    def _preview_selected_profile(self):
        profile = self.get_selected_profile()
        if profile and profile.exists and self.on_preview_reference:
            self.on_preview_reference(Path(profile.audio_path))

    def _delete_selected_profile(self):
        profile = self.get_selected_profile()
        if not profile:
            return
        confirm = messagebox.askyesno(
            "Eliminar muestra",
            f"¿Eliminar la muestra '{profile.name}' del catálogo local?"
        )
        if confirm:
            self.profile_manager.remove_profile(profile.profile_id)
            self._refresh_profiles()
            if self.on_config_changed:
                self.on_config_changed()

    def _on_base_voice_select(self, selected_display: str):
        base = next((v for v in self.base_voices if v.display_name == selected_display), None)
        if base:
            self.config.cloned_base_voice_key = base.key
            if self.on_config_changed:
                self.on_config_changed()

    def _on_tau_change(self, val: float):
        self.config.openvoice_tau = round(float(val), 2)
        self.lbl_tau_val.configure(text=f"{self.config.openvoice_tau:.2f}")
        if self.on_config_changed:
            self.on_config_changed()

    def get_selected_profile(self) -> Optional[ClonedVoiceProfile]:
        if not self.profiles:
            return None
        sel_name = self.combo_profiles.get()
        return next((p for p in self.profiles if p.name == sel_name), self.profiles[0])

    def get_selected_base_voice(self) -> Optional[VoiceInfo]:
        if not self.base_voices:
            return None
        sel_name = self.combo_base_voice.get()
        return next((v for v in self.base_voices if v.display_name == sel_name), self.base_voices[0])

    def get_openvoice_tau(self) -> float:
        return float(self.slider_tau.get())

    def _on_combo_select(self, selected_display: str):
        voice = next((v for v in self.voices if v.display_name == selected_display), None)
        if voice:
            self._update_selected_voice(voice)
            if self.on_voice_changed:
                self.on_voice_changed(voice)

    def _update_selected_voice(self, voice: VoiceInfo):
        self.current_voice = voice
        engine_labels = {
            "edge": "💎 Elegante (Neural Nativo)",
            "piper": "📁 100% Offline (VITS Local)",
            "openvoice": "🧬 Clon Híbrido (OpenVoice v2)",
            "xtts": "🧬 Clon Zero-Shot Offline (XTTS-v2)",
        }
        engine_label = engine_labels.get(voice.engine_type, "🧪 Motor Local")
        status_icon = "✅" if voice.is_installed else "☁️"

        details = f"{status_icon} {voice.gender} • {voice.language_name} • {engine_label}\n{voice.description}"
        self.lbl_details.configure(text=details)

        # Actualizar locutores si existen
        if len(voice.speakers) > 1:
            speaker_names = list(voice.speakers.values())
            self.combo_speakers.configure(values=speaker_names)
            self.combo_speakers.set(speaker_names[0])
            self.speaker_frame.pack(fill="x", padx=10, pady=(2, 2), after=self.info_card)
        else:
            self.speaker_frame.pack_forget()

        # Mostrar u ocultar controles de clonación según motor
        if voice.engine_type in ("openvoice", "xtts"):
            self.clone_frame.pack(fill="x", padx=10, pady=(3, 4), after=self.info_card)
            if voice.engine_type == "openvoice":
                self.openvoice_opts_frame.pack(fill="x", padx=8, pady=(0, 2))
            else:
                self.openvoice_opts_frame.pack_forget()
        else:
            self.openvoice_opts_frame.pack_forget()
            self.clone_frame.pack_forget()

        # Mostrar u ocultar botón de descarga y estado del botón rápido de prueba
        self.btn_quick_preview.configure(state="normal" if voice.is_installed else "disabled")
        if voice.is_installed:
            self.btn_download.pack_forget()
            self.progress_download.pack_forget()
            self.lbl_download_status.pack_forget()
        else:
            self.btn_download.configure(text=f"⬇️ Descargar {voice.name}")
            self.btn_download.pack(fill="x")

    def set_preview_enabled(self, enabled: bool) -> None:
        can_preview = bool(enabled and self.current_voice and self.current_voice.is_installed)
        self.btn_quick_preview.configure(state="normal" if can_preview else "disabled")

    def _on_quick_preview_click(self) -> None:
        if self.on_preview_voice:
            self.on_preview_voice()

    def get_selected_speaker_id(self) -> int:
        if not self.current_voice or len(self.current_voice.speakers) <= 1:
            return 0
        selected_name = self.combo_speakers.get()
        for idx, name in self.current_voice.speakers.items():
            if name == selected_name:
                return idx
        return 0

    def _on_download_click(self):
        if self.current_voice and self.on_download_requested:
            self.btn_download.configure(state="disabled")
            self.progress_download.pack(fill="x", pady=(4, 2))
            self.lbl_download_status.pack(fill="x")
            self.progress_download.set(0)
            self.lbl_download_status.configure(text="Iniciando descarga...")
            self.on_download_requested(self.current_voice)

    def update_download_progress(self, percent: float, msg: str):
        self.progress_download.set(percent / 100.0)
        self.lbl_download_status.configure(text=msg)
        if percent >= 100.0:
            self.btn_download.configure(state="normal")
            if self.current_voice:
                self._update_selected_voice(self.current_voice)

    def _open_models_folder(self):
        self.models_dir.mkdir(parents=True, exist_ok=True)
        try:
            os.startfile(str(self.models_dir.resolve()))
        except Exception:
            subprocess.run(["explorer", str(self.models_dir.resolve())])

