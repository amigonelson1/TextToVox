from tkinter import messagebox
from typing import Callable, Optional
import customtkinter as ctk

from src.core.audio.voice_synth import (
    BUILTIN_SYNTH_PRESETS,
    VoicePresetManager,
    VoiceSynthSettings,
)
from src.core.config_manager import AppConfig


class SynthPanel(ctk.CTkFrame):
    """
    Panel de sintetizador vocal para modificar tono (grave/agudo), formantes (cuerpo),
    expresividad y ecualización, con carga y guardado de preajustes predefinidos.
    """

    def __init__(
        self,
        master,
        config: AppConfig,
        preset_manager: VoicePresetManager,
        on_config_changed: Optional[Callable[[], None]] = None,
        on_reapply_requested: Optional[Callable[[], None]] = None,
        **kwargs
    ):
        super().__init__(master, **kwargs)
        self.config = config
        self.preset_manager = preset_manager
        self.on_config_changed = on_config_changed
        self.on_reapply_requested = on_reapply_requested
        self._presets = self.preset_manager.get_all_presets()

        self._build_ui()
        self._sync_from_config()

    def _build_ui(self):
        # 1. Encabezado + Botón Reset
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=10, pady=(6, 2))

        ctk.CTkLabel(
            header,
            text="🎛️ Sintetizador Vocal (Grave / Agudo)",
            font=ctk.CTkFont(size=13, weight="bold"),
        ).pack(side="left")

        self.btn_reset = ctk.CTkButton(
            header,
            text="↺ Natural",
            width=68,
            height=22,
            font=ctk.CTkFont(size=11),
            fg_color="gray30",
            hover_color="gray25",
            command=self._reset_to_natural,
        )
        self.btn_reset.pack(side="right")

        # 2. Selector de preajustes predefinidos + Guardar / Eliminar
        preset_row = ctk.CTkFrame(self, fg_color="transparent")
        preset_row.pack(fill="x", padx=10, pady=(2, 4))

        self.combo_presets = ctk.CTkOptionMenu(
            preset_row,
            values=list(self._presets.keys()),
            command=self._on_preset_selected,
            dynamic_resizing=False,
            height=28,
        )
        self.combo_presets.pack(side="left", fill="x", expand=True, padx=(0, 4))

        self.btn_save_preset = ctk.CTkButton(
            preset_row,
            text="💾",
            width=32,
            height=28,
            fg_color="#1f6aa5",
            hover_color="#144870",
            command=self._save_custom_preset,
        )
        self.btn_save_preset.pack(side="left", padx=(0, 3))

        self.btn_delete_preset = ctk.CTkButton(
            preset_row,
            text="🗑",
            width=32,
            height=28,
            fg_color="gray35",
            hover_color="#9e291c",
            state="disabled",
            command=self._delete_custom_preset,
        )
        self.btn_delete_preset.pack(side="left")

        # 3. Sliders del sintetizador (usando helper compartido)
        self.slider_pitch, self.lbl_pitch = self._add_slider_row(
            title="Tono (Grave ↔ Agudo):",
            from_=-8.0,
            to=8.0,
            steps=32,
            on_change=self._on_slider_changed,
        )
        self.slider_formant, self.lbl_formant = self._add_slider_row(
            title="Cuerpo / Formantes:",
            from_=0.80,
            to=1.20,
            steps=40,
            on_change=self._on_slider_changed,
        )
        self.slider_range, self.lbl_range = self._add_slider_row(
            title="Expresividad tonal:",
            from_=0.60,
            to=1.50,
            steps=18,
            on_change=self._on_slider_changed,
        )
        self.slider_warmth, self.lbl_warmth = self._add_slider_row(
            title="Calidez (Graves EQ):",
            from_=-6.0,
            to=6.0,
            steps=24,
            on_change=self._on_slider_changed,
        )
        self.slider_clarity, self.lbl_clarity = self._add_slider_row(
            title="Brillo / Presencia (Agudos EQ):",
            from_=-6.0,
            to=6.0,
            steps=24,
            on_change=self._on_slider_changed,
        )

        # 4. Botón para re-aplicar al instante sobre el último audio generado
        self.btn_reapply = ctk.CTkButton(
            self,
            text="🎛️ Aplicar al último audio generado",
            height=28,
            fg_color="gray30",
            hover_color="#1f6aa5",
            state="disabled",
            command=self._on_reapply_click,
        )
        self.btn_reapply.pack(fill="x", padx=10, pady=(4, 8))

    def _add_slider_row(
        self,
        title: str,
        from_: float,
        to: float,
        steps: int,
        on_change: Callable[[float], None],
    ):
        row_header = ctk.CTkFrame(self, fg_color="transparent")
        row_header.pack(fill="x", padx=10, pady=(2, 0))

        ctk.CTkLabel(
            row_header,
            text=title,
            font=ctk.CTkFont(size=11),
        ).pack(side="left")

        val_lbl = ctk.CTkLabel(
            row_header,
            text="",
            font=ctk.CTkFont(size=11, weight="bold"),
        )
        val_lbl.pack(side="right")

        slider = ctk.CTkSlider(
            self,
            from_=from_,
            to=to,
            number_of_steps=steps,
            height=14,
            command=on_change,
        )
        slider.pack(fill="x", padx=10, pady=(1, 3))
        return slider, val_lbl

    def _format_labels(self, settings: VoiceSynthSettings) -> None:
        st = settings.pitch_semitones
        if abs(st) < 0.05:
            pitch_txt = "0.0 st (Natural)"
        elif st < 0:
            pitch_txt = f"{st:.1f} st (Más grave)"
        else:
            pitch_txt = f"+{st:.1f} st (Más agudo)"
        self.lbl_pitch.configure(text=pitch_txt)

        fmt = settings.formant_ratio
        if abs(fmt - 1.0) < 0.01:
            fmt_txt = "1.00x (Natural)"
        elif fmt < 1.0:
            fmt_txt = f"{fmt:.2f}x (Más grueso)"
        else:
            fmt_txt = f"{fmt:.2f}x (Más ligero)"
        self.lbl_formant.configure(text=fmt_txt)

        self.lbl_range.configure(text=f"{settings.pitch_range:.2f}x")
        self.lbl_warmth.configure(text=f"{settings.warmth_db:+.1f} dB")
        self.lbl_clarity.configure(text=f"{settings.clarity_db:+.1f} dB")

    def _apply_settings_to_widgets(self, settings: VoiceSynthSettings) -> None:
        self.slider_pitch.set(settings.pitch_semitones)
        self.slider_formant.set(settings.formant_ratio)
        self.slider_range.set(settings.pitch_range)
        self.slider_warmth.set(settings.warmth_db)
        self.slider_clarity.set(settings.clarity_db)
        self._format_labels(settings)

    def _sync_from_config(self) -> None:
        current = VoiceSynthSettings(
            preset_name=self.config.synth_preset,
            pitch_semitones=self.config.synth_pitch_semitones,
            formant_ratio=self.config.synth_formant_ratio,
            pitch_range=self.config.synth_pitch_range,
            warmth_db=self.config.synth_warmth_db,
            clarity_db=self.config.synth_clarity_db,
        )
        self._apply_settings_to_widgets(current)
        self._refresh_preset_combo(select_name=self.config.synth_preset)

    def _refresh_preset_combo(self, select_name: Optional[str] = None) -> None:
        self._presets = self.preset_manager.get_all_presets()
        names = list(self._presets.keys())
        if select_name and select_name in self._presets:
            self.combo_presets.configure(values=names)
            self.combo_presets.set(select_name)
            can_delete = not self.preset_manager.is_builtin(select_name)
            self.btn_delete_preset.configure(state="normal" if can_delete else "disabled")
        else:
            display = ["Personalizado (Sin guardar)"] + names
            self.combo_presets.configure(values=display)
            self.combo_presets.set("Personalizado (Sin guardar)")
            self.btn_delete_preset.configure(state="disabled")

    def _on_preset_selected(self, chosen_name: str) -> None:
        preset = self._presets.get(chosen_name)
        if not preset:
            return
        self._apply_settings_to_widgets(preset)
        self._save_settings_to_config(preset)
        self._refresh_preset_combo(select_name=chosen_name)
        self._notify()

    def _on_slider_changed(self, _value: float) -> None:
        settings = self.get_settings()
        settings.preset_name = "Personalizado"
        self._format_labels(settings)
        self._save_settings_to_config(settings)
        self._refresh_preset_combo(select_name=None)
        self._notify()

    def _reset_to_natural(self) -> None:
        natural = BUILTIN_SYNTH_PRESETS["Natural (Sin cambios)"]
        self._apply_settings_to_widgets(natural)
        self._save_settings_to_config(natural)
        self._refresh_preset_combo(select_name=natural.preset_name)
        self._notify()

    def _save_custom_preset(self) -> None:
        dialog = ctk.CTkInputDialog(
            text="Escribe un nombre para guardar este tono predefinido:",
            title="Guardar Preajuste Vocal",
        )
        name = dialog.get_input()
        if not name or not name.strip():
            return
        try:
            saved = self.preset_manager.save_custom_preset(name.strip(), self.get_settings())
            self._save_settings_to_config(saved)
            self._refresh_preset_combo(select_name=saved.preset_name)
            self._notify()
        except Exception as exc:
            messagebox.showerror("No se pudo guardar", str(exc))

    def _delete_custom_preset(self) -> None:
        chosen = self.combo_presets.get()
        if self.preset_manager.is_builtin(chosen):
            return
        if messagebox.askyesno("Eliminar preajuste", f"¿Eliminar el preajuste '{chosen}'?"):
            self.preset_manager.delete_custom_preset(chosen)
            self.config.synth_preset = "Personalizado"
            self._refresh_preset_combo(select_name=None)
            self._notify()

    def _save_settings_to_config(self, settings: VoiceSynthSettings) -> None:
        self.config.synth_preset = settings.preset_name
        self.config.synth_pitch_semitones = round(float(settings.pitch_semitones), 2)
        self.config.synth_formant_ratio = round(float(settings.formant_ratio), 3)
        self.config.synth_pitch_range = round(float(settings.pitch_range), 2)
        self.config.synth_warmth_db = round(float(settings.warmth_db), 2)
        self.config.synth_clarity_db = round(float(settings.clarity_db), 2)

    def get_settings(self) -> VoiceSynthSettings:
        return VoiceSynthSettings(
            preset_name=self.config.synth_preset,
            pitch_semitones=round(float(self.slider_pitch.get()), 2),
            formant_ratio=round(float(self.slider_formant.get()), 3),
            pitch_range=round(float(self.slider_range.get()), 2),
            warmth_db=round(float(self.slider_warmth.get()), 2),
            clarity_db=round(float(self.slider_clarity.get()), 2),
        )

    def set_reapply_enabled(self, enabled: bool) -> None:
        self.btn_reapply.configure(state="normal" if enabled else "disabled")

    def _on_reapply_click(self) -> None:
        if self.on_reapply_requested:
            self.on_reapply_requested()

    def _notify(self) -> None:
        if self.on_config_changed:
            self.on_config_changed()
