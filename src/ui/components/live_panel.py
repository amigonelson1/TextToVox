from __future__ import annotations

import webbrowser
from pathlib import Path
from typing import Callable, List, Optional

import customtkinter as ctk

from src.core.audio.live_devices import (
    AudioDeviceInfo,
    VBCABLE_DOWNLOAD_URL,
    detect_virtual_cable_device,
    list_audio_devices,
    resolve_device_by_label,
)
from src.core.audio.live_engine import LiveConfig, LiveTelemetry
from src.core.audio.voice_synth import VoiceSynthSettings
from src.core.cloning import ClonedVoiceProfile
from src.core.config_manager import AppConfig
from src.core.voice_manager.voice_info import VoiceInfo


class LivePanel(ctk.CTkFrame):
    """Panel de cambiador de voz en tiempo real para grabaciones, reuniones y videojuegos."""

    def __init__(
        self,
        master,
        config: AppConfig,
        on_start_requested: Callable[[LiveConfig], None],
        on_stop_requested: Callable[[], None],
        on_record_toggle: Callable[[], None],
        on_recalibrate_requested: Callable[[], None],
        on_runtime_changed: Callable[[], None],
        on_config_changed: Callable[[], None],
        **kwargs,
    ):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.config = config
        self.on_start_requested = on_start_requested
        self.on_stop_requested = on_stop_requested
        self.on_record_toggle = on_record_toggle
        self.on_recalibrate_requested = on_recalibrate_requested
        self.on_runtime_changed = on_runtime_changed
        self.on_config_changed = on_config_changed

        self._input_devices: List[AudioDeviceInfo] = []
        self._output_devices: List[AudioDeviceInfo] = []
        self._label_to_input: dict[str, AudioDeviceInfo] = {}
        self._label_to_output: dict[str, AudioDeviceInfo] = {}

        self._is_running = False
        self._is_recording = False
        self._is_preparing = False
        self._target_voice_text = "Sin voz seleccionada"
        self._guide_win: Optional[ctk.CTkToplevel] = None

        self._build_ui()
        self.refresh_devices(save_if_missing=False)
        self._sync_mode_visibility()

    def _build_ui(self) -> None:
        # Encabezado y selector de modo
        header = ctk.CTkFrame(self)
        header.pack(fill="x", padx=6, pady=(6, 4))

        title_row = ctk.CTkFrame(header, fg_color="transparent")
        title_row.pack(fill="x", padx=10, pady=(8, 4))
        ctk.CTkLabel(
            title_row,
            text="🎙️ Cambiador de Voz en Vivo",
            font=ctk.CTkFont(size=15, weight="bold"),
        ).pack(side="left")

        self.btn_guide = ctk.CTkButton(
            title_row,
            text="❓ Guía Discord / Zoom / Juegos",
            width=195,
            height=26,
            fg_color="#334155",
            hover_color="#1E293B",
            font=ctk.CTkFont(size=11, weight="bold"),
            command=self._open_setup_guide,
        )
        self.btn_guide.pack(side="right")

        ctk.CTkLabel(
            header,
            text=(
                "Cambia tu voz en tiempo real durante llamadas, grabaciones o partidas. "
                "Elige baja latencia (DSP) para respuesta instantánea o OpenVoice para clonar otro timbre."
            ),
            font=ctk.CTkFont(size=11),
            text_color=("gray35", "gray70"),
            justify="left",
            wraplength=680,
        ).pack(anchor="w", padx=10, pady=(0, 6))

        mode_row = ctk.CTkFrame(header, fg_color="transparent")
        mode_row.pack(fill="x", padx=10, pady=(2, 8))
        ctk.CTkLabel(mode_row, text="Motor en vivo:", font=ctk.CTkFont(size=12, weight="bold")).pack(
            side="left", padx=(0, 8)
        )
        self.mode_seg = ctk.CTkSegmentedButton(
            mode_row,
            values=["⚡ Ultrarrápido (~95 ms)", "🧠 Neural OpenVoice (~0.65 s)"],
            command=self._on_mode_changed,
        )
        initial_mode = (
            "🧠 Neural OpenVoice (~0.65 s)"
            if self.config.live_mode == "neural"
            else "⚡ Ultrarrápido (~95 ms)"
        )
        self.mode_seg.set(initial_mode)
        self.mode_seg.pack(side="left", fill="x", expand=True)

        # Zona desplazable con dispositivos y ajustes
        self.scroll_body = ctk.CTkScrollableFrame(self, fg_color="transparent")
        self.scroll_body.pack(fill="both", expand=True, padx=2, pady=2)

        # Tarjeta info del modo elegido
        self.mode_info_card = ctk.CTkFrame(self.scroll_body)
        self.mode_info_card.pack(fill="x", padx=4, pady=4)
        self.lbl_mode_desc = ctk.CTkLabel(
            self.mode_info_card,
            text="",
            font=ctk.CTkFont(size=12),
            justify="left",
            wraplength=660,
        )
        self.lbl_mode_desc.pack(anchor="w", padx=10, pady=(8, 4))

        self.neural_row = ctk.CTkFrame(self.mode_info_card, fg_color="transparent")
        tau_head = ctk.CTkFrame(self.neural_row, fg_color="transparent")
        tau_head.pack(fill="x")
        self.lbl_tau = ctk.CTkLabel(
            tau_head,
            text=f"Similitud timbre destino (τ): {self.config.live_tau:.2f}",
            font=ctk.CTkFont(size=12, weight="bold"),
        )
        self.lbl_tau.pack(side="left")
        self.btn_recalibrate = ctk.CTkButton(
            tau_head,
            text="🎯 Recalibrar Mic",
            width=125,
            height=26,
            fg_color="#0284C7",
            hover_color="#0369A1",
            font=ctk.CTkFont(size=11, weight="bold"),
            state="disabled",
            command=self._on_recalibrate_clicked,
        )
        self.btn_recalibrate.pack(side="right")

        self.slider_tau = ctk.CTkSlider(
            self.neural_row,
            from_=0.15,
            to=0.85,
            number_of_steps=14,
            command=self._on_tau_changed,
        )
        self.slider_tau.set(max(0.15, min(0.85, float(self.config.live_tau))))
        self.slider_tau.pack(fill="x", pady=(4, 2))

        # Tarjeta de dispositivos + detección VB-Cable
        dev_card = ctk.CTkFrame(self.scroll_body)
        dev_card.pack(fill="x", padx=4, pady=4)

        dev_hdr = ctk.CTkFrame(dev_card, fg_color="transparent")
        dev_hdr.pack(fill="x", padx=10, pady=(8, 4))
        ctk.CTkLabel(
            dev_hdr,
            text="🔌 Dispositivos de Audio y Cable Virtual",
            font=ctk.CTkFont(size=13, weight="bold"),
        ).pack(side="left")

        self.btn_refresh_devs = ctk.CTkButton(
            dev_hdr,
            text="🔄 Refrescar",
            width=95,
            height=26,
            command=lambda: self.refresh_devices(save_if_missing=True, force_reinit=not self._is_running),
        )
        self.btn_refresh_devs.pack(side="right")

        self.vbcable_banner = ctk.CTkFrame(dev_card, fg_color=("gray88", "gray20"), corner_radius=6)
        self.vbcable_banner.pack(fill="x", padx=10, pady=(2, 6))
        self.lbl_vbcable = ctk.CTkLabel(
            self.vbcable_banner,
            text="Comprobando cable virtual...",
            font=ctk.CTkFont(size=11),
            justify="left",
            wraplength=500,
        )
        self.lbl_vbcable.pack(side="left", padx=8, pady=6, fill="x", expand=True)
        self.btn_get_vbcable = ctk.CTkButton(
            self.vbcable_banner,
            text="⬇️ Descargar VB-Cable",
            width=145,
            height=26,
            fg_color="#0F766E",
            hover_color="#115E59",
            font=ctk.CTkFont(size=11, weight="bold"),
            command=lambda: webbrowser.open(VBCABLE_DOWNLOAD_URL),
        )

        # Combobox Micrófono
        ctk.CTkLabel(
            dev_card, text="🎤 Micrófono de entrada:", font=ctk.CTkFont(size=12, weight="bold")
        ).pack(anchor="w", padx=10, pady=(4, 2))
        self.combo_input = ctk.CTkComboBox(
            dev_card,
            values=["Detectando micrófonos..."],
            state="readonly",
            command=self._on_device_selection_changed,
        )
        self.combo_input.pack(fill="x", padx=10, pady=(0, 6))

        # Combobox Salida principal
        ctk.CTkLabel(
            dev_card,
            text="🔊 Salida principal (VB-Cable para apps / o Altavoces):",
            font=ctk.CTkFont(size=12, weight="bold"),
        ).pack(anchor="w", padx=10, pady=(2, 2))
        self.combo_output = ctk.CTkComboBox(
            dev_card,
            values=["Detectando salidas..."],
            state="readonly",
            command=self._on_device_selection_changed,
        )
        self.combo_output.pack(fill="x", padx=10, pady=(0, 8))

        # Monitor en auriculares secundario
        mon_row = ctk.CTkFrame(dev_card, fg_color="transparent")
        mon_row.pack(fill="x", padx=10, pady=(2, 4))
        self.switch_monitor = ctk.CTkSwitch(
            mon_row,
            text="🎧 Escucharme en auriculares (Monitor privado)",
            font=ctk.CTkFont(size=12, weight="bold"),
            command=self._on_monitor_toggled,
        )
        if self.config.live_monitor_enabled:
            self.switch_monitor.select()
        self.switch_monitor.pack(side="left")

        self.lbl_mon_vol = ctk.CTkLabel(
            mon_row,
            text=f"{int(round(self.config.live_monitor_volume * 100))}%",
            width=42,
            font=ctk.CTkFont(size=11, weight="bold"),
        )
        self.lbl_mon_vol.pack(side="right")

        self.combo_monitor = ctk.CTkComboBox(
            dev_card,
            values=["Auriculares..."],
            state="readonly",
            command=self._on_device_selection_changed,
        )
        self.combo_monitor.pack(fill="x", padx=10, pady=(0, 4))

        self.slider_mon_vol = ctk.CTkSlider(
            dev_card,
            from_=0.0,
            to=1.0,
            number_of_steps=20,
            command=self._on_monitor_vol_changed,
        )
        self.slider_mon_vol.set(max(0.0, min(1.0, float(self.config.live_monitor_volume))))
        self.slider_mon_vol.pack(fill="x", padx=10, pady=(2, 10))

        # Tarjeta Ganancia & Puerta de Ruido
        mic_card = ctk.CTkFrame(self.scroll_body)
        mic_card.pack(fill="x", padx=4, pady=4)
        ctk.CTkLabel(
            mic_card,
            text="🎛️ Entrada de Micrófono y Puerta de Ruido",
            font=ctk.CTkFont(size=13, weight="bold"),
        ).pack(anchor="w", padx=10, pady=(8, 4))

        gain_hdr = ctk.CTkFrame(mic_card, fg_color="transparent")
        gain_hdr.pack(fill="x", padx=10)
        ctk.CTkLabel(gain_hdr, text="Ganancia del micrófono:", font=ctk.CTkFont(size=12)).pack(side="left")
        self.lbl_gain = ctk.CTkLabel(
            gain_hdr,
            text=f"{self.config.live_input_gain_db:+.1f} dB",
            font=ctk.CTkFont(size=12, weight="bold"),
        )
        self.lbl_gain.pack(side="right")
        self.slider_gain = ctk.CTkSlider(
            mic_card,
            from_=-12.0,
            to=18.0,
            number_of_steps=60,
            command=self._on_gain_changed,
        )
        self.slider_gain.set(max(-12.0, min(18.0, float(self.config.live_input_gain_db))))
        self.slider_gain.pack(fill="x", padx=10, pady=(2, 6))

        gate_hdr = ctk.CTkFrame(mic_card, fg_color="transparent")
        gate_hdr.pack(fill="x", padx=10)
        ctk.CTkLabel(
            gate_hdr,
            text="Puerta de ruido (silencia ventilador/teclado):",
            font=ctk.CTkFont(size=12),
        ).pack(side="left")
        self.lbl_gate = ctk.CTkLabel(
            gate_hdr,
            text=f"{int(round(self.config.live_noise_gate_db))} dB",
            font=ctk.CTkFont(size=12, weight="bold"),
        )
        self.lbl_gate.pack(side="right")
        self.slider_gate = ctk.CTkSlider(
            mic_card,
            from_=-70.0,
            to=-20.0,
            number_of_steps=50,
            command=self._on_gate_changed,
        )
        self.slider_gate.set(max(-70.0, min(-20.0, float(self.config.live_noise_gate_db))))
        self.slider_gate.pack(fill="x", padx=10, pady=(2, 10))

        # Pie fijo (siempre visible): VU Meters + Botón Activar + Grabar
        footer = ctk.CTkFrame(self)
        footer.pack(fill="x", padx=6, pady=(4, 6))

        meters_row = ctk.CTkFrame(footer, fg_color="transparent")
        meters_row.pack(fill="x", padx=10, pady=(8, 4))

        left_m = ctk.CTkFrame(meters_row, fg_color="transparent")
        left_m.pack(side="left", fill="x", expand=True, padx=(0, 6))
        ctk.CTkLabel(left_m, text="Entrada Mic", font=ctk.CTkFont(size=11, weight="bold")).pack(anchor="w")
        self.bar_in = ctk.CTkProgressBar(left_m, height=10)
        self.bar_in.set(0.0)
        self.bar_in.pack(fill="x", pady=(2, 0))

        right_m = ctk.CTkFrame(meters_row, fg_color="transparent")
        right_m.pack(side="left", fill="x", expand=True, padx=(6, 0))
        ctk.CTkLabel(right_m, text="Salida Transformada", font=ctk.CTkFont(size=11, weight="bold")).pack(anchor="w")
        self.bar_out = ctk.CTkProgressBar(right_m, height=10, progress_color="#16A34A")
        self.bar_out.set(0.0)
        self.bar_out.pack(fill="x", pady=(2, 0))

        self.lbl_telemetry = ctk.CTkLabel(
            footer,
            text="Listo para activar en vivo.",
            font=ctk.CTkFont(size=11),
            text_color=("gray30", "gray75"),
        )
        self.lbl_telemetry.pack(anchor="w", padx=10, pady=(2, 6))

        btns_row = ctk.CTkFrame(footer, fg_color="transparent")
        btns_row.pack(fill="x", padx=10, pady=(0, 8))

        self.btn_toggle_live = ctk.CTkButton(
            btns_row,
            text="▶ ACTIVAR EN VIVO",
            height=40,
            fg_color="#16A34A",
            hover_color="#15803D",
            font=ctk.CTkFont(size=13, weight="bold"),
            command=self._on_toggle_live_clicked,
        )
        self.btn_toggle_live.pack(side="left", fill="x", expand=True, padx=(0, 6))

        self.btn_record_live = ctk.CTkButton(
            btns_row,
            text="🔴 Grabar En Vivo",
            width=165,
            height=40,
            fg_color="#334155",
            hover_color="#475569",
            font=ctk.CTkFont(size=12, weight="bold"),
            state="disabled",
            command=self.on_record_toggle,
        )
        self.btn_record_live.pack(side="right")

    def refresh_devices(self, save_if_missing: bool = False, force_reinit: bool = False) -> None:
        inputs, outputs = list_audio_devices(force_reinit=force_reinit)
        self._input_devices = inputs
        self._output_devices = outputs

        self._label_to_input = {d.display_label: d for d in inputs}
        self._label_to_output = {d.display_label: d for d in outputs}

        in_labels = [d.display_label for d in inputs] or ["(No se detectaron micrófonos)"]
        out_labels = [d.display_label for d in outputs] or ["(No se detectaron salidas)"]

        self.combo_input.configure(values=in_labels)
        self.combo_output.configure(values=out_labels)
        self.combo_monitor.configure(values=out_labels)

        sel_in = resolve_device_by_label(self.config.live_input_device, inputs, prefer_virtual=False)
        sel_out = resolve_device_by_label(
            self.config.live_output_device,
            outputs,
            prefer_virtual=not bool(self.config.live_output_device),
        )
        physical_outs = [d for d in outputs if not d.is_virtual_cable] or outputs
        sel_mon = resolve_device_by_label(self.config.live_monitor_device, physical_outs, prefer_virtual=False)

        if sel_in:
            self.combo_input.set(sel_in.display_label)
            self.config.live_input_device = sel_in.display_label
        else:
            self.combo_input.set(in_labels[0])

        if sel_out:
            self.combo_output.set(sel_out.display_label)
            self.config.live_output_device = sel_out.display_label
        else:
            self.combo_output.set(out_labels[0])

        if sel_mon:
            self.combo_monitor.set(sel_mon.display_label)
            self.config.live_monitor_device = sel_mon.display_label
        else:
            self.combo_monitor.set(out_labels[0])

        vc = detect_virtual_cable_device(outputs)
        if vc is not None:
            self.btn_get_vbcable.pack_forget()
            self.lbl_vbcable.configure(
                text=f"✅ Cable Virtual detectado: {vc.name}. En Discord/Zoom/Meet elige 'CABLE Output' como micrófono.",
                text_color=("#15803D", "#4ADE80"),
            )
        else:
            self.lbl_vbcable.configure(
                text=(
                    "⚠️ Sin Cable Virtual instalado (usando altavoces/auriculares). "
                    "Para usarlo como micrófono en Discord, Zoom, Meet o Juegos instala VB-Cable (gratis) y pulsa 🔄 Refrescar."
                ),
                text_color=("#B45309", "#FBBF24"),
            )
            if not self.btn_get_vbcable.winfo_ismapped():
                self.btn_get_vbcable.pack(side="right", padx=(4, 8), pady=6)

        if save_if_missing:
            self.config.save()

    def get_mode(self) -> str:
        val = self.mode_seg.get()
        return "neural" if "Neural" in val else "dsp"

    def _sync_mode_visibility(self) -> None:
        mode = self.get_mode()
        if mode == "neural":
            self.lbl_mode_desc.configure(
                text=(
                    f"🧠 Modo Neural OpenVoice: convierte tu timbre real hacia → {self._target_voice_text}.\n"
                    "Al activar, habla 1 segundo para calibrar tu micrófono automáticamente."
                )
            )
            self.neural_row.pack(fill="x", padx=10, pady=(0, 8))
        else:
            self.neural_row.pack_forget()
            self.lbl_mode_desc.configure(
                text=(
                    "⚡ Modo Ultrarrápido (~95 ms): ideal para Discord, reuniones y videojuegos.\n"
                    "Modifica Tono, Formantes, Expresividad, Calidez y Brillo en vivo usando el panel '🎛️ Sintetizador Vocal'."
                )
            )

    def update_target_voice_label(
        self,
        voice: Optional[VoiceInfo],
        profile: Optional[ClonedVoiceProfile] = None,
    ) -> None:
        if voice is None:
            self._target_voice_text = "Ninguna"
        elif voice.engine_type in ("openvoice", "xtts") and profile is not None:
            self._target_voice_text = f"{profile.name} (Voz Clonada)"
        else:
            self._target_voice_text = voice.name
        if self.get_mode() == "neural":
            self._sync_mode_visibility()

    def _on_mode_changed(self, _value: str) -> None:
        self.config.live_mode = self.get_mode()
        self._sync_mode_visibility()
        self.on_config_changed()

    def _on_device_selection_changed(self, _val: str) -> None:
        self.config.live_input_device = self.combo_input.get()
        self.config.live_output_device = self.combo_output.get()
        self.config.live_monitor_device = self.combo_monitor.get()
        self.on_config_changed()

    def _on_monitor_toggled(self) -> None:
        self.config.live_monitor_enabled = bool(self.switch_monitor.get())
        self.on_runtime_changed()
        self.on_config_changed()

    def _on_monitor_vol_changed(self, val: float) -> None:
        vol = max(0.0, min(1.0, float(val)))
        self.config.live_monitor_volume = vol
        self.lbl_mon_vol.configure(text=f"{int(round(vol * 100))}%")
        self.on_runtime_changed()

    def _on_gain_changed(self, val: float) -> None:
        db = float(val)
        self.config.live_input_gain_db = db
        self.lbl_gain.configure(text=f"{db:+.1f} dB")
        self.on_runtime_changed()

    def _on_gate_changed(self, val: float) -> None:
        db = float(val)
        self.config.live_noise_gate_db = db
        self.lbl_gate.configure(text=f"{int(round(db))} dB")
        self.on_runtime_changed()

    def _on_tau_changed(self, val: float) -> None:
        tau = float(val)
        self.config.live_tau = tau
        self.lbl_tau.configure(text=f"Similitud timbre destino (τ): {tau:.2f}")
        self.on_runtime_changed()

    def _on_recalibrate_clicked(self) -> None:
        self.on_recalibrate_requested()

    def build_live_config(
        self,
        synth_settings: VoiceSynthSettings,
        target_se_path: Optional[Path] = None,
        target_label: str = "",
    ) -> LiveConfig:
        in_dev = self._label_to_input.get(self.combo_input.get()) or (
            self._input_devices[0] if self._input_devices else None
        )
        out_dev = self._label_to_output.get(self.combo_output.get()) or (
            self._output_devices[0] if self._output_devices else None
        )
        mon_dev = self._label_to_output.get(self.combo_monitor.get()) or out_dev
        return LiveConfig(
            mode=self.get_mode(),
            input_device=in_dev,
            output_device=out_dev,
            monitor_device=mon_dev,
            monitor_enabled=bool(self.switch_monitor.get()),
            monitor_volume=float(self.slider_mon_vol.get()),
            input_gain_db=float(self.slider_gain.get()),
            noise_gate_db=float(self.slider_gate.get()),
            tau=float(self.slider_tau.get()),
            synth_settings=synth_settings,
            target_se_path=target_se_path,
            target_label=target_label,
        )

    def _on_toggle_live_clicked(self) -> None:
        if self._is_preparing:
            return
        if self._is_running:
            self.on_stop_requested()
        else:
            self.on_start_requested(self.build_live_config(VoiceSynthSettings()))

    def set_preparing(self, msg: str) -> None:
        self._is_preparing = True
        self.btn_toggle_live.configure(state="disabled", text="⏳ Iniciando...")
        self.mode_seg.configure(state="disabled")
        self.combo_input.configure(state="disabled")
        self.combo_output.configure(state="disabled")
        self.combo_monitor.configure(state="disabled")
        self.btn_refresh_devs.configure(state="disabled")
        self.lbl_telemetry.configure(text=msg)

    def set_running_state(self, running: bool, recording: bool = False) -> None:
        self._is_preparing = False
        self._is_running = running
        self._is_recording = recording
        if running:
            self.btn_toggle_live.configure(
                state="normal",
                text="⏹ DETENER EN VIVO",
                fg_color="#DC2626",
                hover_color="#B91C1C",
            )
            self.btn_record_live.configure(state="normal")
            self.btn_recalibrate.configure(state="normal" if self.get_mode() == "neural" else "disabled")
            self.mode_seg.configure(state="disabled")
            self.combo_input.configure(state="disabled")
            self.combo_output.configure(state="disabled")
            self.combo_monitor.configure(state="disabled")
            self.btn_refresh_devs.configure(state="disabled")
        else:
            self.btn_toggle_live.configure(
                state="normal",
                text="▶ ACTIVAR EN VIVO",
                fg_color="#16A34A",
                hover_color="#15803D",
            )
            self.btn_record_live.configure(
                state="disabled",
                text="🔴 Grabar En Vivo",
                fg_color="#334155",
                hover_color="#475569",
            )
            self.btn_recalibrate.configure(state="disabled")
            self.mode_seg.configure(state="normal")
            self.combo_input.configure(state="readonly")
            self.combo_output.configure(state="readonly")
            self.combo_monitor.configure(state="readonly")
            self.btn_refresh_devs.configure(state="normal")
            self.bar_in.set(0.0)
            self.bar_out.set(0.0)
            self.lbl_telemetry.configure(text="Detenido.")

    def set_recording_state(self, recording: bool) -> None:
        self._is_recording = recording
        if recording:
            self.btn_record_live.configure(
                text="⏹ Guardar Grabación",
                fg_color="#EA580C",
                hover_color="#C2410C",
            )
        else:
            self.btn_record_live.configure(
                text="🔴 Grabar En Vivo",
                fg_color="#334155",
                hover_color="#475569",
            )

    def apply_telemetry(self, tel: LiveTelemetry) -> None:
        self.bar_in.set(max(0.0, min(1.0, tel.input_level)))
        self.bar_out.set(max(0.0, min(1.0, tel.output_level)))
        rec_note = f"  |  🔴 Grabando: {tel.recorded_sec:.1f}s" if tel.recording else ""
        xrun_note = f"  |  ⚠️ Saltos: {tel.xruns}" if tel.xruns > 0 else ""
        self.lbl_telemetry.configure(
            text=(
                f"{tel.status_text}  |  Proc: {tel.proc_latency_ms:.1f} ms · "
                f"Latencia total: ~{tel.total_latency_ms:.0f} ms{rec_note}{xrun_note}"
            )
        )

    def set_busy_external(self, busy: bool) -> None:
        """Bloquea iniciar modo vivo si hay una tarea pesada usando OpenVoice en paralelo."""
        if self._is_running or self._is_preparing:
            return
        if busy and self.get_mode() == "neural":
            self.btn_toggle_live.configure(state="disabled")
        else:
            self.btn_toggle_live.configure(state="normal")

    def shutdown(self) -> None:
        if self._guide_win is not None and self._guide_win.winfo_exists():
            try:
                self._guide_win.destroy()
            except Exception:
                pass
        if self._is_running:
            self.on_stop_requested()

    def _open_setup_guide(self) -> None:
        if self._guide_win is not None and self._guide_win.winfo_exists():
            self._guide_win.focus()
            return

        win = ctk.CTkToplevel(self)
        win.title("Cómo usar texToVox En Vivo en Discord, Zoom, Meet y Juegos")
        win.geometry("620x470")
        win.transient(self.winfo_toplevel())
        self._guide_win = win

        txt = ctk.CTkTextbox(win, wrap="word", font=ctk.CTkFont(size=12))
        txt.pack(fill="both", expand=True, padx=14, pady=(14, 8))
        txt.insert(
            "1.0",
            "GUÍA RÁPIDA: USAR TEXTOVOX COMO MICRÓFONO EN VIVO\n"
            "====================================================\n\n"
            "1) ¿Para qué necesito un Cable Virtual (VB-Cable)?\n"
            "   Windows no permite enviar el audio de un programa al micrófono de\n"
            "   Discord/Zoom/Juegos sin un controlador puente virtual.\n"
            "   • VB-Audio Virtual Cable es gratuito y muy ligero.\n"
            "   • Descárgalo, extrae el ZIP y ejecuta 'VBCABLE_Setup_x64.exe'\n"
            "     como Administrador. Luego pulsa '🔄 Refrescar' en texToVox.\n\n"
            "2) Configuración recomendada en texToVox:\n"
            "   • Micrófono de entrada: Tu micrófono físico (ej. diadema/USB).\n"
            "   • Salida principal: 'CABLE Input (VB-Audio Virtual Cable)'.\n"
            "   • Opcional: Activa '🎧 Escucharme en auriculares' eligiendo tus\n"
            "     audífonos reales para oírte sin meter eco en la llamada.\n"
            "   • Para juegos o charlas fluidas usa '⚡ Ultrarrápido (~95 ms)'.\n"
            "   • Para hablar con la voz clonada/catálogo usa '🧠 Neural OpenVoice'.\n\n"
            "3) En Discord / Zoom / Google Meet / Teams / Steam / Juegos:\n"
            "   • En Dispositivo de ENTRADA (Micrófono): elige 'CABLE Output'.\n"
            "   • En Dispositivo de SALIDA (Altavoces): deja tus auriculares normales.\n"
            "   • Desactiva 'Supresión de ruido' agresiva si atenúa frecuencias\n"
            "     muy graves del sintetizador.\n\n"
            "4) Uso sin Cable Virtual (Solo Altavoces o Grabación):\n"
            "   • Puedes activar en vivo hacia tus auriculares para probar tu voz\n"
            "     y usar '🔴 Grabar En Vivo' para guardar el audio transformado en WAV/MP3.\n"
            "   • ⚠️ Si sales por altavoces externos con micrófono abierto sin audífonos\n"
            "     podría producirse acople (eco/pitido); usa auriculares.",
        )
        txt.configure(state="disabled")

        btn_row = ctk.CTkFrame(win, fg_color="transparent")
        btn_row.pack(fill="x", padx=14, pady=(0, 12))
        ctk.CTkButton(
            btn_row,
            text="⬇️ Abrir descarga oficial de VB-Cable",
            fg_color="#0F766E",
            hover_color="#115E59",
            command=lambda: webbrowser.open(VBCABLE_DOWNLOAD_URL),
        ).pack(side="left")
        ctk.CTkButton(btn_row, text="Cerrar", width=100, command=win.destroy).pack(side="right")
