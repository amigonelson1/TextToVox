import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from scipy import signal
import parselmouth
from parselmouth.praat import call


PRESETS_PATH = Path("config/voice_presets.json")


@dataclass
class VoiceSynthSettings:
    """Parámetros del sintetizador vocal (tono, formantes, expresividad y ecualización)."""
    preset_name: str = "Natural (Sin cambios)"
    pitch_semitones: float = 0.0   # -8.0 (grave) .. +8.0 (agudo)
    formant_ratio: float = 1.0     # 0.80 (grueso) .. 1.20 (ligero/fino)
    pitch_range: float = 1.0       # 0.60 (sobrio) .. 1.50 (expresivo)
    warmth_db: float = 0.0         # -6.0 .. +6.0 dB (graves ~180 Hz)
    clarity_db: float = 0.0        # -6.0 .. +6.0 dB (brillo ~4200 Hz)

    def needs_praat(self) -> bool:
        return (
            abs(self.pitch_semitones) >= 0.05
            or abs(self.formant_ratio - 1.0) >= 0.005
            or abs(self.pitch_range - 1.0) >= 0.01
        )

    def needs_eq(self) -> bool:
        return abs(self.warmth_db) >= 0.05 or abs(self.clarity_db) >= 0.05

    def is_neutral(self) -> bool:
        return not self.needs_praat() and not self.needs_eq()


BUILTIN_SYNTH_PRESETS: Dict[str, VoiceSynthSettings] = {
    "Natural (Sin cambios)": VoiceSynthSettings(
        preset_name="Natural (Sin cambios)",
        pitch_semitones=0.0,
        formant_ratio=1.0,
        pitch_range=1.0,
        warmth_db=0.0,
        clarity_db=0.0,
    ),
    "Radio / Locutor Grave": VoiceSynthSettings(
        preset_name="Radio / Locutor Grave",
        pitch_semitones=-3.0,
        formant_ratio=0.93,
        pitch_range=1.05,
        warmth_db=3.0,
        clarity_db=1.5,
    ),
    "Narrador Profundo / Cine": VoiceSynthSettings(
        preset_name="Narrador Profundo / Cine",
        pitch_semitones=-4.5,
        formant_ratio=0.90,
        pitch_range=0.95,
        warmth_db=3.5,
        clarity_db=1.0,
    ),
    "Corporativo Claro": VoiceSynthSettings(
        preset_name="Corporativo Claro",
        pitch_semitones=-0.5,
        formant_ratio=0.98,
        pitch_range=1.10,
        warmth_db=1.0,
        clarity_db=2.5,
    ),
    "Juvenil / Agudo Natural": VoiceSynthSettings(
        preset_name="Juvenil / Agudo Natural",
        pitch_semitones=3.0,
        formant_ratio=1.06,
        pitch_range=1.15,
        warmth_db=-1.0,
        clarity_db=2.0,
    ),
    "Voz Brillante / Comercial": VoiceSynthSettings(
        preset_name="Voz Brillante / Comercial",
        pitch_semitones=1.5,
        formant_ratio=1.03,
        pitch_range=1.25,
        warmth_db=0.0,
        clarity_db=3.0,
    ),
    "Documental Sobrio": VoiceSynthSettings(
        preset_name="Documental Sobrio",
        pitch_semitones=-1.5,
        formant_ratio=0.97,
        pitch_range=0.75,
        warmth_db=1.5,
        clarity_db=1.0,
    ),
}


class VoicePresetManager:
    """Administra preajustes integrados y predefinidos personalizados en JSON."""

    def __init__(self, presets_path: Path = PRESETS_PATH):
        self.presets_path = Path(presets_path)

    def load_custom_presets(self) -> Dict[str, VoiceSynthSettings]:
        if not self.presets_path.is_file():
            return {}
        try:
            raw = json.loads(self.presets_path.read_text(encoding="utf-8"))
            custom: Dict[str, VoiceSynthSettings] = {}
            for name, vals in raw.items():
                if name in BUILTIN_SYNTH_PRESETS or not isinstance(vals, dict):
                    continue
                filtered = {
                    k: v for k, v in vals.items()
                    if k in VoiceSynthSettings.__dataclass_fields__
                }
                filtered["preset_name"] = name
                custom[name] = VoiceSynthSettings(**filtered)
            return custom
        except Exception:
            return {}

    def get_all_presets(self) -> Dict[str, VoiceSynthSettings]:
        combined = dict(BUILTIN_SYNTH_PRESETS)
        combined.update(self.load_custom_presets())
        return combined

    def is_builtin(self, name: str) -> bool:
        return name in BUILTIN_SYNTH_PRESETS

    def save_custom_preset(self, name: str, settings: VoiceSynthSettings) -> VoiceSynthSettings:
        clean_name = name.strip()
        if not clean_name:
            raise ValueError("El nombre del preajuste no puede estar vacío.")
        if clean_name in BUILTIN_SYNTH_PRESETS:
            raise ValueError(f"'{clean_name}' es un preajuste integrado. Usa otro nombre.")

        custom = self.load_custom_presets()
        stored = VoiceSynthSettings(
            preset_name=clean_name,
            pitch_semitones=round(float(settings.pitch_semitones), 2),
            formant_ratio=round(float(settings.formant_ratio), 3),
            pitch_range=round(float(settings.pitch_range), 2),
            warmth_db=round(float(settings.warmth_db), 2),
            clarity_db=round(float(settings.clarity_db), 2),
        )
        custom[clean_name] = stored
        self._write_custom_file(custom)
        return stored

    def delete_custom_preset(self, name: str) -> bool:
        if name in BUILTIN_SYNTH_PRESETS:
            return False
        custom = self.load_custom_presets()
        if name not in custom:
            return False
        del custom[name]
        self._write_custom_file(custom)
        return True

    def _write_custom_file(self, custom: Dict[str, VoiceSynthSettings]) -> None:
        self.presets_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {k: asdict(v) for k, v in custom.items()}
        self.presets_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


class VoiceSynthProcessor:
    """
    Procesador DSP vocal de alta calidad:
    - Modifica tono (semitonos), cuerpo (formantes) y expresividad mediante Praat TD-PSOLA
      sin cambiar la velocidad ni la duración del habla.
    - Ajusta calidez (graves) y brillo (agudos) con filtros shelving IIR biquad en scipy.
    """

    @staticmethod
    def _biquad_shelf_sos(
        sr: int,
        fc: float,
        gain_db: float,
        shelf_type: str = "low",
        slope: float = 0.707,
    ) -> Optional[np.ndarray]:
        if abs(gain_db) < 0.05 or fc >= sr * 0.45:
            return None

        amp = 10.0 ** (gain_db / 40.0)
        w0 = 2.0 * math.pi * fc / sr
        cos_w0 = math.cos(w0)
        sin_w0 = math.sin(w0)
        alpha = sin_w0 / (2.0 * slope)
        two_sqrt_a_alpha = 2.0 * math.sqrt(amp) * alpha

        if shelf_type == "low":
            b0 = amp * ((amp + 1.0) - (amp - 1.0) * cos_w0 + two_sqrt_a_alpha)
            b1 = 2.0 * amp * ((amp - 1.0) - (amp + 1.0) * cos_w0)
            b2 = amp * ((amp + 1.0) - (amp - 1.0) * cos_w0 - two_sqrt_a_alpha)
            a0 = (amp + 1.0) + (amp - 1.0) * cos_w0 + two_sqrt_a_alpha
            a1 = -2.0 * ((amp - 1.0) + (amp + 1.0) * cos_w0)
            a2 = (amp + 1.0) + (amp - 1.0) * cos_w0 - two_sqrt_a_alpha
        else:
            b0 = amp * ((amp + 1.0) + (amp - 1.0) * cos_w0 + two_sqrt_a_alpha)
            b1 = -2.0 * amp * ((amp - 1.0) + (amp + 1.0) * cos_w0)
            b2 = amp * ((amp + 1.0) + (amp - 1.0) * cos_w0 - two_sqrt_a_alpha)
            a0 = (amp + 1.0) - (amp - 1.0) * cos_w0 + two_sqrt_a_alpha
            a1 = 2.0 * ((amp - 1.0) - (amp + 1.0) * cos_w0)
            a2 = (amp + 1.0) - (amp - 1.0) * cos_w0 - two_sqrt_a_alpha

        return np.array(
            [[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]],
            dtype=np.float64,
        )

    @classmethod
    def process(
        cls,
        audio_int16: np.ndarray,
        sample_rate: int,
        settings: VoiceSynthSettings,
    ) -> np.ndarray:
        if audio_int16 is None or len(audio_int16) == 0 or settings.is_neutral():
            return audio_int16

        if len(audio_int16) < int(0.08 * sample_rate):
            return audio_int16

        y = audio_int16.astype(np.float64) / 32768.0
        peak_in = float(np.max(np.abs(y)))
        if peak_in < 1e-4:
            return audio_int16

        # 1. Tono, formantes y rango melódico (TD-PSOLA vía Praat)
        if settings.needs_praat():
            try:
                snd = parselmouth.Sound(y, sampling_frequency=float(sample_rate))
                pitch = call(snd, "To Pitch", 0.0, 50.0, 600.0)
                orig_f0 = float(call(pitch, "Get quantile", 0.0, 0.0, 0.5, "Hertz"))
                if not math.isnan(orig_f0) and orig_f0 > 0.0:
                    factor = 2.0 ** (float(settings.pitch_semitones) / 12.0)
                    new_f0 = max(42.0, min(550.0, orig_f0 * factor))
                    formant = max(0.75, min(1.30, float(settings.formant_ratio)))
                    p_range = max(0.50, min(1.75, float(settings.pitch_range)))
                    shifted = call(
                        [snd, pitch],
                        "Change gender",
                        formant,
                        new_f0,
                        p_range,
                        1.0,
                    )
                    y = np.asarray(shifted.values[0], dtype=np.float64)
            except Exception:
                # Si un fragmento muy corto o ruidoso falla en extracción de F0, continuar con EQ
                pass

        # 2. Ecualización de calidez y brillo
        if settings.needs_eq():
            for fc, gain_db, kind in (
                (180.0, float(settings.warmth_db), "low"),
                (4200.0, float(settings.clarity_db), "high"),
            ):
                sos = cls._biquad_shelf_sos(sample_rate, fc, gain_db, shelf_type=kind)
                if sos is not None:
                    y = signal.sosfilt(sos, y)

        # 3. Limpiar posibles NaN/Inf y normalizar pico a -1.0 dBFS de forma segura
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        peak_out = float(np.max(np.abs(y)))
        if peak_out > 0.0:
            target_peak = 10.0 ** (-1.0 / 20.0)  # ~0.891
            gain = min(target_peak / peak_out, 4.0) if peak_out < target_peak else (target_peak / peak_out)
            y = y * gain

        y = np.clip(y * 32767.0, -32768.0, 32767.0)
        return y.astype(np.int16)

