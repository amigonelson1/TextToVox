from __future__ import annotations

import math
import queue
import threading
import time
import warnings
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from typing import Callable, Deque, List, Optional, Tuple

import numpy as np
import parselmouth
from parselmouth.praat import call
from scipy import signal
import sounddevice as sd

from src.core.audio.exporter import AudioExporter
from src.core.audio.live_devices import AudioDeviceInfo, find_fallback_hostapi_device
from src.core.audio.voice_synth import VoiceSynthProcessor, VoiceSynthSettings


def _resample_f32(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Remuestreo rápido mono float32 entre tasas de hardware y tasas de motor."""
    if audio.size == 0 or src_sr == dst_sr:
        return np.asarray(audio, dtype=np.float32)
    frac = Fraction(int(dst_sr), int(src_sr)).limit_denominator(500)
    resampled = signal.resample_poly(audio.astype(np.float32, copy=False), frac.numerator, frac.denominator)
    return np.asarray(resampled, dtype=np.float32)


def _soft_limit_f32(audio: np.ndarray, knee: float = 0.90) -> np.ndarray:
    """Limitador suave anti-clipping que conserva volumen uniforme sin distorsión dura."""
    if audio.size == 0:
        return audio
    out = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    peak = float(np.max(np.abs(out)))
    if peak > knee:
        scale = 1.0 / (1.0 - knee)
        abs_x = np.abs(out)
        mask = abs_x > knee
        excess = (abs_x[mask] - knee) * scale
        compressed = knee + (1.0 - knee) * np.tanh(excess)
        out = out.copy()
        out[mask] = np.sign(out[mask]) * compressed
    return np.clip(out, -0.98, 0.98).astype(np.float32, copy=False)


class _NoiseGateEnvelope:
    """Puerta de ruido con histéresis (hangover) y rampa suave por muestra sin cortes bruscos."""

    def __init__(self, sample_rate: int, threshold_db: float = -45.0, hangover_ms: float = 180.0):
        self.sample_rate = int(sample_rate)
        self.threshold_db = float(threshold_db)
        self.hangover_samples = max(1, int(self.sample_rate * hangover_ms / 1000.0))
        self._hold_left = 0
        self._current_gain = 0.0
        self.is_open = False

    def set_threshold_db(self, threshold_db: float) -> None:
        self.threshold_db = float(max(-75.0, min(-15.0, threshold_db)))

    def process_hop(self, hop: np.ndarray) -> Tuple[np.ndarray, bool, float]:
        """Retorna (hop_atenudado, puerta_abierta, rms_db)."""
        if hop.size == 0:
            return hop, False, -90.0

        rms = float(np.sqrt(np.mean(np.square(hop, dtype=np.float64)) + 1e-12))
        rms_db = 20.0 * math.log10(max(rms, 1e-6))
        peak = float(np.max(np.abs(hop)))
        peak_db = 20.0 * math.log10(max(peak, 1e-6))

        # Abrir si el RMS supera el umbral o si hay consonante/transiente claro 6 dB encima
        triggered = (rms_db >= self.threshold_db) or (peak_db >= (self.threshold_db + 8.0))
        if triggered:
            self._hold_left = self.hangover_samples
        elif self._hold_left > 0:
            self._hold_left = max(0, self._hold_left - len(hop))

        target_gain = 1.0 if self._hold_left > 0 else 0.0
        self.is_open = target_gain > 0.5

        if abs(target_gain - self._current_gain) < 1e-4:
            if target_gain == 0.0:
                return np.zeros_like(hop, dtype=np.float32), False, rms_db
            return hop.astype(np.float32, copy=False), True, rms_db

        # Rampa suave (ataque rápido ~8 ms, cierre ~40 ms) evitando clics
        ramp_len = min(
            len(hop),
            int(self.sample_rate * (0.008 if target_gain > self._current_gain else 0.045)),
        )
        ramp_len = max(1, ramp_len)
        env = np.empty(len(hop), dtype=np.float32)
        env[:ramp_len] = np.linspace(self._current_gain, target_gain, ramp_len, dtype=np.float32)
        if ramp_len < len(hop):
            env[ramp_len:] = target_gain
        self._current_gain = target_gain
        return (hop * env).astype(np.float32, copy=False), self.is_open, rms_db


class _StatefulBiquadEQ:
    """Filtros shelving IIR con memoria continua `zi` entre hops para cero clics en los bordes."""

    def __init__(self, sample_rate: int):
        self.sample_rate = int(sample_rate)
        self._last_key: Optional[Tuple[float, float]] = None
        self._sos_low: Optional[np.ndarray] = None
        self._sos_high: Optional[np.ndarray] = None
        self._zi_low: Optional[np.ndarray] = None
        self._zi_high: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._zi_low = None
        self._zi_high = None

    def _ensure_filters(self, warmth_db: float, clarity_db: float) -> None:
        key = (round(float(warmth_db), 2), round(float(clarity_db), 2))
        if key == self._last_key:
            return
        self._last_key = key
        self._sos_low = VoiceSynthProcessor._biquad_shelf_sos(
            self.sample_rate, 180.0, key[0], shelf_type="low"
        )
        self._sos_high = VoiceSynthProcessor._biquad_shelf_sos(
            self.sample_rate, 4200.0, key[1], shelf_type="high"
        )
        if self._sos_low is None:
            self._zi_low = None
        if self._sos_high is None:
            self._zi_high = None

    def process(self, audio: np.ndarray, settings: VoiceSynthSettings) -> np.ndarray:
        if audio.size == 0 or not settings.needs_eq():
            self._zi_low = None
            self._zi_high = None
            return audio

        self._ensure_filters(settings.warmth_db, settings.clarity_db)
        y = audio.astype(np.float64, copy=False)
        if self._sos_low is not None:
            if self._zi_low is None:
                self._zi_low = signal.sosfilt_zi(self._sos_low) * float(y[0])
            y, self._zi_low = signal.sosfilt(self._sos_low, y, zi=self._zi_low)
        if self._sos_high is not None:
            if self._zi_high is None:
                self._zi_high = signal.sosfilt_zi(self._sos_high) * float(y[0])
            y, self._zi_high = signal.sosfilt(self._sos_high, y, zi=self._zi_high)
        return y.astype(np.float32)


class _RaisedCosineCrossfader:
    """Overlap-crossfade exacto de longitud constante por hop con curva coseno alzado."""

    def __init__(self, xfade_samples: int):
        self.xfade_samples = max(16, int(xfade_samples))
        t = np.linspace(0.0, 1.0, self.xfade_samples, endpoint=False, dtype=np.float32)
        self._fade_in = 0.5 * (1.0 - np.cos(np.pi * t))
        self._fade_out = 1.0 - self._fade_in
        self._prev_tail = np.zeros(self.xfade_samples, dtype=np.float32)

    def reset(self) -> None:
        self._prev_tail.fill(0.0)

    def blend(self, pre_overlap: np.ndarray, hop_body_and_tail: np.ndarray) -> np.ndarray:
        """
        Combina:
        - `pre_overlap`: últimos `xfade_samples` del bloque anterior re-procesados en el contexto nuevo.
        - `hop_body_and_tail`: bloque nuevo de longitud exacta `hop_samples` (`body + next_tail`).
        Retorna exactamente `hop_samples` muestras continuas sin saltos de fase ni amplitud.
        """
        n_hop = len(hop_body_and_tail)
        n_xf = min(self.xfade_samples, n_hop, len(pre_overlap))

        out = np.empty(n_hop, dtype=np.float32)
        if n_xf > 0:
            xfaded = (
                self._prev_tail[:n_xf] * self._fade_out[:n_xf]
                + pre_overlap[-n_xf:] * self._fade_in[:n_xf]
            )
            out[:n_xf] = xfaded
            if n_hop > n_xf:
                out[n_xf:] = hop_body_and_tail[: n_hop - n_xf]
            self._prev_tail = hop_body_and_tail[-n_xf:].copy()
        else:
            out[:] = hop_body_and_tail

        return out


class LiveDSPStreamProcessor:
    """
    Procesador en vivo ultrarrápido (<95 ms algorítmicos) basado en Praat TD-PSOLA + EQ IIR continuo.
    Ideal para reuniones, Discord, Zoom, Meet y videojuegos sin esperar inferencia neuronal.
    """

    def __init__(
        self,
        settings: Optional[VoiceSynthSettings] = None,
        sample_rate: int = 24000,
        hop_ms: float = 75.0,
        ctx_ms: float = 160.0,
        xfade_ms: float = 16.0,
    ):
        self.sample_rate = int(sample_rate)
        self.hop_samples = max(256, int(round(self.sample_rate * hop_ms / 1000.0)))
        self.ctx_samples = max(self.hop_samples, int(round(self.sample_rate * ctx_ms / 1000.0)))
        self.xfade_samples = min(
            self.hop_samples // 2,
            max(64, int(round(self.sample_rate * xfade_ms / 1000.0))),
        )

        self._lock = threading.Lock()
        self._settings = settings or VoiceSynthSettings()
        self._ring = np.zeros(self.ctx_samples + self.hop_samples, dtype=np.float32)
        self._crossfader = _RaisedCosineCrossfader(self.xfade_samples)
        self._eq = _StatefulBiquadEQ(self.sample_rate)
        self._ema_f0: Optional[float] = None

    def update_settings(self, settings: VoiceSynthSettings) -> None:
        with self._lock:
            self._settings = VoiceSynthSettings(
                preset_name=settings.preset_name,
                pitch_semitones=settings.pitch_semitones,
                formant_ratio=settings.formant_ratio,
                pitch_range=settings.pitch_range,
                warmth_db=settings.warmth_db,
                clarity_db=settings.clarity_db,
            )

    def reset(self) -> None:
        self._ring.fill(0.0)
        self._crossfader.reset()
        self._eq.reset()
        self._ema_f0 = None

    def process_hop(self, hop: np.ndarray, gate_open: bool = True) -> np.ndarray:
        """Procesa exactamente `len(hop)` muestras a `self.sample_rate`."""
        hop_f32 = np.asarray(hop, dtype=np.float32)
        n_in = len(hop_f32)
        if n_in == 0:
            return hop_f32

        # Mantener ring desplazado exactamente n_in muestras
        if n_in >= len(self._ring):
            self._ring[:] = hop_f32[-len(self._ring):]
        else:
            self._ring[:-n_in] = self._ring[n_in:]
            self._ring[-n_in:] = hop_f32

        with self._lock:
            cfg = self._settings

        ctx_len = len(self._ring) - n_in
        n_xf = min(self.xfade_samples, n_in // 2, ctx_len)

        if not gate_open:
            # En silencio tras gate: limpiar estado de F0 gradualmente y cruzar hacia ceros
            self._ema_f0 = None
            pre_overlap = np.zeros(n_xf, dtype=np.float32)
            out_hop = self._crossfader.blend(pre_overlap, np.zeros(n_in, dtype=np.float32))
            return self._eq.process(out_hop, cfg)

        if not cfg.needs_praat():
            pre_overlap = self._ring[ctx_len - n_xf: ctx_len] if n_xf > 0 else np.zeros(0, dtype=np.float32)
            out_hop = self._crossfader.blend(pre_overlap, hop_f32)
            out_hop = self._eq.process(out_hop, cfg)
            return _soft_limit_f32(out_hop)

        # Procesamiento TD-PSOLA en la ventana [contexto + hop actual]
        proc_window = self._ring.astype(np.float64)
        peak = float(np.max(np.abs(proc_window)))
        if peak >= 1e-4:
            try:
                snd = parselmouth.Sound(proc_window, sampling_frequency=float(self.sample_rate))
                pitch = call(snd, "To Pitch", 0.0, 55.0, 550.0)
                orig_f0 = float(call(pitch, "Get quantile", 0.0, 0.0, 0.5, "Hertz"))
                if not math.isnan(orig_f0) and 45.0 <= orig_f0 <= 550.0:
                    if self._ema_f0 is None:
                        self._ema_f0 = orig_f0
                    else:
                        # Evitar saltos bruscos de octava entre ventanas de 75 ms
                        if abs(math.log2(orig_f0 / self._ema_f0)) < 0.75:
                            self._ema_f0 = 0.65 * self._ema_f0 + 0.35 * orig_f0
                        else:
                            self._ema_f0 = 0.85 * self._ema_f0 + 0.15 * orig_f0

                ref_f0 = self._ema_f0
                if ref_f0 is not None and ref_f0 > 0.0:
                    factor = 2.0 ** (float(cfg.pitch_semitones) / 12.0)
                    new_f0 = max(42.0, min(550.0, ref_f0 * factor))
                    formant = max(0.75, min(1.30, float(cfg.formant_ratio)))
                    p_range = max(0.50, min(1.75, float(cfg.pitch_range)))
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        shifted = call(
                            [snd, pitch],
                            "Change gender",
                            formant,
                            new_f0,
                            p_range,
                            1.0,
                        )
                    arr = np.asarray(shifted.values[0], dtype=np.float32)
                    # Alinear exactamente por la cola si Praat difiere 1-2 muestras
                    if len(arr) >= len(self._ring):
                        proc_window = arr[-len(self._ring):]
                    elif len(arr) >= n_in + n_xf:
                        padded = np.zeros(len(self._ring), dtype=np.float32)
                        padded[-len(arr):] = arr
                        proc_window = padded
            except Exception:
                pass

        proc_f32 = proc_window.astype(np.float32, copy=False)
        pre_overlap = proc_f32[ctx_len - n_xf: ctx_len] if n_xf > 0 else np.zeros(0, dtype=np.float32)
        body_tail = proc_f32[ctx_len: ctx_len + n_in]
        out_hop = self._crossfader.blend(pre_overlap, body_tail)
        out_hop = self._eq.process(out_hop, cfg)
        return _soft_limit_f32(out_hop)


class LiveNeuralStreamProcessor:
    """
    Procesador neuronal en streaming usando OpenVoice v2 sobre PCM en memoria (sin I/O a disco).
    Auto-calibra el timbre del micrófono en ~0.9s de habla útil y convierte hacia la voz elegida.
    """

    def __init__(
        self,
        openvoice_engine,
        tgt_se_path: Path,
        synth_settings: Optional[VoiceSynthSettings] = None,
        tau: float = 0.45,
        hop_samples: int = 12800,    # 50 * 256 frames (~580.5 ms @ 22050 Hz)
        ctx_samples: int = 7680,     # 30 * 256 frames (~348.3 ms @ 22050 Hz)
        xfade_samples: int = 896,    # ~40.6 ms
    ):
        self.ov = openvoice_engine
        self.sample_rate = int(openvoice_engine.get_sample_rate() or 22050)
        self.hop_samples = int(hop_samples)
        self.ctx_samples = int(ctx_samples)
        self.xfade_samples = int(xfade_samples)
        self._right_pad = 768  # compense filter_length(1024) - hop_length(256) con center=False

        self._lock = threading.Lock()
        self._tgt_se_path = Path(tgt_se_path).resolve()
        self._tau = float(max(0.15, min(0.85, tau)))
        self._settings = synth_settings or VoiceSynthSettings()

        self._ring = np.zeros(self.ctx_samples + self.hop_samples, dtype=np.float32)
        self._crossfader = _RaisedCosineCrossfader(self.xfade_samples)
        self._eq = _StatefulBiquadEQ(self.sample_rate)

        self._live_se_key = "live_mic_active"
        self._calib_chunks: List[np.ndarray] = []
        self._calib_samples = 0
        self._calibrated = False
        self._refined = False
        self._recalibrate_requested = False

    def update_settings(self, settings: VoiceSynthSettings) -> None:
        with self._lock:
            self._settings = VoiceSynthSettings(
                preset_name=settings.preset_name,
                pitch_semitones=settings.pitch_semitones,
                formant_ratio=settings.formant_ratio,
                pitch_range=settings.pitch_range,
                warmth_db=settings.warmth_db,
                clarity_db=settings.clarity_db,
            )

    def update_tau(self, tau: float) -> None:
        with self._lock:
            self._tau = float(max(0.15, min(0.85, tau)))

    def update_target_se(self, tgt_se_path: Path) -> None:
        with self._lock:
            self._tgt_se_path = Path(tgt_se_path).resolve()

    def request_recalibration(self) -> None:
        with self._lock:
            self._calib_chunks.clear()
            self._calib_samples = 0
            self._calibrated = False
            self._refined = False
            self._recalibrate_requested = True

    @property
    def is_calibrated(self) -> bool:
        return self._calibrated

    @property
    def calibrated_seconds(self) -> float:
        return float(self._calib_samples / self.sample_rate)

    def _try_calibrate(self, voiced_hop: np.ndarray) -> None:
        if self._refined:
            return

        rms = float(np.sqrt(np.mean(np.square(voiced_hop)) + 1e-12))
        if rms < 0.012:
            return

        self._calib_chunks.append(voiced_hop.copy())
        self._calib_samples += len(voiced_hop)

        # Primera calibración rápida tras ~0.85s de habla útil
        if not self._calibrated and self._calib_samples >= int(self.sample_rate * 0.85):
            audio_cat = np.concatenate(self._calib_chunks)
            try:
                self.ov.extract_pcm_embedding(
                    audio_cat,
                    sample_rate=self.sample_rate,
                    se_key=self._live_se_key,
                    min_duration_sec=0.35,
                )
                self._calibrated = True
            except Exception:
                # Si todavía tenía demasiado silencio interno, seguir acumulando hasta 2.5s
                if self._calib_samples > int(self.sample_rate * 3.0):
                    self._calib_chunks = self._calib_chunks[-3:]
                    self._calib_samples = sum(len(c) for c in self._calib_chunks)
        # Refinamiento único al alcanzar ~3.0s de voz limpia del usuario
        elif self._calibrated and not self._refined and self._calib_samples >= int(self.sample_rate * 2.9):
            audio_cat = np.concatenate(self._calib_chunks)
            try:
                self.ov.extract_pcm_embedding(
                    audio_cat,
                    sample_rate=self.sample_rate,
                    se_key=self._live_se_key,
                    min_duration_sec=1.2,
                )
                self._refined = True
                self._calib_chunks.clear()
            except Exception:
                self._refined = True
                self._calib_chunks.clear()

    def process_hop(self, hop: np.ndarray, gate_open: bool = True) -> np.ndarray:
        hop_f32 = np.asarray(hop, dtype=np.float32)
        n_in = len(hop_f32)
        if n_in == 0:
            return hop_f32

        if n_in >= len(self._ring):
            self._ring[:] = hop_f32[-len(self._ring):]
        else:
            self._ring[:-n_in] = self._ring[n_in:]
            self._ring[-n_in:] = hop_f32

        with self._lock:
            cfg = self._settings
            tgt_se_path = self._tgt_se_path
            tau = self._tau

        ctx_len = len(self._ring) - n_in
        n_xf = min(self.xfade_samples, n_in // 2, ctx_len)

        if not gate_open:
            pre_overlap = np.zeros(n_xf, dtype=np.float32)
            out_hop = self._crossfader.blend(pre_overlap, np.zeros(n_in, dtype=np.float32))
            return self._eq.process(out_hop, cfg)

        self._try_calibrate(hop_f32)
        if not self._calibrated:
            # Mientras calibra el primer segundo, devolver passthrough limpio suave
            pre_overlap = self._ring[ctx_len - n_xf: ctx_len] if n_xf > 0 else np.zeros(0, dtype=np.float32)
            out_hop = self._crossfader.blend(pre_overlap, hop_f32 * 0.75)
            return _soft_limit_f32(self._eq.process(out_hop, cfg))

        # Añadir padding derecho reflejado/suave para compensar center=False en STFT de OpenVoice
        tail_pad = np.zeros(self._right_pad, dtype=np.float32)
        window = np.concatenate([self._ring, tail_pad])
        out_i16, _ = self.ov.convert_pcm_segment(
            window,
            src_se_key=self._live_se_key,
            tgt_se_path=tgt_se_path,
            tau=tau,
        )
        out_f32 = out_i16.astype(np.float32) / 32768.0

        # Emparejar volumen RMS del bloque convertido respecto a entrada para coherencia en llamada
        rms_in = float(np.sqrt(np.mean(np.square(self._ring)) + 1e-12))
        rms_conv = float(np.sqrt(np.mean(np.square(out_f32)) + 1e-12))
        if rms_conv > 1e-4 and rms_in > 1e-4:
            gain = max(0.35, min(2.4, rms_in / rms_conv))
            out_f32 = out_f32 * gain

        # Asegurar longitud mínima >= ctx_len + n_in para corte seguro sin deriva
        needed = ctx_len + n_in
        if len(out_f32) < needed:
            out_f32 = np.pad(out_f32, (0, needed - len(out_f32)), mode="constant")

        pre_overlap = out_f32[ctx_len - n_xf: ctx_len] if n_xf > 0 else np.zeros(0, dtype=np.float32)
        body_tail = out_f32[ctx_len: ctx_len + n_in]
        out_hop = self._crossfader.blend(pre_overlap, body_tail)
        out_hop = self._eq.process(out_hop, cfg)
        return _soft_limit_f32(out_hop)


class _OutRingBuffer:
    """Buffer FIFO protegido por lock entre el hilo de procesamiento y el callback de PortAudio."""

    def __init__(self, max_samples: int):
        self._lock = threading.Lock()
        self._buf: Deque[np.ndarray] = deque()
        self._total = 0
        self._max_samples = max_samples
        self.xruns = 0

    def push(self, chunk: np.ndarray) -> None:
        if chunk.size == 0:
            return
        arr = np.asarray(chunk, dtype=np.float32).copy()
        with self._lock:
            self._buf.append(arr)
            self._total += len(arr)
            # Evitar que el retardo crezca indefinidamente si el dispositivo de salida va lento
            while self._total > self._max_samples and self._buf:
                dropped = self._buf.popleft()
                self._total -= len(dropped)
                self.xruns += 1

    def pop(self, num_frames: int) -> np.ndarray:
        out = np.zeros(num_frames, dtype=np.float32)
        pos = 0
        with self._lock:
            while pos < num_frames and self._buf:
                head = self._buf[0]
                take = min(num_frames - pos, len(head))
                out[pos: pos + take] = head[:take]
                pos += take
                self._total -= take
                if take == len(head):
                    self._buf.popleft()
                else:
                    self._buf[0] = head[take:]
        if pos < num_frames and pos > 0:
            # Atenuar borde si hubo micro-underrun parcial
            fade = min(64, pos)
            out[pos - fade: pos] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
        return out

    def clear(self) -> None:
        with self._lock:
            self._buf.clear()
            self._total = 0


@dataclass
class LiveConfig:
    mode: str = "dsp"  # "dsp" | "neural"
    input_device: Optional[AudioDeviceInfo] = None
    output_device: Optional[AudioDeviceInfo] = None
    monitor_device: Optional[AudioDeviceInfo] = None
    monitor_enabled: bool = False
    monitor_volume: float = 0.70
    input_gain_db: float = 0.0
    noise_gate_db: float = -45.0
    tau: float = 0.45
    synth_settings: VoiceSynthSettings = field(default_factory=VoiceSynthSettings)
    target_se_path: Optional[Path] = None
    target_label: str = ""


@dataclass
class LiveTelemetry:
    running: bool = False
    mode: str = "dsp"
    input_level: float = 0.0
    output_level: float = 0.0
    is_speaking: bool = False
    proc_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    calibrating: bool = False
    calibrated_sec: float = 0.0
    recording: bool = False
    recorded_sec: float = 0.0
    xruns: int = 0
    status_text: str = "Detenido"
    error_text: Optional[str] = None


class LiveVoiceSession:
    """
    Sesión de audio dúplex completa para cambiar la voz en vivo hacia:
    - Cable virtual (Discord, Zoom, Google Meet, Teams, OBS, videojuegos) o altavoces.
    - Monitor opcional simultáneo en auriculares privados.
    - Grabación continua en vivo hacia WAV/MP3.
    """

    def __init__(
        self,
        config: LiveConfig,
        openvoice_engine=None,
        on_telemetry: Optional[Callable[[LiveTelemetry], None]] = None,
    ):
        self.config = config
        self.openvoice_engine = openvoice_engine
        self.on_telemetry = on_telemetry

        self._stop_event = threading.Event()
        self._param_lock = threading.Lock()
        self._worker_thread: Optional[threading.Thread] = None

        self._in_stream: Optional[sd.InputStream] = None
        self._out_stream: Optional[sd.OutputStream] = None
        self._mon_stream: Optional[sd.OutputStream] = None

        self._in_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=48)
        self._out_ring: Optional[_OutRingBuffer] = None
        self._mon_ring: Optional[_OutRingBuffer] = None

        self._processor = None
        self._gate: Optional[_NoiseGateEnvelope] = None

        self._in_sr = 48000
        self._out_sr = 48000
        self._mon_sr = 48000
        self._proc_sr = 24000

        self._rec_lock = threading.Lock()
        self._recording = False
        self._rec_chunks: List[np.ndarray] = []
        self._rec_samples = 0

        self._telemetry = LiveTelemetry(mode=config.mode)

    @property
    def is_running(self) -> bool:
        return self._worker_thread is not None and self._worker_thread.is_alive() and not self._stop_event.is_set()

    @property
    def is_recording(self) -> bool:
        return self._recording

    def get_telemetry(self) -> LiveTelemetry:
        return self._telemetry

    def _build_processor(self):
        if self.config.mode == "neural":
            if self.openvoice_engine is None or not self.openvoice_engine.is_loaded():
                raise RuntimeError("OpenVoice v2 no está cargado para el modo Neural En Vivo.")
            if not self.config.target_se_path or not Path(self.config.target_se_path).is_file():
                raise FileNotFoundError("Falta el perfil timbre (embedding) de la voz objetivo.")
            proc = LiveNeuralStreamProcessor(
                openvoice_engine=self.openvoice_engine,
                tgt_se_path=Path(self.config.target_se_path),
                synth_settings=self.config.synth_settings,
                tau=self.config.tau,
            )
            return proc
        return LiveDSPStreamProcessor(settings=self.config.synth_settings)

    @staticmethod
    def _open_stream_with_fallback(
        device: AudioDeviceInfo,
        kind: str,
        callback,
        blocksize: int,
    ):
        def _try_open(dev: AudioDeviceInfo, try_sr: int):
            if kind == "input":
                return sd.InputStream(
                    device=dev.index,
                    channels=1,
                    samplerate=try_sr,
                    blocksize=blocksize,
                    dtype="float32",
                    latency="low",
                    callback=callback,
                )
            return sd.OutputStream(
                device=dev.index,
                channels=1,
                samplerate=try_sr,
                blocksize=blocksize,
                dtype="float32",
                latency="low",
                callback=callback,
            )

        last_err: Optional[Exception] = None
        sr = device.default_samplerate or 48000
        for try_sr in dict.fromkeys((sr, 48000, 44100)):
            try:
                return _try_open(device, int(try_sr)), device, int(try_sr)
            except Exception as exc:
                last_err = exc

        fb = find_fallback_hostapi_device(device, kind=kind)
        if fb is not None:
            fb_sr = fb.default_samplerate or 48000
            for try_sr in dict.fromkeys((fb_sr, 48000, 44100)):
                try:
                    return _try_open(fb, int(try_sr)), fb, int(try_sr)
                except Exception as exc:
                    last_err = exc

        raise RuntimeError(f"No se pudo abrir dispositivo '{device.name}' ({kind}): {last_err}")

    def start(self) -> None:
        if self.is_running:
            return
        if self.config.input_device is None:
            raise RuntimeError("No hay ningún micrófono de entrada seleccionado.")
        if self.config.output_device is None:
            raise RuntimeError("No hay ningún dispositivo de salida seleccionado.")

        self._stop_event.clear()
        self._processor = self._build_processor()
        self._proc_sr = self._processor.sample_rate
        self._gate = _NoiseGateEnvelope(
            sample_rate=self._proc_sr,
            threshold_db=self.config.noise_gate_db,
        )

        # Limpiar colas previas
        while not self._in_queue.empty():
            try:
                self._in_queue.get_nowait()
            except queue.Empty:
                break

        hop_sec = self._processor.hop_samples / float(self._proc_sr)
        max_buf_sec = 0.65 if self.config.mode == "dsp" else 1.8

        def _in_cb(indata, frames, time_info, status):
            mono = indata[:, 0].copy() if indata.ndim > 1 else indata.copy()
            try:
                self._in_queue.put_nowait(mono)
            except queue.Full:
                try:
                    self._in_queue.get_nowait()
                    self._in_queue.put_nowait(mono)
                except Exception:
                    pass

        def _out_cb(outdata, frames, time_info, status):
            buf = self._out_ring.pop(frames) if self._out_ring is not None else np.zeros(frames, dtype=np.float32)
            outdata[:, 0] = buf

        def _mon_cb(outdata, frames, time_info, status):
            if self._mon_ring is None:
                outdata.fill(0.0)
                return
            buf = self._mon_ring.pop(frames)
            with self._param_lock:
                vol = float(self.config.monitor_volume) if self.config.monitor_enabled else 0.0
            outdata[:, 0] = np.clip(buf * vol, -0.98, 0.98)

        cb_frames_in = 512 if self.config.mode == "dsp" else 1024
        cb_frames_out = 512 if self.config.mode == "dsp" else 1024

        try:
            self._in_stream, used_in, self._in_sr = self._open_stream_with_fallback(
                self.config.input_device, "input", _in_cb, cb_frames_in
            )
            self._out_ring = _OutRingBuffer(int(max_buf_sec * (self.config.output_device.default_samplerate or 48000) * 2))
            self._out_stream, used_out, self._out_sr = self._open_stream_with_fallback(
                self.config.output_device, "output", _out_cb, cb_frames_out
            )
            self._out_ring._max_samples = int(max_buf_sec * self._out_sr)

            # Monitor secundario solo si está activado y apunta a dispositivo físico distinto al principal
            if (
                self.config.monitor_enabled
                and self.config.monitor_device is not None
                and self.config.monitor_device.index != used_out.index
            ):
                self._mon_ring = _OutRingBuffer(int(max_buf_sec * 48000))
                self._mon_stream, _, self._mon_sr = self._open_stream_with_fallback(
                    self.config.monitor_device, "output", _mon_cb, cb_frames_out
                )
                self._mon_ring._max_samples = int(max_buf_sec * self._mon_sr)
            else:
                self._mon_ring = None
                self._mon_stream = None

            self._worker_thread = threading.Thread(
                target=self._worker_loop,
                args=(hop_sec,),
                daemon=True,
                name="LiveVoiceWorker",
            )
            self._worker_thread.start()

            self._out_stream.start()
            if self._mon_stream is not None:
                self._mon_stream.start()
            self._in_stream.start()
        except Exception:
            self.stop()
            raise

    def _worker_loop(self, hop_sec: float) -> None:
        pending_in = np.zeros(0, dtype=np.float32)
        proc_hop_target = self._processor.hop_samples
        in_hop_target = max(128, int(round(hop_sec * self._in_sr)))
        ema_in = 0.0
        ema_out = 0.0
        ema_proc_ms = 0.0
        last_ui_push = 0.0

        while not self._stop_event.is_set():
            try:
                chunk = self._in_queue.get(timeout=0.05)
                pending_in = chunk if pending_in.size == 0 else np.concatenate([pending_in, chunk])
            except queue.Empty:
                continue

            while len(pending_in) >= in_hop_target and not self._stop_event.is_set():
                in_hop = pending_in[:in_hop_target]
                pending_in = pending_in[in_hop_target:]

                t0 = time.perf_counter()
                with self._param_lock:
                    gain_db = self.config.input_gain_db
                    gate_db = self.config.noise_gate_db
                    mode = self.config.mode
                    mon_enabled = self.config.monitor_enabled

                if self._gate is not None:
                    self._gate.set_threshold_db(gate_db)

                # Ganancia de entrada + nivel VU bruto
                if abs(gain_db) >= 0.1:
                    in_hop = _soft_limit_f32(in_hop * (10.0 ** (gain_db / 20.0)), knee=0.92)
                rms_in = float(np.sqrt(np.mean(np.square(in_hop)) + 1e-12))
                ema_in = 0.55 * ema_in + 0.45 * min(1.0, rms_in * 4.5)

                # Remuestrear entrada -> tasa interna exacta y ajustar tamaño al hop del procesador
                proc_in = _resample_f32(in_hop, self._in_sr, self._proc_sr)
                if len(proc_in) > proc_hop_target:
                    proc_in = proc_in[:proc_hop_target]
                elif len(proc_in) < proc_hop_target:
                    proc_in = np.pad(proc_in, (0, proc_hop_target - len(proc_in)), mode="constant")

                gated_in, gate_open, _ = self._gate.process_hop(proc_in)
                proc_out = self._processor.process_hop(gated_in, gate_open=gate_open)

                rms_out = float(np.sqrt(np.mean(np.square(proc_out)) + 1e-12))
                ema_out = 0.55 * ema_out + 0.45 * min(1.0, rms_out * 4.5)

                # Remuestrear salida -> dispositivo principal (y monitor si existe)
                out_pcm = _resample_f32(proc_out, self._proc_sr, self._out_sr)
                if self._out_ring is not None:
                    self._out_ring.push(out_pcm)

                if mon_enabled and self._mon_ring is not None:
                    mon_pcm = out_pcm if self._mon_sr == self._out_sr else _resample_f32(proc_out, self._proc_sr, self._mon_sr)
                    self._mon_ring.push(mon_pcm)

                # Grabar audio convertido si está activa la grabación en vivo
                if self._recording:
                    with self._rec_lock:
                        if self._recording and self._rec_samples < self._proc_sr * 3600:
                            self._rec_chunks.append(proc_out.copy())
                            self._rec_samples += len(proc_out)

                proc_ms = (time.perf_counter() - t0) * 1000.0
                ema_proc_ms = proc_ms if ema_proc_ms == 0.0 else (0.75 * ema_proc_ms + 0.25 * proc_ms)
                total_ms = (hop_sec * 1000.0) + ema_proc_ms + (
                    16.0 if mode == "dsp" else 41.0
                ) + 12.0

                now = time.perf_counter()
                if now - last_ui_push >= 0.06:
                    last_ui_push = now
                    calibrating = isinstance(self._processor, LiveNeuralStreamProcessor) and not self._processor.is_calibrated
                    calib_sec = self._processor.calibrated_seconds if isinstance(self._processor, LiveNeuralStreamProcessor) else 0.0
                    status_msg = (
                        f"🎯 Habla ~1s para calibrar timbre del mic ({calib_sec:.1f}s)..."
                        if calibrating
                        else ("🗣️ Transmitiendo voz modificada" if gate_open else "🔇 En espera (puerta de ruido cerrada)")
                    )
                    self._telemetry = LiveTelemetry(
                        running=True,
                        mode=mode,
                        input_level=ema_in,
                        output_level=ema_out,
                        is_speaking=gate_open,
                        proc_latency_ms=ema_proc_ms,
                        total_latency_ms=total_ms,
                        calibrating=calibrating,
                        calibrated_sec=calib_sec,
                        recording=self._recording,
                        recorded_sec=self._rec_samples / float(self._proc_sr),
                        xruns=self._out_ring.xruns if self._out_ring else 0,
                        status_text=status_msg,
                    )
                    if self.on_telemetry:
                        try:
                            self.on_telemetry(self._telemetry)
                        except Exception:
                            pass

    def update_synth_settings(self, settings: VoiceSynthSettings) -> None:
        with self._param_lock:
            self.config.synth_settings = settings
        if self._processor is not None:
            self._processor.update_settings(settings)

    def update_runtime_params(
        self,
        input_gain_db: Optional[float] = None,
        noise_gate_db: Optional[float] = None,
        monitor_enabled: Optional[bool] = None,
        monitor_volume: Optional[float] = None,
        tau: Optional[float] = None,
    ) -> None:
        with self._param_lock:
            if input_gain_db is not None:
                self.config.input_gain_db = float(input_gain_db)
            if noise_gate_db is not None:
                self.config.noise_gate_db = float(noise_gate_db)
            if monitor_enabled is not None:
                self.config.monitor_enabled = bool(monitor_enabled)
            if monitor_volume is not None:
                self.config.monitor_volume = float(max(0.0, min(1.0, monitor_volume)))
            if tau is not None:
                self.config.tau = float(tau)
        if tau is not None and isinstance(self._processor, LiveNeuralStreamProcessor):
            self._processor.update_tau(tau)

    def update_target_se(self, tgt_se_path: Path, target_label: str = "") -> None:
        with self._param_lock:
            self.config.target_se_path = Path(tgt_se_path)
            if target_label:
                self.config.target_label = target_label
        if isinstance(self._processor, LiveNeuralStreamProcessor):
            self._processor.update_target_se(tgt_se_path)

    def request_recalibration(self) -> None:
        if isinstance(self._processor, LiveNeuralStreamProcessor):
            self._processor.request_recalibration()

    def start_recording(self) -> None:
        with self._rec_lock:
            self._rec_chunks.clear()
            self._rec_samples = 0
            self._recording = True

    def stop_recording_and_export(
        self,
        output_dir: Path,
        export_format: str = "WAV",
    ) -> Tuple[Optional[Path], Optional[Path], float]:
        with self._rec_lock:
            self._recording = False
            chunks = list(self._rec_chunks)
            self._rec_chunks.clear()
            self._rec_samples = 0

        if not chunks:
            return None, None, 0.0

        audio_f32 = np.concatenate(chunks)
        dur_sec = float(len(audio_f32) / float(self._proc_sr))
        audio_i16 = np.clip(audio_f32 * 32767.0, -32768.0, 32767.0).astype(np.int16)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"vivo_{stamp}"

        wav_path: Optional[Path] = None
        mp3_path: Optional[Path] = None
        if "WAV" in export_format or export_format == "Ambos (WAV + MP3)":
            wav_path = output_dir / f"{base_name}.wav"
            AudioExporter.export_wav(audio_i16, self._proc_sr, wav_path)
        if "MP3" in export_format or export_format == "Ambos (WAV + MP3)":
            mp3_path = output_dir / f"{base_name}.mp3"
            AudioExporter.export_mp3(audio_i16, self._proc_sr, mp3_path)

        return wav_path, mp3_path, dur_sec

    def stop(self) -> None:
        self._stop_event.set()
        for stream_attr in ("_in_stream", "_out_stream", "_mon_stream"):
            stream = getattr(self, stream_attr, None)
            if stream is not None:
                try:
                    stream.stop()
                except Exception:
                    pass
                try:
                    stream.close()
                except Exception:
                    pass
                setattr(self, stream_attr, None)

        if self._worker_thread is not None and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2.0)
        self._worker_thread = None

        if self._out_ring is not None:
            self._out_ring.clear()
        if self._mon_ring is not None:
            self._mon_ring.clear()

        self._telemetry = LiveTelemetry(running=False, mode=self.config.mode, status_text="Detenido")
        if self.on_telemetry:
            try:
                self.on_telemetry(self._telemetry)
            except Exception:
                pass
