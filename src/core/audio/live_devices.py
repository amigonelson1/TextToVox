from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import sounddevice as sd


VBCABLE_DOWNLOAD_URL = "https://vb-audio.com/Cable/"

_HOSTAPI_RANK: Dict[str, int] = {
    "Windows WASAPI": 0,
    "Windows DirectSound": 1,
    "MME": 2,
    "Windows WDM-KS": 99,  # WDM-KS suele bloquear modo compartido en Windows
}

_HOSTAPI_SHORT: Dict[str, str] = {
    "Windows WASAPI": "WASAPI",
    "Windows DirectSound": "DirectSound",
    "MME": "MME",
    "Windows WDM-KS": "WDM-KS",
}

_GENERIC_ALIAS_PREFIXES = (
    "asignador de sonido",
    "microsoft sound mapper",
    "controlador primario",
    "primary sound",
)

_VIRTUAL_OUTPUT_HINTS = (
    "cable input",
    "vb-audio virtual cable",
    "vbcable",
    "voicemeeter input",
    "voicemeeter aux input",
    "voicemeeter vaio",
    "virtual audio cable",
    "virtual cable",
    "virtual mic",
)

_VIRTUAL_ANY_HINTS = _VIRTUAL_OUTPUT_HINTS + (
    "cable output",
    "voicemeeter output",
    "voicemeeter aux output",
)

_LOW_PRIORITY_HINTS = (
    "hands-free",
    "hf audio",
    "stereo mix",
    "mezcla estéreo",
    "wave (",
)


@dataclass(frozen=True)
class AudioDeviceInfo:
    index: int
    name: str
    hostapi_index: int
    hostapi_name: str
    max_input_channels: int
    max_output_channels: int
    default_samplerate: int
    default_low_input_latency: float
    default_low_output_latency: float
    is_virtual_cable: bool
    is_default_input: bool = False
    is_default_output: bool = False

    @property
    def hostapi_short(self) -> str:
        return _HOSTAPI_SHORT.get(self.hostapi_name, self.hostapi_name)

    @property
    def display_label(self) -> str:
        prefix = "🔌 [Cable Virtual] " if self.is_virtual_cable else ""
        return f"{prefix}{self.name} [{self.hostapi_short}]"


def _normalize_device_key(name: str) -> str:
    cleaned = re.sub(r"\s+", " ", name.strip().lower())
    # MME trunca nombres a 31 caracteres; usamos prefijo común de 28 para dedup limpio
    return cleaned[:28]


def _hostapi_rank(hostapi_name: str) -> int:
    return _HOSTAPI_RANK.get(hostapi_name, 50)


def _is_generic_alias(name: str) -> bool:
    low = name.strip().lower()
    return any(low.startswith(p) for p in _GENERIC_ALIAS_PREFIXES)


def _is_virtual_name(name: str, output_only: bool = False) -> bool:
    low = name.strip().lower()
    hints = _VIRTUAL_OUTPUT_HINTS if output_only else _VIRTUAL_ANY_HINTS
    return any(h in low for h in hints)


def _query_raw_devices(force_reinit: bool = False) -> Tuple[List[AudioDeviceInfo], Optional[int], Optional[int]]:
    if force_reinit:
        try:
            sd._terminate()
            sd._initialize()
        except Exception:
            pass

    hostapis = sd.query_hostapis()
    raw_devices = sd.query_devices()

    default_in_idx: Optional[int] = None
    default_out_idx: Optional[int] = None
    try:
        def_pair = sd.default.device
        if isinstance(def_pair, (list, tuple)) and len(def_pair) >= 2:
            default_in_idx = int(def_pair[0]) if def_pair[0] is not None and def_pair[0] >= 0 else None
            default_out_idx = int(def_pair[1]) if def_pair[1] is not None and def_pair[1] >= 0 else None
    except Exception:
        pass

    # Preferir defaults de WASAPI si existen
    for ha in hostapis:
        if ha.get("name") == "Windows WASAPI":
            if isinstance(ha.get("default_input_device"), int) and ha["default_input_device"] >= 0:
                default_in_idx = int(ha["default_input_device"])
            if isinstance(ha.get("default_output_device"), int) and ha["default_output_device"] >= 0:
                default_out_idx = int(ha["default_output_device"])
            break

    items: List[AudioDeviceInfo] = []
    for idx, dev in enumerate(raw_devices):
        ha_idx = int(dev.get("hostapi", 0))
        ha_name = str(hostapis[ha_idx].get("name", "Unknown")) if 0 <= ha_idx < len(hostapis) else "Unknown"
        name = str(dev.get("name", f"Device {idx}")).strip()
        in_ch = int(dev.get("max_input_channels", 0))
        out_ch = int(dev.get("max_output_channels", 0))
        sr = int(round(float(dev.get("default_samplerate", 48000) or 48000)))
        items.append(
            AudioDeviceInfo(
                index=idx,
                name=name,
                hostapi_index=ha_idx,
                hostapi_name=ha_name,
                max_input_channels=in_ch,
                max_output_channels=out_ch,
                default_samplerate=max(8000, sr),
                default_low_input_latency=float(dev.get("default_low_input_latency", 0.02) or 0.02),
                default_low_output_latency=float(dev.get("default_low_output_latency", 0.02) or 0.02),
                is_virtual_cable=_is_virtual_name(name, output_only=False),
                is_default_input=(idx == default_in_idx),
                is_default_output=(idx == default_out_idx),
            )
        )
    return items, default_in_idx, default_out_idx


def _filter_and_rank(items: List[AudioDeviceInfo], kind: str) -> List[AudioDeviceInfo]:
    if kind == "input":
        candidates = [d for d in items if d.max_input_channels > 0]
    elif kind == "output":
        candidates = [d for d in items if d.max_output_channels > 0]
    else:
        raise ValueError(f"Tipo de dispositivo inválido: {kind}")

    # Evitar WDM-KS y alias genéricos si hay dispositivos reales
    no_wdm = [d for d in candidates if _hostapi_rank(d.hostapi_name) < 90]
    if no_wdm:
        candidates = no_wdm
    no_aliases = [d for d in candidates if not _is_generic_alias(d.name)]
    if no_aliases:
        candidates = no_aliases

    # Deduplicar por nombre físico conservando la API de menor latencia (WASAPI > DirectSound > MME)
    best_by_key: Dict[str, AudioDeviceInfo] = {}
    for d in candidates:
        key = _normalize_device_key(d.name)
        prev = best_by_key.get(key)
        if prev is None:
            best_by_key[key] = d
            continue
        prev_score = (
            _hostapi_rank(prev.hostapi_name),
            0 if (prev.is_default_input if kind == "input" else prev.is_default_output) else 1,
        )
        cur_score = (
            _hostapi_rank(d.hostapi_name),
            0 if (d.is_default_input if kind == "input" else d.is_default_output) else 1,
        )
        if cur_score < prev_score:
            best_by_key[key] = d

    deduped = list(best_by_key.values())

    def _sort_key(d: AudioDeviceInfo) -> Tuple[int, int, int, int, str]:
        low_name = d.name.lower()
        is_virt = 0 if (kind == "output" and _is_virtual_name(d.name, output_only=True)) else 1
        is_def = 0 if (d.is_default_input if kind == "input" else d.is_default_output) else 1
        is_low_prio = 1 if any(h in low_name for h in _LOW_PRIORITY_HINTS) else 0
        return (is_virt, is_def, is_low_prio, _hostapi_rank(d.hostapi_name), d.name.lower())

    deduped.sort(key=_sort_key)
    return deduped


def list_audio_devices(force_reinit: bool = False) -> Tuple[List[AudioDeviceInfo], List[AudioDeviceInfo]]:
    """Lista dispositivos filtrados y priorizados (WASAPI primero) para entrada y salida."""
    items, _, _ = _query_raw_devices(force_reinit=force_reinit)
    return _filter_and_rank(items, "input"), _filter_and_rank(items, "output")


def detect_virtual_cable_device(outputs: Optional[List[AudioDeviceInfo]] = None) -> Optional[AudioDeviceInfo]:
    """Detecta un endpoint de salida tipo VB-Cable / VoiceMeeter para enrutar voz a Discord/Zoom/Juegos."""
    if outputs is None:
        _, outputs = list_audio_devices()
    virt_outputs = [d for d in outputs if _is_virtual_name(d.name, output_only=True)]
    if not virt_outputs:
        return None
    for preferred in ("cable input", "voicemeeter input"):
        for d in virt_outputs:
            if preferred in d.name.lower():
                return d
    return virt_outputs[0]


def resolve_device_by_label(
    label_or_name: Optional[str],
    candidates: List[AudioDeviceInfo],
    prefer_virtual: bool = False,
) -> Optional[AudioDeviceInfo]:
    """Resuelve un AudioDeviceInfo a partir de etiqueta guardada, subcadena o fallback por defecto."""
    if not candidates:
        return None

    target = (label_or_name or "").strip()
    if target:
        for d in candidates:
            if d.display_label == target or d.name == target:
                return d
        norm_t = _normalize_device_key(re.sub(r"^🔌\s*\[Cable Virtual\]\s*", "", target))
        norm_t = re.sub(r"\s*\[[^\]]+\]\s*$", "", norm_t).strip()
        for d in candidates:
            if norm_t and norm_t in _normalize_device_key(d.name):
                return d

    if prefer_virtual:
        vc = detect_virtual_cable_device(candidates)
        if vc is not None:
            return vc

    for d in candidates:
        if d.is_default_input or d.is_default_output:
            return d
    return candidates[0]


def default_live_devices() -> Tuple[Optional[AudioDeviceInfo], Optional[AudioDeviceInfo], Optional[AudioDeviceInfo]]:
    """Retorna (mic_entrada, salida_principal, salida_monitor_auriculares)."""
    inputs, outputs = list_audio_devices()
    mic = resolve_device_by_label(None, inputs, prefer_virtual=False)
    virtual = detect_virtual_cable_device(outputs)
    physical_out = next((d for d in outputs if d.is_default_output and not d.is_virtual_cable), None)
    if physical_out is None:
        physical_out = next((d for d in outputs if not d.is_virtual_cable), outputs[0] if outputs else None)
    primary_out = virtual if virtual is not None else physical_out
    monitor_out = physical_out
    return mic, primary_out, monitor_out


def find_fallback_hostapi_device(device: AudioDeviceInfo, kind: str) -> Optional[AudioDeviceInfo]:
    """Si WASAPI falla por formato exclusivo/BT, busca el mismo dispositivo físico vía DirectSound o MME."""
    items, _, _ = _query_raw_devices(force_reinit=False)
    key = _normalize_device_key(device.name)
    alts = [
        d for d in items
        if d.index != device.index
        and _normalize_device_key(d.name) == key
        and ((kind == "input" and d.max_input_channels > 0) or (kind == "output" and d.max_output_channels > 0))
        and _hostapi_rank(d.hostapi_name) < 90
    ]
    alts.sort(key=lambda d: _hostapi_rank(d.hostapi_name))
    return alts[0] if alts else None
