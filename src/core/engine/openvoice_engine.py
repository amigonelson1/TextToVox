import base64
import io
import uuid
import wave
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import soundfile as sf

from src.core.engine.base import BaseTTSEngine, AudioChunkData
from src.core.engine.clone_worker_client import CloneWorkerClient
from src.core.engine.edge_engine import EdgeTTSEngine
from src.core.engine.piper_engine import PiperEngine
from src.core.voice_manager.voice_info import VoiceInfo


CALIBRATION_TEXT_ES = (
    "Hola, esta es una muestra de referencia acústica en español para calibrar "
    "el tono, las vocales claras, la dicción pausada y el timbre vocal natural."
)


class OpenVoiceEngine(BaseTTSEngine):
    """
    Motor de clonación híbrida para español latino natural:
    1) Genera la dicción, pausas y acento auténtico usando una voz base latina (Edge-TTS o Piper).
    2) Transfiere el timbre del audio de referencia mediante OpenVoice v2 (ToneColorConverter).
    """

    def __init__(
        self,
        worker_client: Optional[CloneWorkerClient] = None,
        cache_dir: Path = Path("models/cloning/cache"),
    ):
        self.worker = worker_client or CloneWorkerClient()
        self.cache_dir = Path(cache_dir)
        self.tmp_dir = self.cache_dir / "_tmp"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.tmp_dir.mkdir(parents=True, exist_ok=True)

        self.model_dir: Optional[Path] = None
        self.base_voice: Optional[VoiceInfo] = None
        self.reference_audio: Optional[Path] = None
        self.tau: float = 0.35

        self._edge_engine = EdgeTTSEngine()
        self._piper_engine = PiperEngine()
        self._loaded_piper_key: Optional[str] = None
        self._sample_rate: int = 22050
        self._is_loaded: bool = False

        self._cached_src_se_path: Optional[Path] = None
        self._cached_tgt_se_path: Optional[Path] = None

    def load_model(self, model_path: Path, config_path: Optional[Path] = None) -> None:
        ckpt_dir = Path(model_path)
        if not (ckpt_dir / "checkpoint.pth").is_file() or not (ckpt_dir / "config.json").is_file():
            raise FileNotFoundError(
                f"Modelo OpenVoice v2 incompleto en {ckpt_dir}. Descárgalo primero."
            )
        resp = self.worker.send_command({
            "cmd": "load_openvoice",
            "ckpt_dir": str(ckpt_dir.resolve()),
        })
        self._sample_rate = int(resp.get("sample_rate", 22050))
        self.model_dir = ckpt_dir
        self._is_loaded = True

    def configure_cloning(
        self,
        base_voice: VoiceInfo,
        reference_audio: Path,
        tau: float = 0.35,
    ) -> None:
        self.base_voice = base_voice
        self.reference_audio = Path(reference_audio)
        self.tau = float(max(0.1, min(0.85, tau)))

    def _synthesize_voice_to_wav(self, voice: VoiceInfo, text: str, speed: float, dest_wav: Path) -> None:
        if voice.engine_type == "edge":
            self._edge_engine.set_voice(voice.voice_identifier)
            chunks = self._edge_engine.synthesize(text=text, speed=speed)
        elif voice.engine_type == "piper":
            if not voice.is_installed or not voice.model_path:
                raise RuntimeError(f"La voz base offline '{voice.name}' no está instalada.")
            if self._loaded_piper_key != voice.key or not self._piper_engine.is_loaded():
                self._piper_engine.load_model(voice.model_path, voice.config_path)
                self._loaded_piper_key = voice.key
            chunks = self._piper_engine.synthesize(text=text, speed=speed)
        else:
            raise ValueError(f"Motor base no soportado para timbre de voz: {voice.engine_type}")

        if not chunks:
            raise RuntimeError("La voz base no generó audio para este fragmento.")

        sr = chunks[0].sample_rate
        raw_bytes = b"".join(c.audio_bytes for c in chunks)
        dest_wav.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(dest_wav), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(raw_bytes)

    def _synthesize_base_to_wav(self, text: str, speed: float, dest_wav: Path) -> None:
        if self.base_voice is None:
            raise RuntimeError("No se configuró una voz base latina para OpenVoice.")
        self._synthesize_voice_to_wav(self.base_voice, text=text, speed=speed, dest_wav=dest_wav)

    def ensure_catalog_voice_embedding(self, voice: VoiceInfo) -> Path:
        """Extrae y cachea el embedding vocal de cualquier voz Edge o Piper instalada."""
        safe_key = voice.key.replace("/", "_")
        cal_wav = self.cache_dir / f"base_{safe_key}.wav"
        se_path = self.cache_dir / f"base_{safe_key}.se.pt"
        if not se_path.is_file():
            if not cal_wav.is_file():
                self._synthesize_voice_to_wav(voice, CALIBRATION_TEXT_ES, speed=1.0, dest_wav=cal_wav)
            self.worker.send_command({
                "cmd": "extract_openvoice_se",
                "audio_path": str(cal_wav.resolve()),
                "se_cache_path": str(se_path.resolve()),
                "min_duration_sec": 1.0,
            })
        return se_path

    def ensure_audio_embedding(
        self,
        audio_path: Path,
        prefix: str = "ref",
        min_duration_sec: float = 2.0,
    ) -> Path:
        """Extrae y cachea el embedding vocal de un archivo de audio usando huella de tamaño/mtime."""
        src = Path(audio_path)
        if not src.is_file():
            raise FileNotFoundError(f"No se encontró el archivo de audio: {src}")
        stat = src.stat()
        tag = f"{src.stem}_{stat.st_size}_{stat.st_mtime_ns}"
        safe_tag = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in tag)
        se_path = self.cache_dir / f"{prefix}_{safe_tag}.se.pt"
        if not se_path.is_file():
            self.worker.send_command({
                "cmd": "extract_openvoice_se",
                "audio_path": str(src.resolve()),
                "se_cache_path": str(se_path.resolve()),
                "min_duration_sec": float(min_duration_sec),
            })
        return se_path

    def convert_wav_segment(
        self,
        src_wav: Path,
        src_se_path: Path,
        tgt_se_path: Path,
        tau: Optional[float] = None,
    ) -> tuple[np.ndarray, int]:
        """Convierte un fragmento WAV usando embeddings ya extraídos y retorna PCM int16 + sample rate."""
        if not self._is_loaded:
            raise RuntimeError("OpenVoice v2 no está cargado.")

        token = uuid.uuid4().hex[:10]
        out_wav = self.tmp_dir / f"ov_vc_{token}.wav"
        try:
            resp = self.worker.send_command({
                "cmd": "convert_openvoice",
                "src_wav": str(Path(src_wav).resolve()),
                "src_se_path": str(Path(src_se_path).resolve()),
                "tgt_se_path": str(Path(tgt_se_path).resolve()),
                "out_wav": str(out_wav.resolve()),
                "tau": float(self.tau if tau is None else tau),
            })
            self._sample_rate = int(resp.get("sample_rate", self._sample_rate))
            pcm_data, sr = sf.read(str(out_wav), dtype="int16", always_2d=False)
            if pcm_data.ndim > 1:
                pcm_data = pcm_data[:, 0]
            self._sample_rate = int(sr)
            return np.asarray(pcm_data, dtype=np.int16), self._sample_rate
        finally:
            try:
                if out_wav.exists():
                    out_wav.unlink()
            except Exception:
                pass

    def extract_pcm_embedding(
        self,
        pcm_audio: np.ndarray,
        sample_rate: int,
        se_key: str = "live_mic",
        min_duration_sec: float = 0.4,
    ) -> str:
        """Extrae en RAM el embedding del micrófono en vivo sin escribir archivos temporales."""
        if not self._is_loaded:
            raise RuntimeError("OpenVoice v2 no está cargado.")

        arr = np.asarray(pcm_audio)
        if arr.dtype != np.int16:
            arr = np.clip(arr.astype(np.float32) * 32767.0, -32768.0, 32767.0).astype(np.int16)

        resp = self.worker.send_command({
            "cmd": "extract_openvoice_se_pcm",
            "pcm_b64": base64.b64encode(arr.tobytes()).decode("ascii"),
            "sample_rate": int(sample_rate),
            "se_key": se_key,
            "min_duration_sec": float(min_duration_sec),
        })
        return str(resp.get("se_key", se_key))

    def convert_pcm_segment(
        self,
        pcm_audio: np.ndarray,
        src_se_key: str,
        tgt_se_path: Path,
        tau: Optional[float] = None,
    ) -> tuple[np.ndarray, int]:
        """Convierte un segmento PCM int16/float32 directamente en RAM para streaming en vivo."""
        if not self._is_loaded:
            raise RuntimeError("OpenVoice v2 no está cargado.")

        arr = np.asarray(pcm_audio)
        if arr.dtype != np.int16:
            arr = np.clip(arr.astype(np.float32) * 32767.0, -32768.0, 32767.0).astype(np.int16)

        resp = self.worker.send_command({
            "cmd": "convert_openvoice_pcm",
            "pcm_b64": base64.b64encode(arr.tobytes()).decode("ascii"),
            "src_se_key": src_se_key,
            "tgt_se_path": str(Path(tgt_se_path).resolve()),
            "tau": float(self.tau if tau is None else tau),
        })
        self._sample_rate = int(resp.get("sample_rate", self._sample_rate))
        out_bytes = base64.b64decode(resp["pcm_b64"])
        out_i16 = np.frombuffer(out_bytes, dtype=np.int16).copy()
        return out_i16, self._sample_rate

    def _ensure_speaker_embeddings(self) -> tuple[Path, Path]:
        if self.base_voice is None:
            raise RuntimeError("Falta seleccionar una voz base latina.")
        if self.reference_audio is None or not self.reference_audio.is_file():
            raise FileNotFoundError("Selecciona primero un archivo de audio de referencia válido (.wav/.mp3/.ogg/.flac).")

        src_se_path = self.ensure_catalog_voice_embedding(self.base_voice)
        tgt_se_path = self.ensure_audio_embedding(self.reference_audio, prefix="ref", min_duration_sec=2.0)

        self._cached_src_se_path = src_se_path
        self._cached_tgt_se_path = tgt_se_path
        return src_se_path, tgt_se_path

    def synthesize(
        self,
        text: str,
        speed: float = 1.0,
        speaker_id: Optional[int] = None
    ) -> List[AudioChunkData]:
        text_clean = text.strip()
        if not text_clean:
            return []

        if not self._is_loaded:
            raise RuntimeError("OpenVoice v2 no está cargado.")

        src_se_path, tgt_se_path = self._ensure_speaker_embeddings()
        token = uuid.uuid4().hex[:10]
        src_wav = self.tmp_dir / f"ov_src_{token}.wav"

        try:
            self._synthesize_base_to_wav(text_clean, speed=speed, dest_wav=src_wav)
            pcm_data, sr = self.convert_wav_segment(src_wav, src_se_path, tgt_se_path, tau=self.tau)

            return [
                AudioChunkData(
                    audio_bytes=pcm_data.tobytes(),
                    sample_rate=sr,
                    sample_width=2,
                    channels=1,
                )
            ]
        finally:
            try:
                if src_wav.exists():
                    src_wav.unlink()
            except Exception:
                pass

    def is_loaded(self) -> bool:
        return self._is_loaded

    def get_sample_rate(self) -> int:
        return self._sample_rate

    def get_speakers(self) -> Dict[int, str]:
        return {0: "Voz Clonada (OpenVoice v2)"}

