import uuid
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import soundfile as sf

from src.core.engine.base import BaseTTSEngine, AudioChunkData
from src.core.engine.clone_worker_client import CloneWorkerClient


class XTTSv2Engine(BaseTTSEngine):
    """
    Motor de clonación de voz 100% offline basado en Coqui XTTS-v2 (uso personal CPML).
    Genera voz en español directamente a partir del clip de referencia sin internet.
    """

    def __init__(
        self,
        worker_client: Optional[CloneWorkerClient] = None,
        cache_dir: Path = Path("models/cloning/cache"),
    ):
        self.worker = worker_client or CloneWorkerClient()
        self.cache_dir = Path(cache_dir)
        self.tmp_dir = self.cache_dir / "_tmp"
        self.tmp_dir.mkdir(parents=True, exist_ok=True)

        self.model_dir: Optional[Path] = None
        self.reference_audio: Optional[Path] = None
        self.language: str = "es"
        self._sample_rate: int = 24000
        self._is_loaded: bool = False

    def load_model(self, model_path: Path, config_path: Optional[Path] = None) -> None:
        ckpt_dir = Path(model_path)
        required = ("model.pth", "dvae.pth", "vocab.json", "config.json")
        if not all((ckpt_dir / fname).is_file() for fname in required):
            raise FileNotFoundError(
                f"Modelo XTTS-v2 incompleto en {ckpt_dir}. Descárgalo primero."
            )

        resp = self.worker.send_command({
            "cmd": "load_xtts",
            "model_dir": str(ckpt_dir.resolve()),
        })
        self._sample_rate = int(resp.get("sample_rate", 24000))
        self.model_dir = ckpt_dir
        self._is_loaded = True

    def set_reference_audio(self, reference_audio: Path, language: str = "es") -> None:
        self.reference_audio = Path(reference_audio)
        self.language = language

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
            raise RuntimeError("XTTS-v2 no está cargado.")

        if self.reference_audio is None or not self.reference_audio.is_file():
            raise FileNotFoundError("Selecciona primero un archivo de audio de referencia válido (.wav/.mp3/.ogg/.flac).")

        token = uuid.uuid4().hex[:10]
        out_wav = self.tmp_dir / f"xtts_out_{token}.wav"

        try:
            resp = self.worker.send_command({
                "cmd": "synthesize_xtts",
                "text": text_clean,
                "ref_wav": str(self.reference_audio.resolve()),
                "language": self.language,
                "speed": float(speed),
                "out_wav": str(out_wav.resolve()),
            })
            self._sample_rate = int(resp.get("sample_rate", 24000))

            pcm_data, sr = sf.read(str(out_wav), dtype="int16", always_2d=False)
            if pcm_data.ndim > 1:
                pcm_data = pcm_data[:, 0]
            self._sample_rate = int(sr)

            return [
                AudioChunkData(
                    audio_bytes=np.asarray(pcm_data, dtype=np.int16).tobytes(),
                    sample_rate=self._sample_rate,
                    sample_width=2,
                    channels=1,
                )
            ]
        finally:
            try:
                if out_wav.exists():
                    out_wav.unlink()
            except Exception:
                pass

    def is_loaded(self) -> bool:
        return self._is_loaded

    def get_sample_rate(self) -> int:
        return self._sample_rate

    def get_speakers(self) -> Dict[int, str]:
        return {0: "Voz Clonada (XTTS-v2)"}

