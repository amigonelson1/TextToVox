import hashlib
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import soundfile as sf
import librosa
import torch

from src.core.cloning.openvoice_runtime import utils, models
from src.core.cloning.openvoice_runtime.mel_processing import spectrogram_torch


class OpenVoiceConverter:
    """
    Wrapper ligero y 100% local alrededor de ToneColorConverter (OpenVoice v2).
    Se ejecuta en el entorno con PyTorch y no requiere Whisper ni modelos de marca de agua.
    """

    def __init__(self, checkpoint_dir: Path, device: str = "cpu"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.config_path = self.checkpoint_dir / "config.json"
        self.ckpt_path = self.checkpoint_dir / "checkpoint.pth"
        self.device = device

        if not self.config_path.is_file() or not self.ckpt_path.is_file():
            raise FileNotFoundError(
                f"Faltan archivos de OpenVoice v2 en {self.checkpoint_dir}. "
                "Se requieren config.json y checkpoint.pth."
            )

        self.hps = utils.get_hparams_from_file(str(self.config_path))
        self.sample_rate = int(self.hps.data.sampling_rate)

        self.model = models.SynthesizerTrn(
            len(getattr(self.hps, "symbols", [])),
            self.hps.data.filter_length // 2 + 1,
            n_speakers=self.hps.data.n_speakers,
            **vars(self.hps.model) if hasattr(self.hps.model, "__dict__") else self.hps.model,
        ).to(self.device)
        self.model.eval()

        checkpoint_dict = torch.load(
            str(self.ckpt_path),
            map_location=torch.device(self.device),
            weights_only=False
        )
        self.model.load_state_dict(checkpoint_dict["model"], strict=False)

    @staticmethod
    def load_audio_clean(audio_path: Path, target_sr: int) -> np.ndarray:
        """Carga WAV/MP3/OGG/FLAC a mono float32 normalizado."""
        try:
            audio, orig_sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            if orig_sr != target_sr:
                audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
        except Exception:
            audio, _ = librosa.load(str(audio_path), sr=target_sr, mono=True)
        return np.asarray(audio, dtype=np.float32)

    @classmethod
    def extract_voiced_segments(
        cls,
        audio_path: Path,
        target_sr: int,
        max_total_seconds: float = 45.0,
        segment_seconds: float = 10.0,
        min_duration_sec: float = 2.0,
    ) -> Tuple[List[np.ndarray], float]:
        """Elimina silencios largos y divide la voz activa en segmentos consistentes."""
        audio = cls.load_audio_clean(audio_path, target_sr)
        if audio.size == 0:
            raise ValueError("El audio está vacío.")

        # Normalizar nivel de entrada antes de detección de actividad
        peak = float(np.max(np.abs(audio)))
        if peak < 1e-4:
            raise ValueError("El audio no tiene volumen audible.")
        audio = audio / max(peak, 1e-6) * 0.9

        # Extraer tramos con voz activa
        intervals = librosa.effects.split(audio, top_db=28, frame_length=1024, hop_length=256)
        if len(intervals) > 0:
            voiced = np.concatenate([audio[s:e] for s, e in intervals])
        else:
            voiced = audio

        duration_sec = float(len(voiced) / target_sr)
        if duration_sec < min_duration_sec:
            raise ValueError(
                f"El audio útil dura solo {duration_sec:.1f}s tras quitar silencios "
                f"(mínimo requerido: {min_duration_sec:.1f}s)."
            )

        max_samples = int(max_total_seconds * target_sr)
        if len(voiced) > max_samples:
            voiced = voiced[:max_samples]
            duration_sec = max_total_seconds

        seg_samples = max(int(segment_seconds * target_sr), target_sr * 3)
        segments = [
            voiced[i:i + seg_samples]
            for i in range(0, len(voiced), seg_samples)
            if len(voiced[i:i + seg_samples]) >= int(min(1.5, min_duration_sec) * target_sr)
        ]
        if not segments:
            segments = [voiced]

        return segments, duration_sec

    @staticmethod
    def file_fingerprint(path: Path) -> str:
        stat = Path(path).stat()
        key = f"{Path(path).resolve()}|{stat.st_size}|{stat.st_mtime_ns}"
        return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]

    def _segments_from_array(
        self,
        audio_f32: np.ndarray,
        orig_sr: int,
        max_total_seconds: float = 45.0,
        segment_seconds: float = 10.0,
        min_duration_sec: float = 2.0,
    ) -> Tuple[List[np.ndarray], float]:
        audio = np.asarray(audio_f32, dtype=np.float32)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        if int(orig_sr) != int(self.sample_rate):
            audio = librosa.resample(audio, orig_sr=int(orig_sr), target_sr=int(self.sample_rate))
        if audio.size == 0:
            raise ValueError("El audio está vacío.")

        peak = float(np.max(np.abs(audio)))
        if peak < 1e-4:
            raise ValueError("El audio no tiene volumen audible.")
        audio = audio / max(peak, 1e-6) * 0.9

        intervals = librosa.effects.split(audio, top_db=28, frame_length=1024, hop_length=256)
        voiced = np.concatenate([audio[s:e] for s, e in intervals]) if len(intervals) > 0 else audio

        duration_sec = float(len(voiced) / self.sample_rate)
        if duration_sec < min_duration_sec:
            raise ValueError(
                f"El audio útil dura solo {duration_sec:.1f}s tras quitar silencios "
                f"(mínimo requerido: {min_duration_sec:.1f}s)."
            )

        max_samples = int(max_total_seconds * self.sample_rate)
        if len(voiced) > max_samples:
            voiced = voiced[:max_samples]
            duration_sec = max_total_seconds

        seg_samples = max(int(segment_seconds * self.sample_rate), self.sample_rate * 3)
        segments = [
            voiced[i:i + seg_samples]
            for i in range(0, len(voiced), seg_samples)
            if len(voiced[i:i + seg_samples]) >= int(min(1.5, min_duration_sec) * self.sample_rate)
        ]
        if not segments:
            segments = [voiced]

        return segments, duration_sec

    def _compute_embedding_from_segments(self, segments: List[np.ndarray]) -> torch.Tensor:
        embeddings = []
        for seg in segments:
            y = torch.from_numpy(seg).float().unsqueeze(0).to(self.device)
            spec = spectrogram_torch(
                y,
                self.hps.data.filter_length,
                self.hps.data.sampling_rate,
                self.hps.data.hop_length,
                self.hps.data.win_length,
                center=False,
            ).to(self.device)
            with torch.no_grad():
                g = self.model.ref_enc(spec.transpose(1, 2)).unsqueeze(-1)
                embeddings.append(g.detach())
        return torch.stack(embeddings).mean(0)

    def extract_embedding_from_array(
        self,
        audio_f32: np.ndarray,
        orig_sr: Optional[int] = None,
        min_duration_sec: float = 0.4,
    ) -> Tuple[torch.Tensor, float]:
        """Extrae speaker embedding directamente desde un array en memoria."""
        segments, duration_sec = self._segments_from_array(
            audio_f32=audio_f32,
            orig_sr=int(orig_sr or self.sample_rate),
            min_duration_sec=min_duration_sec,
        )
        return self._compute_embedding_from_segments(segments), duration_sec

    def extract_speaker_embedding(
        self,
        audio_path: Path,
        se_cache_path: Optional[Path] = None,
        min_duration_sec: float = 2.0,
    ) -> Tuple[torch.Tensor, float]:
        """Extrae o carga desde caché el vector de timbre vocal (speaker embedding)."""
        if se_cache_path and Path(se_cache_path).is_file():
            gs = torch.load(str(se_cache_path), map_location=self.device, weights_only=True)
            return gs.to(self.device), 0.0

        segments, duration_sec = self.extract_voiced_segments(
            audio_path,
            self.sample_rate,
            min_duration_sec=min_duration_sec,
        )
        gs = self._compute_embedding_from_segments(segments)

        if se_cache_path:
            se_cache_path = Path(se_cache_path)
            se_cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(gs.cpu(), str(se_cache_path))

        return gs, duration_sec

    def convert_array(
        self,
        audio_f32: np.ndarray,
        src_se: torch.Tensor,
        tgt_se: torch.Tensor,
        tau: float = 0.3,
    ) -> np.ndarray:
        """Convierte un fragmento mono float32 (a self.sample_rate) en memoria sin escribir en disco."""
        audio = np.asarray(audio_f32, dtype=np.float32)
        if audio.size == 0:
            raise ValueError("El audio base generado está vacío.")

        peak_in = float(np.max(np.abs(audio)))
        if peak_in > 0.99:
            audio = audio / peak_in * 0.95

        y = torch.from_numpy(audio).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            spec = spectrogram_torch(
                y,
                self.hps.data.filter_length,
                self.hps.data.sampling_rate,
                self.hps.data.hop_length,
                self.hps.data.win_length,
                center=False,
            ).to(self.device)
            spec_lengths = torch.LongTensor([spec.size(-1)]).to(self.device)
            out_audio = self.model.voice_conversion(
                spec,
                spec_lengths,
                sid_src=src_se.to(self.device),
                sid_tgt=tgt_se.to(self.device),
                tau=float(tau),
            )[0][0, 0].detach().cpu().float().numpy()

        return np.clip(out_audio, -0.99, 0.99).astype(np.float32)

    def convert(
        self,
        src_audio_path: Path,
        src_se: torch.Tensor,
        tgt_se: torch.Tensor,
        output_wav_path: Path,
        tau: float = 0.3,
    ) -> int:
        """Convierte el timbre del audio base al timbre objetivo conservando idioma, acento y prosodia."""
        audio = self.load_audio_clean(src_audio_path, self.sample_rate)
        out_audio = self.convert_array(audio, src_se=src_se, tgt_se=tgt_se, tau=tau)
        output_wav_path = Path(output_wav_path)
        output_wav_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_wav_path), out_audio, self.sample_rate, subtype="PCM_16")
        return self.sample_rate

