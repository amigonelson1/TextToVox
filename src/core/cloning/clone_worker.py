"""
Worker persistente en Python 3.12 (.venv_clone) para ejecutar OpenVoice v2 y Coqui XTTS-v2
mediante protocolo JSON-Lines por stdin/stdout sin bloquear ni contaminar el entorno principal.
"""
import base64
import gc
import json
import os
import sys
import traceback
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Garantizar que cualquier print()/log de librerías vaya a stderr y NUNCA rompa el JSONL por stdout
_PROTO_STDOUT = sys.stdout
sys.stdout = sys.stderr

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ.setdefault("COQUI_TOS_AGREED", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import librosa  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402


class CloneWorkerState:
    def __init__(self):
        self.device = "cpu"
        self.ov_converter = None
        self.ov_ckpt_dir: Optional[str] = None
        self.ov_se_cache: Dict[str, torch.Tensor] = {}

        self.xtts_model = None
        self.xtts_config = None
        self.xtts_dir: Optional[str] = None
        self.xtts_latents_cache: Dict[str, Tuple[Any, Any]] = {}

    @staticmethod
    def _file_key(path: Path) -> str:
        st = path.stat()
        return f"{path.resolve()}|{st.st_size}|{st.st_mtime_ns}"

    @staticmethod
    def _load_audio_soundfile(audiopath: Any, sampling_rate: int) -> torch.Tensor:
        """Reemplazo puro con soundfile+librosa que evita dependencias externas de DLLs FFmpeg/torchcodec."""
        path_str = str(audiopath)
        try:
            audio, orig_sr = sf.read(path_str, dtype="float32", always_2d=False)
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            if int(orig_sr) != int(sampling_rate):
                audio = librosa.resample(audio, orig_sr=int(orig_sr), target_sr=int(sampling_rate))
        except Exception:
            audio, _ = librosa.load(path_str, sr=int(sampling_rate), mono=True)

        audio = np.asarray(audio, dtype=np.float32)
        if audio.size == 0:
            raise ValueError(f"El archivo de audio está vacío: {path_str}")
        peak = float(np.max(np.abs(audio)))
        if peak > 1.0:
            audio = audio / peak * 0.98
        return torch.from_numpy(audio).unsqueeze(0)

    def handle_load_openvoice(self, req: Dict[str, Any]) -> Dict[str, Any]:
        from src.core.cloning.openvoice_converter import OpenVoiceConverter

        ckpt_dir = str(Path(req["ckpt_dir"]).resolve())
        if self.ov_converter is None or self.ov_ckpt_dir != ckpt_dir:
            self.ov_converter = OpenVoiceConverter(Path(ckpt_dir), device=self.device)
            self.ov_ckpt_dir = ckpt_dir

        return {"ok": True, "sample_rate": self.ov_converter.sample_rate}

    def handle_extract_openvoice_se(self, req: Dict[str, Any]) -> Dict[str, Any]:
        if self.ov_converter is None:
            raise RuntimeError("OpenVoice no está cargado.")

        audio_path = Path(req["audio_path"])
        se_cache_path = Path(req["se_cache_path"]) if req.get("se_cache_path") else None
        min_duration_sec = float(req.get("min_duration_sec", 2.0))
        gs, duration_sec = self.ov_converter.extract_speaker_embedding(
            audio_path=audio_path,
            se_cache_path=se_cache_path,
            min_duration_sec=min_duration_sec,
        )
        if se_cache_path:
            self.ov_se_cache[str(se_cache_path.resolve())] = gs
        return {"ok": True, "duration_sec": duration_sec}

    def handle_convert_openvoice(self, req: Dict[str, Any]) -> Dict[str, Any]:
        if self.ov_converter is None:
            raise RuntimeError("OpenVoice no está cargado.")

        src_se_path = str(Path(req["src_se_path"]).resolve())
        tgt_se_path = str(Path(req["tgt_se_path"]).resolve())

        if src_se_path not in self.ov_se_cache:
            self.ov_se_cache[src_se_path] = torch.load(
                src_se_path, map_location=self.device, weights_only=True
            )
        if tgt_se_path not in self.ov_se_cache:
            self.ov_se_cache[tgt_se_path] = torch.load(
                tgt_se_path, map_location=self.device, weights_only=True
            )

        sr = self.ov_converter.convert(
            src_audio_path=Path(req["src_wav"]),
            src_se=self.ov_se_cache[src_se_path],
            tgt_se=self.ov_se_cache[tgt_se_path],
            output_wav_path=Path(req["out_wav"]),
            tau=float(req.get("tau", 0.35)),
        )
        return {"ok": True, "sample_rate": sr}

    def handle_extract_openvoice_se_pcm(self, req: Dict[str, Any]) -> Dict[str, Any]:
        if self.ov_converter is None:
            raise RuntimeError("OpenVoice no está cargado.")

        pcm_bytes = base64.b64decode(req["pcm_b64"])
        audio_f32 = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        se_key = str(req["se_key"])
        orig_sr = int(req.get("sample_rate", self.ov_converter.sample_rate))
        min_duration_sec = float(req.get("min_duration_sec", 0.4))

        gs, duration_sec = self.ov_converter.extract_embedding_from_array(
            audio_f32=audio_f32,
            orig_sr=orig_sr,
            min_duration_sec=min_duration_sec,
        )
        self.ov_se_cache[se_key] = gs
        return {"ok": True, "se_key": se_key, "duration_sec": duration_sec}

    def handle_convert_openvoice_pcm(self, req: Dict[str, Any]) -> Dict[str, Any]:
        if self.ov_converter is None:
            raise RuntimeError("OpenVoice no está cargado.")

        src_se_key = str(req["src_se_key"])
        if src_se_key not in self.ov_se_cache:
            if Path(src_se_key).is_file():
                self.ov_se_cache[src_se_key] = torch.load(
                    src_se_key, map_location=self.device, weights_only=True
                )
            else:
                raise KeyError(f"Speaker embedding no calibrado en memoria: {src_se_key}")

        tgt_se_path = str(Path(req["tgt_se_path"]).resolve())
        if tgt_se_path not in self.ov_se_cache:
            self.ov_se_cache[tgt_se_path] = torch.load(
                tgt_se_path, map_location=self.device, weights_only=True
            )

        pcm_bytes = base64.b64decode(req["pcm_b64"])
        audio_f32 = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        out_f32 = self.ov_converter.convert_array(
            audio_f32=audio_f32,
            src_se=self.ov_se_cache[src_se_key],
            tgt_se=self.ov_se_cache[tgt_se_path],
            tau=float(req.get("tau", 0.45)),
        )
        out_i16 = np.clip(out_f32 * 32767.0, -32768.0, 32767.0).astype(np.int16)
        out_b64 = base64.b64encode(out_i16.tobytes()).decode("ascii")
        return {"ok": True, "sample_rate": self.ov_converter.sample_rate, "pcm_b64": out_b64}

    def handle_load_xtts(self, req: Dict[str, Any]) -> Dict[str, Any]:
        import torchaudio
        import TTS.tts.models.xtts as xtts_mod
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts

        # Parchar carga de audio sin depender de DLLs externas de FFmpeg/torchcodec en Windows
        xtts_mod.load_audio = self._load_audio_soundfile

        def _safe_torchaudio_load(path_in, *args, **kwargs):
            data, sr = sf.read(str(path_in), dtype="float32", always_2d=True)
            return torch.from_numpy(data.T.copy()), int(sr)

        torchaudio.load = _safe_torchaudio_load

        model_dir = str(Path(req["model_dir"]).resolve())
        if self.xtts_model is None or self.xtts_dir != model_dir:
            cfg = XttsConfig()
            cfg.load_json(str(Path(model_dir) / "config.json"))
            model = Xtts.init_from_config(cfg)
            model.load_checkpoint(cfg, checkpoint_dir=model_dir, eval=True)
            model.to(self.device)
            self.xtts_config = cfg
            self.xtts_model = model
            self.xtts_dir = model_dir
            self.xtts_latents_cache.clear()

        sr = int(getattr(self.xtts_config.audio, "output_sample_rate", 24000))
        return {"ok": True, "sample_rate": sr}

    def handle_synthesize_xtts(self, req: Dict[str, Any]) -> Dict[str, Any]:
        if self.xtts_model is None:
            raise RuntimeError("XTTS-v2 no está cargado.")

        ref_path = Path(req["ref_wav"])
        if not ref_path.is_file():
            raise FileNotFoundError(f"Audio de referencia no encontrado: {ref_path}")

        cache_key = self._file_key(ref_path)
        if cache_key not in self.xtts_latents_cache:
            if len(self.xtts_latents_cache) >= 8:
                self.xtts_latents_cache.clear()
            gpt_cond_latent, speaker_embedding = self.xtts_model.get_conditioning_latents(
                audio_path=[str(ref_path)],
                max_ref_length=25,
                gpt_cond_len=12,
                gpt_cond_chunk_len=6,
                sound_norm_refs=True,
                librosa_trim_db=None,
            )
            self.xtts_latents_cache[cache_key] = (gpt_cond_latent, speaker_embedding)

        gpt_cond_latent, speaker_embedding = self.xtts_latents_cache[cache_key]

        out = self.xtts_model.inference(
            text=req["text"],
            language=req.get("language", "es"),
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            temperature=0.65,
            length_penalty=1.0,
            repetition_penalty=5.0,
            top_k=50,
            top_p=0.85,
            speed=float(req.get("speed", 1.0)),
            enable_text_splitting=False,
        )

        wav = np.asarray(out["wav"], dtype=np.float32)
        wav = np.clip(wav, -0.99, 0.99)
        sr = int(getattr(self.xtts_config.audio, "output_sample_rate", 24000))

        out_wav = Path(req["out_wav"])
        out_wav.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_wav), wav, sr, subtype="PCM_16")
        return {"ok": True, "sample_rate": sr}

    def handle_unload(self) -> Dict[str, Any]:
        self.ov_converter = None
        self.ov_ckpt_dir = None
        self.ov_se_cache.clear()
        self.xtts_model = None
        self.xtts_config = None
        self.xtts_dir = None
        self.xtts_latents_cache.clear()
        gc.collect()
        return {"ok": True}


def _reply(obj: Dict[str, Any]) -> None:
    _PROTO_STDOUT.write(json.dumps(obj, ensure_ascii=False) + "\n")
    _PROTO_STDOUT.flush()


def main() -> None:
    state = CloneWorkerState()
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            cmd = req.get("cmd", "")
            if cmd == "ping":
                _reply({"ok": True, "pong": True})
            elif cmd == "load_openvoice":
                _reply(state.handle_load_openvoice(req))
            elif cmd == "extract_openvoice_se":
                _reply(state.handle_extract_openvoice_se(req))
            elif cmd == "convert_openvoice":
                _reply(state.handle_convert_openvoice(req))
            elif cmd == "extract_openvoice_se_pcm":
                _reply(state.handle_extract_openvoice_se_pcm(req))
            elif cmd == "convert_openvoice_pcm":
                _reply(state.handle_convert_openvoice_pcm(req))
            elif cmd == "load_xtts":
                _reply(state.handle_load_xtts(req))
            elif cmd == "synthesize_xtts":
                _reply(state.handle_synthesize_xtts(req))
            elif cmd == "unload":
                _reply(state.handle_unload())
            elif cmd == "quit":
                _reply({"ok": True})
                break
            else:
                _reply({"ok": False, "error": f"Comando desconocido: {cmd}"})
        except Exception as exc:
            err_detail = f"{exc}\n{traceback.format_exc()[-600:]}"
            _reply({"ok": False, "error": str(exc), "detail": err_detail})


if __name__ == "__main__":
    main()

