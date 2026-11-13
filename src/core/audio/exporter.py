import wave
from pathlib import Path
from typing import Union
import numpy as np
import lameenc


class AudioExporter:
    """Exporta arrays de audio PCM a formatos WAV y MP3 en disco."""

    @staticmethod
    def export_wav(
        audio_data: Union[np.ndarray, bytes],
        sample_rate: int,
        output_path: Union[str, Path],
        channels: int = 1
    ) -> Path:
        """Exporta el array de audio PCM o buffer de bytes a un archivo WAV."""
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(audio_data, bytes):
            audio_int16 = np.frombuffer(audio_data, dtype=np.int16)
        elif audio_data.dtype != np.int16:
            audio_int16 = (audio_data * 32767).astype(np.int16)
        else:
            audio_int16 = audio_data

        with wave.open(str(out_p), "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sample_rate)
            wf.writeframes(audio_int16.tobytes())

        return out_p

    @staticmethod
    def export_mp3(
        audio_data: Union[np.ndarray, bytes],
        sample_rate: int,
        output_path: Union[str, Path],
        channels: int = 1,
        bitrate_kbps: int = 192,
        quality: int = 2
    ) -> Path:
        """
        Exporta el array de audio PCM o buffer a un archivo MP3 utilizando lameenc directamente,
        sin requerir binarios externos de FFmpeg.
        """
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(audio_data, bytes):
            audio_int16 = np.frombuffer(audio_data, dtype=np.int16)
        elif audio_data.dtype != np.int16:
            audio_int16 = (audio_data * 32767).astype(np.int16)
        else:
            audio_int16 = audio_data

        encoder = lameenc.Encoder()
        encoder.set_bit_rate(bitrate_kbps)
        encoder.set_in_sample_rate(sample_rate)
        encoder.set_channels(channels)
        encoder.set_quality(quality)  # 2: Excelente calidad acústica

        raw_pcm = audio_int16.tobytes()
        mp3_data = encoder.encode(raw_pcm) + encoder.flush()

        out_p.write_bytes(mp3_data)
        return out_p

