import hashlib
import json
import re
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional


SUPPORTED_REFERENCE_EXTS = (".wav", ".mp3", ".ogg", ".flac")


def _sanitize_name(name: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", name.strip())
    return re.sub(r"\s+", "_", cleaned).strip("._") or "voz"


@dataclass
class ClonedVoiceProfile:
    """Representa una muestra de voz de referencia guardada localmente para clonación."""
    profile_id: str
    name: str
    audio_path: str
    sha256: str = ""
    notes: str = ""

    @property
    def exists(self) -> bool:
        return bool(self.audio_path) and Path(self.audio_path).is_file()


class ClonedProfileManager:
    """Administra perfiles locales de voces clonadas (archivos de referencia en disco)."""

    def __init__(self, profiles_dir: Path = Path("models/cloning/profiles")):
        self.profiles_dir = Path(profiles_dir)
        self.audio_dir = self.profiles_dir / "audio"
        self.index_path = self.profiles_dir / "profiles.json"
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        self.audio_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _file_sha256(path: Path) -> str:
        hasher = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def load_profiles(self) -> List[ClonedVoiceProfile]:
        if not self.index_path.exists():
            return []
        try:
            data = json.loads(self.index_path.read_text(encoding="utf-8"))
            profiles = [
                ClonedVoiceProfile(**{k: v for k, v in item.items() if k in ClonedVoiceProfile.__dataclass_fields__})
                for item in data
            ]
            return [p for p in profiles if p.exists]
        except Exception:
            return []

    def save_profiles(self, profiles: List[ClonedVoiceProfile]) -> None:
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        payload = [asdict(p) for p in profiles]
        self.index_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def import_profile(self, source_audio: Path, display_name: Optional[str] = None) -> ClonedVoiceProfile:
        """Importa y copia un archivo de referencia (.wav/.mp3/.ogg/.flac) al catálogo local."""
        src = Path(source_audio)
        if not src.is_file():
            raise FileNotFoundError(f"No se encontró el archivo de audio: {src}")

        ext = src.suffix.lower()
        if ext not in SUPPORTED_REFERENCE_EXTS:
            raise ValueError(
                f"Formato no soportado ({ext}). Usa {', '.join(SUPPORTED_REFERENCE_EXTS)}."
            )

        sha = self._file_sha256(src)
        profiles = self.load_profiles()

        # Si ya existe exactamente ese mismo archivo por hash, reutilizarlo
        for existing in profiles:
            if existing.sha256 == sha and existing.exists:
                if display_name and display_name.strip():
                    existing.name = display_name.strip()
                    self.save_profiles(profiles)
                return existing

        clean_name = (display_name or src.stem).strip() or "Voz Personal"
        safe_stem = _sanitize_name(clean_name)
        profile_id = f"{safe_stem}_{sha[:8]}"
        dest_audio = self.audio_dir / f"{profile_id}{ext}"
        shutil.copy2(src, dest_audio)

        profile = ClonedVoiceProfile(
            profile_id=profile_id,
            name=clean_name,
            audio_path=str(dest_audio.resolve()),
            sha256=sha
        )
        profiles.append(profile)
        self.save_profiles(profiles)
        return profile

    def remove_profile(self, profile_id: str) -> bool:
        profiles = self.load_profiles()
        kept: List[ClonedVoiceProfile] = []
        removed: Optional[ClonedVoiceProfile] = None
        for p in profiles:
            if p.profile_id == profile_id:
                removed = p
            else:
                kept.append(p)

        if removed is None:
            return False

        self.save_profiles(kept)
        audio_p = Path(removed.audio_path)
        if audio_p.is_file() and audio_p.parent.resolve() == self.audio_dir.resolve():
            try:
                audio_p.unlink()
            except Exception:
                pass
        return True
