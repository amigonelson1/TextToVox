import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Dict, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def find_clone_python() -> Optional[Path]:
    """Localiza el ejecutable Python del entorno dedicado de clonación (.venv_clone)."""
    candidates = [
        PROJECT_ROOT / ".venv_clone" / "Scripts" / "python.exe",
        PROJECT_ROOT / ".venv_clone" / "bin" / "python",
    ]
    for cand in candidates:
        if cand.is_file():
            return cand
    return None


class CloneWorkerClient:
    """
    Cliente persistente por JSON-Lines para ejecutar OpenVoice v2 y Coqui XTTS-v2
    en el entorno aislado (.venv_clone) sin conflictos de dependencias ni fugas de RAM.
    """

    def __init__(self):
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._aborted = False

    @staticmethod
    def is_environment_ready() -> bool:
        return find_clone_python() is not None

    def _ensure_started(self) -> subprocess.Popen:
        if self._proc is not None and self._proc.poll() is None:
            return self._proc

        py_exe = find_clone_python()
        if py_exe is None:
            raise RuntimeError(
                "No se encontró el entorno de clonación (.venv_clone). "
                "Ejecuta 'instalar_clonacion.bat' para habilitar la clonación de voz."
            )

        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUNBUFFERED"] = "1"
        env["COQUI_TOS_AGREED"] = "1"

        creationflags = 0
        if os.name == "nt" and hasattr(subprocess, "CREATE_NO_WINDOW"):
            creationflags = subprocess.CREATE_NO_WINDOW

        worker_script = PROJECT_ROOT / "src" / "core" / "cloning" / "clone_worker.py"
        self._aborted = False
        self._proc = subprocess.Popen(
            [str(py_exe), "-u", str(worker_script)],
            cwd=str(PROJECT_ROOT),
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            creationflags=creationflags,
        )
        return self._proc

    def send_command(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Envía un comando JSON y retorna la respuesta síncrona del worker."""
        with self._lock:
            proc = self._ensure_started()
            assert proc.stdin is not None and proc.stdout is not None

            line = json.dumps(payload, ensure_ascii=False)
            try:
                proc.stdin.write(line + "\n")
                proc.stdin.flush()
            except Exception as exc:
                self._cleanup_proc()
                raise RuntimeError(f"Se perdió conexión con el motor de clonación: {exc}") from exc

            while True:
                resp_line = proc.stdout.readline()
                if not resp_line:
                    stderr_tail = ""
                    if proc.stderr is not None:
                        try:
                            stderr_tail = proc.stderr.read().strip()[-1200:]
                        except Exception:
                            pass
                    was_aborted = self._aborted
                    self._cleanup_proc()
                    if was_aborted:
                        raise InterruptedError("Generación cancelada por el usuario.")
                    raise RuntimeError(
                        f"El proceso de clonación finalizó inesperadamente.\n{stderr_tail}"
                    )

                resp_line = resp_line.strip()
                if not resp_line:
                    continue

                try:
                    data = json.loads(resp_line)
                except json.JSONDecodeError:
                    # Ignorar cualquier línea suelta que no sea JSON válido
                    continue

                if not data.get("ok", False):
                    raise RuntimeError(data.get("error", "Error desconocido en el worker de clonación."))
                return data

    def abort_current(self) -> None:
        """Detiene inmediatamente el worker actual para cancelar una síntesis larga en CPU."""
        proc = self._proc
        self._aborted = True
        if proc is not None and proc.poll() is None:
            try:
                proc.kill()
            except Exception:
                pass

    def unload_and_shutdown(self) -> None:
        """Libera memoria descargando modelos y cerrando el proceso worker."""
        proc = self._proc
        if proc is None:
            return
        if proc.poll() is None:
            try:
                if proc.stdin:
                    proc.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
                    proc.stdin.flush()
                proc.wait(timeout=2.0)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        self._cleanup_proc()

    def _cleanup_proc(self) -> None:
        if self._proc is not None:
            for stream in (self._proc.stdin, self._proc.stdout, self._proc.stderr):
                if stream is not None:
                    try:
                        stream.close()
                    except Exception:
                        pass
            self._proc = None
