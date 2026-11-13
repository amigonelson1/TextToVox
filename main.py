import sys
import logging
from pathlib import Path

# Agregar directorio actual al sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.ui.app_window import TexToVoxApp


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    logging.info("Iniciando texToVox...")
    app = TexToVoxApp()
    app.mainloop()


if __name__ == "__main__":
    main()

