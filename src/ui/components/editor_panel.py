import customtkinter as ctk
from tkinter import filedialog, messagebox
from typing import Callable, Optional


class EditorPanel(ctk.CTkFrame):
    """Panel de edición de texto con contadores dinámicos y herramientas de carga rápida."""

    def __init__(self, master, on_text_change: Optional[Callable[[], None]] = None, **kwargs):
        super().__init__(master, **kwargs)
        self.on_text_change = on_text_change

        # Encabezado
        self.header_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.header_frame.pack(fill="x", padx=10, pady=(5, 5))

        self.title_lbl = ctk.CTkLabel(
            self.header_frame,
            text="📝 Texto a Convertir",
            font=ctk.CTkFont(size=15, weight="bold")
        )
        self.title_lbl.pack(side="left")

        # Botones rápidos
        self.btn_sample = ctk.CTkButton(
            self.header_frame,
            text="Ejemplo",
            width=70,
            height=26,
            command=self._load_sample_text
        )
        self.btn_sample.pack(side="right", padx=(5, 0))

        self.btn_open = ctk.CTkButton(
            self.header_frame,
            text="Cargar .txt",
            width=80,
            height=26,
            command=self._open_file
        )
        self.btn_open.pack(side="right", padx=(5, 0))

        self.btn_clear = ctk.CTkButton(
            self.header_frame,
            text="Limpiar",
            width=70,
            height=26,
            fg_color="gray40",
            hover_color="gray30",
            command=self.clear_text
        )
        self.btn_clear.pack(side="right", padx=(5, 0))

        # Cuadro de texto principal
        self.textbox = ctk.CTkTextbox(
            self,
            font=ctk.CTkFont(size=14),
            wrap="word",
            undo=True
        )
        self.textbox.pack(fill="both", expand=True, padx=10, pady=(0, 5))
        self.textbox.bind("<KeyRelease>", self._on_key_release)

        # Barra inferior de métricas
        self.stats_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.stats_frame.pack(fill="x", padx=10, pady=(0, 5))

        self.lbl_stats = ctk.CTkLabel(
            self.stats_frame,
            text="0 palabras | 0 caracteres | Duración estimada: 00:00",
            font=ctk.CTkFont(size=12),
            text_color="gray60"
        )
        self.lbl_stats.pack(side="left")

    def _on_key_release(self, event=None):
        self.update_stats()
        if self.on_text_change:
            self.on_text_change()

    def update_stats(self):
        text = self.get_text().strip()
        chars = len(text)
        words = len(text.split()) if text else 0

        # Estimación: promedio en español ~135 palabras por minuto
        est_seconds = int((words / 135.0) * 60) if words > 0 else 0
        mins, secs = divmod(est_seconds, 60)
        est_str = f"{mins:02d}:{secs:02d}"

        self.lbl_stats.configure(
            text=f"{words:,} palabras | {chars:,} caracteres | Duración est.: ~{est_str}"
        )

    def get_text(self) -> str:
        return self.textbox.get("1.0", "end-1c")

    def set_text(self, text: str):
        self.textbox.delete("1.0", "end")
        self.textbox.insert("1.0", text)
        self.update_stats()
        if self.on_text_change:
            self.on_text_change()

    def clear_text(self):
        text = self.get_text().strip()
        if not text:
            return

        confirm = messagebox.askyesno(
            title="Confirmar limpieza",
            message="¿Estás seguro de que deseas borrar todo el texto del editor?",
            icon="warning"
        )
        if confirm:
            self.textbox.delete("1.0", "end")
            self.update_stats()
            if self.on_text_change:
                self.on_text_change()

    def _open_file(self):
        file_path = filedialog.askopenfilename(
            title="Seleccionar archivo de texto",
            filetypes=[("Archivos de texto", "*.txt *.md *.log"), ("Todos los archivos", "*.*")]
        )
        if file_path:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
            except UnicodeDecodeError:
                with open(file_path, "r", encoding="latin-1") as f:
                    content = f.read()
            self.set_text(content)

    def _load_sample_text(self):
        sample = (
            "Bienvenido a texToVox. Esta es una solución local en Python "
            "para sintetizar voz neural con alta fidelidad y naturalidad, "
            "completamente offline.\n\n"
            "Puedes redactar o pegar guiones completos, audiolibros, artículos "
            "o notas extensas. El procesador dividirá el texto en fragmentos "
            "automáticos, preservando las pausas de oraciones y párrafos.\n\n"
            "¡Elige una voz de la lista, ajusta la velocidad y presiona Generar Audio!"
        )
        self.set_text(sample)

