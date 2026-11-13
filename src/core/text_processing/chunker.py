import re
from dataclasses import dataclass
from typing import List


DEFAULT_PREVIEW_TEXT = (
    "Hola, prueba rápida de voz. Escucha este ritmo.\n\n"
    "Así suena la pausa entre párrafos."
)


def _trim_sentence_prefix(paragraph: str, max_chars: int) -> str:
    """Extrae el inicio natural de un párrafo respetando el límite de caracteres."""
    text = " ".join(paragraph.split())
    if len(text) <= max_chars:
        return text

    sentences = re.split(r"(?<=[.?!;:])\s+", text)
    acc = ""
    for sent in sentences:
        candidate = f"{acc} {sent}".strip() if acc else sent
        if len(candidate) <= max_chars:
            acc = candidate
        else:
            break
    if acc:
        return acc

    cut = text[:max_chars].rsplit(" ", 1)[0].strip()
    return (cut or text[:max_chars]).rstrip(",;:") + "..."


def build_preview_text(raw_text: str, max_chars: int = 190) -> str:
    """
    Construye un texto breve de previsualización (<= ~190 caracteres):
    - Si el editor está vacío, usa una muestra corta con pausa entre oración y párrafo.
    - Si el editor tiene 2+ párrafos, toma un extracto de los dos primeros párrafos
      manteniendo el salto entre párrafos para apreciar la pausa configurada.
    - Si tiene 1 párrafo, toma sus primeras oraciones o hasta max_chars.
    """
    cleaned = (raw_text or "").strip()
    if not cleaned:
        return DEFAULT_PREVIEW_TEXT

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", cleaned) if p.strip()]
    if len(paragraphs) >= 2:
        per_para_budget = max(60, (max_chars - 2) // 2)
        p1 = _trim_sentence_prefix(paragraphs[0], per_para_budget)
        p2 = _trim_sentence_prefix(paragraphs[1], per_para_budget)
        return f"{p1}\n\n{p2}"

    return _trim_sentence_prefix(paragraphs[0], max_chars)


@dataclass
class TextChunk:
    """Representa un fragmento de texto procesable por el motor TTS."""
    text: str
    is_paragraph_end: bool = False
    index: int = 0


class SmartTextChunker:
    """
    Divide textos de cualquier longitud en fragmentos óptimos para síntesis neural.
    - Respeta límites de oraciones (. ? ! ; :).
    - Evita cortar en medio de números decimales (ej. 3.14).
    - Evita fragmentos excesivamente largos (> max_chars).
    - Distingue pausas de oración vs pausas de párrafo.
    """

    def __init__(self, max_chars: int = 250, min_chars: int = 20):
        self.max_chars = max_chars
        self.min_chars = min_chars

    def split(self, text: str) -> List[TextChunk]:
        """Divide el texto en una lista ordenada de TextChunk."""
        cleaned = text.strip()
        if not cleaned:
            return []

        # 1. Separar en párrafos
        paragraphs = re.split(r"\n\s*\n+", cleaned)
        all_chunks: List[TextChunk] = []
        chunk_idx = 0

        for p_idx, para in enumerate(paragraphs):
            para = para.strip()
            if not para:
                continue

            # 2. Separar párrafos en oraciones
            # Divide solo cuando el signo (. ? ! ; :) va seguido de un espacio o salto de línea
            sentences = re.split(r"(?<=[.?!;:])\s+", para)

            para_subchunks: List[str] = []
            for sent in sentences:
                sent = sent.strip()
                if not sent:
                    continue

                if len(sent) <= self.max_chars:
                    para_subchunks.append(sent)
                else:
                    # Si la oración es muy larga, subdividir por comas
                    comma_parts = re.split(r"(?<=[,])\s+", sent)
                    current_sub = ""
                    for part in comma_parts:
                        if not current_sub:
                            current_sub = part
                        elif len(current_sub) + len(part) + 1 <= self.max_chars:
                            current_sub += " " + part
                        else:
                            para_subchunks.append(current_sub)
                            current_sub = part
                    if current_sub:
                        # Si todavía excede max_chars, cortar por palabras
                        if len(current_sub) > self.max_chars:
                            words = current_sub.split()
                            word_acc = ""
                            for w in words:
                                if not word_acc:
                                    word_acc = w
                                elif len(word_acc) + len(w) + 1 <= self.max_chars:
                                    word_acc += " " + w
                                else:
                                    para_subchunks.append(word_acc)
                                    word_acc = w
                            if word_acc:
                                para_subchunks.append(word_acc)
                        else:
                            para_subchunks.append(current_sub)

            # Empacar en TextChunk
            for i, chunk_text in enumerate(para_subchunks):
                is_last_in_para = (i == len(para_subchunks) - 1)
                all_chunks.append(
                    TextChunk(
                        text=chunk_text,
                        is_paragraph_end=is_last_in_para,
                        index=chunk_idx
                    )
                )
                chunk_idx += 1

        return all_chunks

