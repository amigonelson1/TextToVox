import re
from typing import Dict

# Diccionario de abreviaturas comunes en español
SPANISH_ABBREVIATIONS: Dict[str, str] = {
    r"\bDr\.(?=\s|$)": "Doctor",
    r"\bDra\.(?=\s|$)": "Doctora",
    r"\bSr\.(?=\s|$)": "Señor",
    r"\bSra\.(?=\s|$)": "Señora",
    r"\bSrta\.(?=\s|$)": "Señorita",
    r"\bProf\.(?=\s|$)": "Profesor",
    r"\bProfa\.(?=\s|$)": "Profesora",
    r"\bpág\.(?=\s|$)": "página",
    r"\bpágs\.(?=\s|$)": "páginas",
    r"\bej\.(?=\s|$)": "ejemplo",
    r"\betc\.(?=\s|$)": "etcétera",
    r"\bart\.(?=\s|$)": "artículo",
    r"\bcap\.(?=\s|$)": "capítulo",
    r"\bnúm\.(?=\s|$)": "número",
    r"\bNo\.(?=\s|$)": "número",
    r"\bno\.(?=\s|$)": "número",
    r"\bkm/h\b": "kilómetros por hora",
    r"\bkm\b": "kilómetros",
    r"\bcm\b": "centímetros",
    r"\bmm\b": "milímetros",
    r"\bkg\b": "kilogramos",
    r"\bmg\b": "miligramos",
    r"\bvol\.(?=\s|$)": "volumen",
    r"\bed\.(?=\s|$)": "edición",
    r"\baprox\.(?=\s|$)": "aproximadamente",
    r"\bEE\.UU\.(?=\s|$)": "Estados Unidos",
    r"\bEEUU\b": "Estados Unidos",
    r"\bCía\.(?=\s|$)": "Compañía",
    r"\bAv\.(?=\s|$)": "Avenida",
    r"\bCra\.(?=\s|$)": "Carrera",
    r"\bCl\.(?=\s|$)": "Calle",
    r"\bDiag\.(?=\s|$)": "Diagonal",
    r"\bTransv\.(?=\s|$)": "Transversal",
    r"\bdepto\.(?=\s|$)": "departamento",
}


def int_to_spanish(n: int) -> str:
    """Convierte un entero positivo (hasta 999.999.999) a su representación en español hablado."""
    unidades = ['', 'uno', 'dos', 'tres', 'cuatro', 'cinco', 'seis', 'siete', 'ocho', 'nueve', 'diez',
                'once', 'doce', 'trece', 'catorce', 'quince', 'dieciséis', 'diecisiete', 'dieciocho', 'diecinueve']
    decenas = ['', '', 'veinte', 'treinta', 'cuarenta', 'cincuenta', 'sesenta', 'setenta', 'ochenta', 'noventa']
    centenas = ['', 'ciento', 'doscientos', 'trescientos', 'cuatrocientos', 'quinientos', 'seiscientos', 'setecientos', 'ochocientos', 'novecientos']

    if n == 0:
        return 'cero'
    if n == 20:
        return 'veinte'
    if n == 100:
        return 'cien'
    if n < 20:
        return unidades[n]
    if n < 30:
        if n == 21:
            return 'veintiuno'
        elif n == 22:
            return 'veintidós'
        elif n == 23:
            return 'veintitrés'
        elif n == 26:
            return 'veintiséis'
        return 'veinti' + unidades[n % 10]
    if n < 100:
        return decenas[n // 10] + ('' if n % 10 == 0 else ' y ' + unidades[n % 10])
    if n < 1000:
        return centenas[n // 100] + ('' if n % 100 == 0 else ' ' + int_to_spanish(n % 100))
    if n < 2000:
        return 'mil' + ('' if n % 1000 == 0 else ' ' + int_to_spanish(n % 1000))
    if n < 1000000:
        mil_parte = int_to_spanish(n // 1000)
        resto = n % 1000
        return f"{mil_parte} mil" + ('' if resto == 0 else ' ' + int_to_spanish(resto))
    if n < 2000000:
        resto = n % 1000000
        return 'un millón' + ('' if resto == 0 else ' ' + int_to_spanish(resto))
    if n < 1000000000:
        mill_parte = int_to_spanish(n // 1000000)
        resto = n % 1000000
        return f"{mill_parte} millones" + ('' if resto == 0 else ' ' + int_to_spanish(resto))
    return str(n)


def normalize_spanish_text(text: str) -> str:
    """
    Normaliza el texto en español para una mejor síntesis fonética:
    - Expande abreviaturas comunes.
    - Convierte cifras numéricas a palabras habladas para entonación natural.
    - Limpia caracteres de formato Markdown.
    - Suaviza guiones largos y comillas.
    """
    if not text:
        return ""

    normalized = text

    # Reemplazar URLs por "enlace web"
    normalized = re.sub(r"https?://\S+|www\.\S+", " enlace web ", normalized)

    # Eliminar formato Markdown básico
    normalized = re.sub(r"#{1,6}\s*", "", normalized)
    normalized = re.sub(r"\*\*([^*]+)\*\*", r"\1", normalized)
    normalized = re.sub(r"\*([^*]+)\*", r"\1", normalized)
    normalized = re.sub(r"__([^_]+)__", r"\1", normalized)
    normalized = re.sub(r"`([^`]+)`", r"\1", normalized)

    # Símbolos comunes
    normalized = re.sub(r"\b(\d+)\s*%", r"\1 por ciento", normalized)
    normalized = re.sub(r"\s*%\s*", " por ciento ", normalized)
    normalized = re.sub(r"°C\b", " grados centígrados ", normalized)
    normalized = re.sub(r"°F\b", " grados Fahrenheit ", normalized)
    normalized = re.sub(r"\s*&\s*", " y ", normalized)
    normalized = re.sub(r"\s*@\s*", " arroba ", normalized)

    # Moneda ($)
    normalized = re.sub(r"\$\s*(\d+)", r"\1 dólares", normalized)

    # Limpiar comillas elegantes y rayas
    normalized = normalized.replace("“", '"').replace("”", '"')
    normalized = normalized.replace("‘", "'").replace("’", "'")
    normalized = normalized.replace("—", " - ").replace("–", " - ")

    # Expandir abreviaturas usando expresiones regulares
    for pattern, replacement in SPANISH_ABBREVIATIONS.items():
        normalized = re.sub(pattern, replacement, normalized)

    # Convertir números enteros a texto (hasta 9 dígitos)
    def _replace_num(match):
        val = int(match.group())
        return int_to_spanish(val)

    normalized = re.sub(r"\b\d{1,9}\b", _replace_num, normalized)

    # Reducir espacios en blanco múltiples
    normalized = re.sub(r"[ \t]+", " ", normalized)
    # Reducir saltos de línea excesivos a máximo 2
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)

    return normalized.strip()
