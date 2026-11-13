# 🔊 texToVox

Generador de voz neural (Text-to-Speech) de escritorio para Windows, **enfocado en máxima calidad, naturalidad y elegancia en español nativo latinoamericano, sin suscripciones ni límites de generación**.

---

## 🌟 Voces Nativas Elegantes (Prioridad Regional)

La aplicación cuenta con una cuidada selección de voces neuronales con dicción nativa humana, sin acentos extranjeros:

### 💎 Voces de Estudio (Español Nativo Elegante)
- **🇨🇴 Gonzalo (Colombia - Masculino):** Dicción corporativa impecable, timbre cálido y formal. La voz preferida para presentaciones de negocio y videos de alto nivel.
- **🇨🇴 Salomé (Colombia - Femenino):** Entonación humana suave, clara y cercana. Perfecta para tutoriales, cursos formativos y narraciones.
- **🇲🇽 Dalia (México / Neutro - Femenino):** Español neutro latinoamericano limpio, sin modismos marcados, ideal para difusión regional.
- **🇲🇽 Jorge (México / Neutro - Masculino):** Tono periodístico y documental, firme y profesional.
- **🇦🇷 Tomás (Argentina - Masculino):** Cadencia rioplatense auténtica, natural y fluida.
- **🇦🇷 Elena (Argentina - Femenino):** Locutora rioplatense con entonación expresiva y dinámica.
- **🇪🇸 Álvaro y Elvira (España):** Locución castellana peninsular natural.

### 📁 Voces 100% Offline Locales (Piper TTS)
Para cuando no dispongas de conexión a internet:
- **Claude (México / Offline):** Masculino en VITS local.
- **Daniela (Argentina / Offline):** Femenina rioplatense en VITS local.
- **Aldo (México / Offline):** Masculino narrativo en VITS local.

### 🧬 Clonación de Voz (Uso Personal / No Comercial)
Carga cualquier muestra de voz limpia de 6 a 20 segundos (`.wav`, `.mp3`, `.ogg`, `.flac`) desde el panel **Catálogo de Voces**:
- **OpenVoice v2 (`Clon Híbrido LatAm`):** Mantiene el acento y cadencia natural de una voz base latina (Colombiana, Mexicana, Argentina o Piper Offline) y transfiere el timbre de tu muestra (`~125 MB`, `RTF ~0.70` en CPU i7-11700 ya cacheado).
- **Coqui XTTS-v2 (`Clon 100% Offline`):** Clonación zero-shot directa sin internet con tu timbre y entonación en español (`~2.0 GB`, licencia personal CPML, `RTF ~3.10–3.82` en CPU).

### 🎛️ Sintetizador Vocal (Tono, Timbre y Preajustes)
Ajusta el carácter de cualquier voz (Edge-TTS, Piper, OpenVoice v2 o XTTS-v2) desde el panel **Sintetizador Vocal**:
- **Tono (`-8.0` a `+8.0` st):** Vuelve cualquier voz más grave o más aguda mediante TD-PSOLA (Praat/Parselmouth) sin cambiar la velocidad ni generar efecto robótico.
- **Cuerpo / Formantes (`0.80x` a `1.20x`) y Expresividad Tonal (`0.60x` a `1.50x`):** Modifica el timbre vocal y el rango melódico de forma independiente al tono.
- **Ecualización (`Calidez 180 Hz` y `Brillo 4.2 kHz`):** Filtros biquad shelf de `-6` a `+6 dB` para presencia grave o claridad en agudos.
- **Preajustes Incluidos y Propios:** Elige entre 7 perfiles de fábrica (`Natural (Sin cambios)`, `Radio / Locutor Grave`, `Narrador Profundo / Cine`, `Corporativo Claro`, `Juvenil / Agudo Natural`, `Voz Brillante / Comercial`, `Documental Sobrio`), guarda tus propios perfiles en `config/voice_presets.json` y reaplica ajustes al instante sobre el último audio generado sin volver a sintetizar el modelo neuronal.

### 🎧 Previsualización de Voz y Pausas entre Párrafos
Antes de generar y guardar archivos en disco, puedes probar cualquier voz seleccionada con tus ajustes exactos:
- **Controles de Ritmo:** Ajusta la **Velocidad de habla** (`0.50x`–`2.00x`), **Pausa entre oraciones** (`50`–`800 ms`) y **Pausa entre párrafos** (`150`–`2000 ms`).
- **Previsualización Instantánea (`▶ Probar` / `🎧 Previsualizar Voz`):** Reproduce un fragmento corto directamente en el reproductor integrado **sin exportar archivos WAV/MP3 a tu carpeta de salida**. Usa tus dos primeros párrafos si ya hay texto en el editor, o una muestra de dos párrafos si está vacío para que escuches el cambio entre párrafos y los ajustes del sintetizador.

### 🎭 Cambiar Voz a Audio (Conversor Voz a Voz)
En la pestaña **🎭 Cambiar Voz a Audio** puedes subir una grabación propia (`.wav`, `.mp3`, `.ogg`, `.flac`) o grabar directo desde el micrófono para transformarla conservando tus palabras, pausas y entonación hacia cualquier voz elegida en el **Catálogo de Voces** (incluyendo voces clonadas):
- **Previsualización de 8 s sin guardar** antes de procesar el audio completo.
- **Fuerza de conversión (`tau`):** ajusta entre suavidad orgánica (`0.15`) y parecido vocal marcado (`0.75`).
- **Compatible con el Sintetizador Vocal:** combina la conversión neuronal (OpenVoice v2) con el ajuste de tono Grave/Agudo, formantes y ecualización.

### 🎙️ Cambiador de Voz en Vivo (Grabaciones, Reuniones y Videojuegos)
En la pestaña **🎙️ En Vivo (Reuniones / Juegos)** puedes transformar tu micrófono en tiempo real:
- **Modo ⚡ Ultrarrápido (~95 ms totales):** Modifica tono, formantes, expresividad, calidez y brillo mediante TD-PSOLA + filtros IIR continuos al vuelo usando los ajustes del **Sintetizador Vocal**. Ideal para Discord, Zoom, Meet, Teams y videojuegos.
- **Modo 🧠 Neural OpenVoice (~0.65 s):** Auto-calibra tu micrófono al hablar ~1 s y convierte tu timbre hacia cualquier voz latina elegida o voz clonada en memoria (sin escritura a disco).
- **Enrutamiento y Cable Virtual:** Prioriza dispositivos WASAPI de baja latencia, detecta automáticamente **VB-Cable (`CABLE Input`)** para entrar como micrófono (`CABLE Output`) en llamadas/juegos, permite **Monitor privado en auriculares** y incluye **🔴 Grabar En Vivo** directo a WAV/MP3.

---

## 🎧 Muestras de Audio Reales Generadas

Puedes escuchar de inmediato las muestras de audio generadas en [`output/demostracion_final/`](file:///d:/Desarrollos/texToVox/output/demostracion_final) y las pruebas de clonación en [`output/cloning_test/`](file:///d:/Desarrollos/texToVox/output/cloning_test):

1. **Gonzalo (Colombia):** [`1_Gonzalo_Colombia_Masculino.wav`](file:///d:/Desarrollos/texToVox/output/demostracion_final/1_Gonzalo_Colombia_Masculino.wav) / [`MP3`](file:///d:/Desarrollos/texToVox/output/demostracion_final/1_Gonzalo_Colombia_Masculino.mp3)
2. **Salomé (Colombia):** [`2_Salome_Colombia_Femenino.wav`](file:///d:/Desarrollos/texToVox/output/demostracion_final/2_Salome_Colombia_Femenino.wav) / [`MP3`](file:///d:/Desarrollos/texToVox/output/demostracion_final/2_Salome_Colombia_Femenino.mp3)
3. **Dalia (México / Neutro):** [`3_Dalia_Mexico_Neutro_Femenino.wav`](file:///d:/Desarrollos/texToVox/output/demostracion_final/3_Dalia_Mexico_Neutro_Femenino.wav) / [`MP3`](file:///d:/Desarrollos/texToVox/output/demostracion_final/3_Dalia_Mexico_Neutro_Femenino.mp3)
4. **Tomás (Argentina):** [`4_Tomas_Argentina_Masculino.wav`](file:///d:/Desarrollos/texToVox/output/demostracion_final/4_Tomas_Argentina_Masculino.wav) / [`MP3`](file:///d:/Desarrollos/texToVox/output/demostracion_final/4_Tomas_Argentina_Masculino.mp3)

---

## 🚀 Requisitos e Instalación

### Requisitos Previos
- **Sistema Operativo:** Windows 10 u 11 (64-bit).
- **Python:** Versión 3.10 a 3.14 (64-bit).

### Instalación de Dependencias

```bash
pip install -r requirements.txt
```

Si mueves el proyecto a otro equipo o necesitas reinstalar el entorno aislado de clonación (`.venv_clone` en Python 3.12 con PyTorch CPU, Coqui XTTS-v2 y librerías de audio), ejecuta:

```bash
instalar_clonacion.bat
```

---

## 💻 Ejecución de la Aplicación

Haz doble clic en [`iniciar_texToVox.bat`](file:///d:/Desarrollos/texToVox/iniciar_texToVox.bat) o ejecuta en tu terminal:

```bash
python main.py
```

---

## 📦 Compilar a Ejecutable Independiente (.exe)

Para generar una versión portátil para Windows sin requerir instalación previa de Python:

- Haz doble clic en [`compilar_exe.bat`](file:///d:/Desarrollos/texToVox/compilar_exe.bat) (o corre `python build_exe.py`).
- El instalable se creará en la carpeta `dist/texToVox/texToVox.exe`.
# TextToVox
