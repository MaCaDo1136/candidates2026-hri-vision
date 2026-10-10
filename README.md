# Reto Avanzado Mario Casas Donjuan - HRI/Vision Roborregos

Sistema en ROS 2 Jazzy que detecta a varias personas frente a una cámara, identifica quién está hablando combinando cara, voz y movimiento de labios, y contesta preguntas usando únicamente el documento privado de esa persona.

La identidad funciona como permiso, no como llave de ruteo: si no se puede confirmar quién habla, el sistema no responde con información de nadie.

## Cómo funciona

1. La cámara y el micrófono publican imagen y audio.
2. Visión: detección de caras (YOLOv8n-face), tracking, identificación de cara (InsightFace) y movimiento de labios (MediaPipe Face Mesh).
3. Audio: detección de voz (Silero VAD), transcripción (faster-whisper) e identificación de voz (Resemblyzer).
4. Fusión: la identidad de la voz debe coincidir con una cara visible. El resultado es uno de tres estados: responder, preguntar (confirmar identidad) o negar.
5. Si se responde, el RAG consulta solo los documentos de esa persona (ChromaDB + modelo de lenguaje local con Ollama).
6. Una página web muestra la cámara, quién habló, la pregunta, la respuesta y la confianza.

## Requisitos

- Ubuntu 24.04 (ROS 2 Jazzy lo exige). En Docker sirve la imagen `ubuntu:24.04`.
- Unos 15 gb de disco por los modelos y 8gb de ram de preferencia.
- Cámara y micrófono.

## Instalación desde cero

El repositorio se clona dentro de la carpeta `src` de un workspace de ROS:

```bash
sudo apt-get update && sudo apt-get install -y git
mkdir -p ~/hri_vision_ws
git clone https://github.com/MaCaDo1136/candidates2026-hri-vision.git ~/hri_vision_ws/src
bash ~/hri_vision_ws/src/run.sh
```

`run.sh` hace todo en orden y se puede volver a correr cuando se quiera:

1. Instala ROS 2 Jazzy si no está.
2. Instala las dependencias del sistema y de Python (`requirements.txt`).
3. Instala Ollama y descarga el modelo `qwen2.5:1.5b`.
4. Compila todos los paquetes con `colcon`.
5. Abre el menú para registrar personas.
6. Lanza el sistema.

La primera vez tarda varios minutos. Los demás modelos (InsightFace, Whisper, embeddings del RAG) se descargan solos al arrancar los nodos. El modelo de caras viene incluido en el repositorio.

## Registrar personas

El menú aparece al correr `run.sh`, y también se puede abrir solo con `./people.sh`.

- **Nuevo usuario:** pide un id y un nombre, y después cara, voz y documento. Se puede omitir cualquiera y completarlo luego.
- **Cara:** con la cámara (varias muestras) o desde una foto. Con una sola persona frente a la cámara.
- **Voz:** una grabación corta hablando normal.
- **Documento:** se escribe en un editor o se importa un `.txt`. Al guardarlo se indexa automáticamente para el RAG.
- Hay opciones para actualizar, eliminar, reindexar con otra estrategia de chunking, respaldar y una autoprueba que verifica que el RAG no mezcle información entre personas.

Los datos se guardan en `~/hri_people` y la base vectorial en `~/rag_db`. Si cambias algo con el sistema ya corriendo, reinícialo para que los nodos lo carguen.

## Ejecución

Normalmente basta `bash ~/hri_vision_ws/src/run.sh`. Para lanzar sin pasar por el menú:

```bash
source /opt/ros/jazzy/setup.bash
source ~/hri_vision_ws/install/setup.bash
ros2 launch identity_fusion identity_fusion.launch.py
```

La interfaz web queda en `http://localhost:8080`.

## Docker

Probado con `ubuntu:24.04` (arm64). El contenedor parte limpio y `run.sh` instala todo.

```bash
docker run -it --name hri -p 8080:8080 ubuntu:24.04 bash
apt-get update && apt-get install -y git
mkdir -p ~/hri_vision_ws
git clone https://github.com/MaCaDo1136/candidates2026-hri-vision.git ~/hri_vision_ws/src
bash ~/hri_vision_ws/src/run.sh
```

Para volver a entrar después: `docker start -ai hri`.

### Cámara y micrófono

- **Host Linux:** agrega `--device /dev/video0 --device /dev/snd` al `docker run`.
- **Docker Desktop en Mac o Windows:** el contenedor no ve los dispositivos. La cámara y el micrófono se envían por red desde el equipo anfitrión.

Cámara (en el Mac, requiere `ffmpeg`; el bucle reinicia el servidor tras cada conexión):

```bash
while true; do
  ffmpeg -f avfoundation -framerate 30 -video_size 640x480 -i "0" -r 10 \
    -c:v libx264 -preset ultrafast -tune zerolatency -g 10 -f mpegts "tcp://0.0.0.0:8090?listen=1"
  sleep 1
done
```

Micrófono (en el Mac, requiere `pulseaudio` con Homebrew):

```bash
pulseaudio --exit-idle-time=-1 --load="module-native-protocol-tcp auth-anonymous=1"
```

Dentro del contenedor, una sola vez:

```bash
apt-get install -y pulseaudio-utils libasound2-plugins
cat > ~/.asoundrc <<'EOF'
pcm.!default { type pulse }
ctl.!default { type pulse }
EOF
cat >> ~/.bashrc <<'EOF'
export HRI_CAMERA_SOURCE="tcp://host.docker.internal:8090"
export PULSE_SERVER=tcp:host.docker.internal:4713
export OPENCV_FFMPEG_CAPTURE_OPTIONS="fflags;nobuffer|flags;low_delay"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
EOF
source ~/.bashrc
```

Con la cámara llegando por red conviene registrar las caras desde una foto: el retraso del stream puede hacer que la captura en vivo tome fotogramas viejos. Para copiar una foto al contenedor: `docker cp foto.jpg hri:/root/foto.jpg`.

El nodo `visualizer_node` hace que se vea

## Variables de entorno

| Variable | Uso |
| --- | --- |
| `HRI_CAMERA_SOURCE` | Índice de cámara (`0`) o URL de un stream. Por defecto `0`. |
| `PULSE_SERVER` | Servidor PulseAudio cuando el audio llega por red. |
| `HRI_PEOPLE_DIR` | Carpeta de personas. Por defecto `~/hri_people`. |
| `OMP_NUM_THREADS` | Hilos por nodo. `run.sh` usa 2 por defecto para que los nodos no compitan por el CPU. |

Si estás en Mac o Windows, necesitas poner las variables a mano con los siguientes comandos

```bash
cat >> ~/.bashrc <<'EOF'
export HRI_CAMERA_SOURCE="tcp://host.docker.internal:8090"
export PULSE_SERVER=tcp:host.docker.internal:4713
export OPENCV_FFMPEG_CAPTURE_OPTIONS="fflags;nobuffer|flags;low_delay"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
EOF
source ~/.bashrc
```

## Estructura

| Paquete | Contenido |
| --- | --- |
| `vision_pipeline` | Cámara, detector de caras, tracker, identificación de cara, labios, visualizador. |
| `audio_pipeline` | VAD, transcripción, identificación de voz. |
| `identity_fusion` | Fusión de identidad, decisión de respuesta e interfaz web. |
| `rag_service` | Servicio de RAG con filtro por persona. |
| `hri_common` | Registro de personas y menú de administración. |
| `hri_vision_interfaces` | Mensajes y servicios propios. |

## Interfaces propias

Definidas en `hri_vision_interfaces`. El paquete tiene tres mensajes y un servicio. No usa acciones (`.action`).

### Mensajes

**AudioSegment**: segmento de voz detectado.

| Campo | Tipo |
| --- | --- |
| `header` | `std_msgs/Header` |
| `samples` | `float32[]` |
| `sample_rate` | `int32` |

Lo publica `vad_node` en `/audio/segments`. Lo consumen `transcribe_node` y `voice_id_node`.

**Transcript**: texto transcrito de un segmento.

| Campo | Tipo |
| --- | --- |
| `header` | `std_msgs/Header` |
| `text` | `string` |
| `language` | `string` |

Lo publica `transcribe_node`. Lo consume `identity_fusion_node`.

**VoiceIdentity**: persona identificada por la voz.

| Campo | Tipo |
| --- | --- |
| `header` | `std_msgs/Header` |
| `speaker` | `string` |
| `score` | `float32` |

Lo publica `voice_id_node` en `/audio/identity`. Lo consume `identity_fusion_node`.

### Servicio

**RagQuery**: consulta al RAG limitada a los documentos de una persona.

| Campo | Tipo | Dirección |
| --- | --- | --- |
| `person_id` | `string` | petición |
| `question` | `string` | petición |
| `answer` | `string` | respuesta |
| `found` | `bool` | respuesta |

Lo atiende `rag_service_node`. Lo llama `identity_fusion_node`. `found` es falso cuando no hay información en el documento de esa persona.

## Topics principales

| Topic | Contenido |
| --- | --- |
| `/camera/image_raw` | Imagen de la cámara. |
| `/faces/tracked` | Caras detectadas con id de tracking. |
| `/faces/identified` | Cara con la persona identificada y su puntaje. |
| `/faces/speaking` | Actividad de labios por cara. |
| `/audio/segments` | Segmentos de voz detectados (`AudioSegment`). |
| `/audio/transcripts` | Texto transcrito de cada segmento (`Transcript`). |
| `/audio/identity` | Persona identificada por la voz. |
| `/fusion/state` | Decisión de la fusión (JSON). |
| `/system/response` | Respuesta final del sistema. |

## Problemas comunes

- **El sistema no reconoce a alguien recién registrado:** reinicia el launch; los nodos leen el registro al arrancar.
- **`can't open camera` o `Error querying device`:** no hay cámara o micrófono visibles; revisa `HRI_CAMERA_SOURCE`, `PULSE_SERVER` o los `--device` de Docker.
- **La imagen llega con retraso:** revisa el uso de CPU (`docker stats`); el sistema carga varios modelos a la vez.
- **Un nodo se queda con una versión vieja:** `run.sh` recompila siempre antes de lanzar; si lanzas a mano, corre `colcon build` primero.

## Limitaciones

Los umbrales de decisión y de reconocimiento no se calibraron con una muestra grande de personas; son valores ajustados a mano con pruebas pequeñas.
