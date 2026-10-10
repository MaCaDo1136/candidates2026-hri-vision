# vision_pipeline

Todo lo que sale de la cámara: detecta caras, les da un id estable, dice quién es cada una y si está moviendo la boca.

## Flujo

```
camera_node
   /camera/image_raw
      |-> face_detector_node -> /faces/detections
      |                              |
      |                        tracker_node -> /faces/tracked
      |                              |
      |-> face_id_node (imagen + tracked) -> /faces/identified
      |-> mouth_movement_node (imagen + tracked) -> /faces/speaking
```

Los tres topics de caras son `vision_msgs/Detection2DArray`. La fusión solo consume `/faces/identified`; `/faces/speaking` por ahora se muestra en la interfaz web.

## Nodos

| Nodo | Entrada | Salida | Qué hace |
| --- | --- | --- | --- |
| `camera_node` | cámara o stream | `/camera/image_raw` | Lee el video en un hilo aparte y publica siempre el último fotograma, para que no se acumule retraso. |
| `face_detector_node` | `/camera/image_raw` | `/faces/detections` | Detecta caras con YOLOv8n-face. |
| `tracker_node` | `/faces/detections` | `/faces/tracked` | Asigna un id que se mantiene mientras la cara siga en cuadro (por traslape entre cuadros, IoU). |
| `face_id_node` | `/camera/image_raw`, `/faces/tracked` | `/faces/identified` | Calcula el embedding de cada cara y lo compara con las personas registradas. Debajo del umbral queda como desconocida. |
| `mouth_movement_node` | `/camera/image_raw`, `/faces/tracked` | `/faces/speaking` | Mide el movimiento de los labios de cada cara para marcar quién está hablando. |
| `visualizer_node` | los cuatro topics de arriba | ventana de OpenCV | Dibuja cajas, ids, nombre y estado de habla. Necesita pantalla; en Docker se usa la interfaz web. |

## Librerías y modelos

- OpenCV y `cv_bridge` para video y conversión a mensajes de ROS.
- Ultralytics con `yolov8n-face-lindevs.pt` para detección. El modelo viene en el repositorio, en `vision_pipeline/models/`.
- InsightFace (`buffalo_l`) para el embedding de la cara. Se descarga en `~/.insightface` la primera vez que corre.
- MediaPipe Face Mesh para los puntos de los labios. Como alternativa, el nodo puede usar los landmarks de InsightFace (`backend:=insightface`).
- NumPy para los cálculos y los embeddings guardados.

## Parámetros

| Nodo | Parámetro | Valor por defecto | Para qué sirve |
| --- | --- | --- | --- |
| `face_id_node` | `threshold` | 0.5 | Similitud coseno mínima para aceptar una identidad. |
| `face_id_node` | `process_every` | 10 | Identifica una de cada N imágenes, para no gastar CPU. |
| `mouth_movement_node` | `rate_hz` | 8.0 | Frecuencia con la que se mide la boca. |
| `mouth_movement_node` | `window_sec` | 2.0 | Ventana de tiempo sobre la que se mide el movimiento. |
| `mouth_movement_node` | `speaking_threshold` | 0.02 | Movimiento mínimo para considerar que habla. |
| `mouth_movement_node` | `hold_sec` | 0.8 | Cuánto se mantiene "hablando" después del último movimiento. |
| `mouth_movement_node` | `backend` | `mediapipe` | `mediapipe` o `insightface`. |

Cualquiera se puede cambiar al lanzar, por ejemplo:

```bash
ros2 run vision_pipeline face_id_node --ros-args -p threshold:=0.6
```

## Registro de caras

`face_embedding_generator.py` toma hasta 8 muestras de la cámara (mínimo 3, máximo 15 segundos) y guarda el promedio como embedding de la persona. Normalmente no se corre a mano: lo usa el menú de `people.sh`, que también permite registrar desde una foto. La cámara se elige con la variable `HRI_CAMERA_SOURCE`.

## Lanzar solo visión

```bash
ros2 launch vision_pipeline vision_pipeline.launch.py
```
