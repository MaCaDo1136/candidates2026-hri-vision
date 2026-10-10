# identity_fusion

Decide si el sistema contesta, pide confirmar o se niega, cruzando la voz con las caras que hay en cuadro. También incluye la interfaz web.

## Flujo

```
/faces/identified  --\
/audio/identity    ---+-> identity_fusion_node -> /fusion/state
/audio/transcripts --/                         -> /system/response
                              |
                              +-> servicio /rag/query (solo si responde)
```

## Cómo decide

La decisión se toma cada vez que llega una transcripción nueva.

1. Toma la última voz identificada. Si es muy vieja (más de 2.5 s) o es `Unknown`, el estado es `negar`.
2. Busca entre las caras que hay en cuadro una que sea de esa misma persona. Si no hay ninguna, el estado es `negar`: la voz sola no basta.
3. La confianza es la menor entre la puntuación de la cara y la de la voz.
4. Según la confianza:

| Confianza | Estado | Qué hace el sistema |
| --- | --- | --- |
| 0.55 o más | `responder` | Consulta el RAG con el `person_id` confirmado y publica la respuesta. |
| entre 0.35 y 0.55 | `preguntar` | Pide que se acerque y hable de nuevo. No consulta el RAG. |
| menos de 0.35 | `negar` | Dice que no pudo confirmar la identidad. No consulta el RAG. |

El nombre que se envía al RAG sale siempre de la identidad confirmada por cara y voz, nunca del texto de la pregunta.

Limitación actual: el movimiento de labios (`/faces/speaking`) todavía no entra en la decisión. Se muestra en la interfaz web, pero la fusión solo cruza voz y cara.

## Parámetros de `identity_fusion_node`

| Parámetro | Valor por defecto | Para qué sirve |
| --- | --- | --- |
| `fusion_window_sec` | 2.5 | Qué tan reciente debe ser la voz y la cara para tomarlas en cuenta. |
| `threshold_respond` | 0.55 | Confianza mínima para responder. |
| `threshold_ask` | 0.35 | Confianza mínima para preguntar en vez de negar. |

## Topics

| Topic | Dirección | Contenido |
| --- | --- | --- |
| `/faces/identified` | entrada | Caras con su identidad y puntaje. |
| `/audio/identity` | entrada | Quién habló, según la voz. |
| `/audio/transcripts` | entrada | Lo que se dijo. |
| `/fusion/state` | salida | JSON con pregunta, persona, confianza, estado, voz y caras vistas. |
| `/system/response` | salida | Texto de la respuesta final. |

## Interfaz web (`ui_node`)

Un servidor HTTP en el puerto 8080 que muestra la cámara con las cajas, la voz detectada, el estado de la fusión y la conversación (la pregunta y la respuesta de cada turno). Se abre en `http://localhost:8080`.

| Parámetro | Valor por defecto |
| --- | --- |
| `host` | `0.0.0.0` |
| `port` | 8080 |
| `max_fps` | 15 |
| `jpeg_quality` | 75 |

Lleva copias de los umbrales de la fusión (`threshold_respond`, `threshold_ask`) para dibujar los estados; si cambias esos valores en la fusión, cámbialos también aquí.

## Lanzar todo el sistema

```bash
ros2 launch identity_fusion identity_fusion.launch.py
```

Este launch incluye los de `vision_pipeline` y `audio_pipeline`, y además arranca `rag_service_node`, `identity_fusion_node` y `ui_node`.
