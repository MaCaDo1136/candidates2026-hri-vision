# audio_pipeline

Escucha el micrófono, detecta cuándo alguien habla, transcribe lo que dijo y averigua de quién es la voz.

## Flujo

```
vad_node -> /audio/segments (AudioSegment)
               |-> transcribe_node  -> /audio/transcripts (Transcript)
               |-> voice_id_node    -> /audio/identity (VoiceIdentity)
```

La transcripción y la identificación de voz leen el mismo topic en paralelo. Así la identidad no tiene que esperar a que termine de transcribirse.

## Nodos

| Nodo | Entrada | Salida | Qué hace |
| --- | --- | --- | --- |
| `vad_node` | micrófono | `/audio/segments` | Captura audio a 16 kHz, detecta voz con Silero VAD y publica cada frase completa como un segmento. Descarta ruidos muy cortos. |
| `transcribe_node` | `/audio/segments` | `/audio/transcripts` | Transcribe con faster-whisper, con el idioma forzado a español. |
| `voice_id_node` | `/audio/segments` | `/audio/identity` | Calcula el embedding de la voz y lo compara con las personas registradas. Si ninguna pasa el umbral, publica `Unknown`. |

## Librerías y modelos

- `sounddevice` (necesita `portaudio19-dev` en el sistema) para el micrófono.
- Silero VAD (`silero-vad`, con `torch`) para detectar voz.
- faster-whisper para la transcripción.
- Resemblyzer para el embedding de voz y similitud coseno.
- NumPy para el manejo de audio.

Los modelos de Whisper y Resemblyzer se descargan solos la primera vez.

## Parámetros

| Nodo | Parámetro | Valor por defecto | Para qué sirve |
| --- | --- | --- | --- |
| `voice_id_node` | `threshold` | 0.60 | Similitud mínima para aceptar a un hablante. |

## Registro de voces

`voice_embedding_generator.py` graba unos 8 segundos y exige al menos 2 segundos de voz real; guarda el
