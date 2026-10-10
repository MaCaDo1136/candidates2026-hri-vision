# rag_service

Responde preguntas usando solo el documento de la persona que preguntó. Se expone como el servicio `/rag/query`.

## Cómo responde

1. Recibe `person_id` y `question` (`RagQuery`).
2. Convierte la pregunta en un embedding con `multilingual-e5-small`.
3. Busca en ChromaDB únicamente entre los fragmentos de ese `person_id`. El filtro se aplica en la búsqueda, así que los documentos de otras personas nunca llegan a comparar.
4. Si ningún fragmento queda por debajo de la distancia máxima, responde `found = false`.
5. Si hay contexto, se lo pasa a un modelo local de Ollama para redactar la respuesta. Si Ollama falla, devuelve directamente los fragmentos encontrados.

## Librerías y modelos

- `sentence-transformers` con `intfloat/multilingual-e5-small` para los embeddings.
- ChromaDB como base vectorial, guardada en `~/rag_db`.
- Ollama con `qwen2.5:1.5b` para generar la respuesta.

## Parámetros

| Parámetro | Valor por defecto | Para qué sirve |
| --- | --- | --- |
| `max_distance` | 0.22 | Distancia máxima para considerar que un fragmento sí contesta la pregunta. |
| `n_results` | 5 | Fragmentos que se recuperan. |
| `llm_model` | `qwen2.5:1.5b` | Modelo de Ollama que redacta la respuesta. |

## Chunking

Los documentos se parten en fragmentos antes de indexarse. Hay cuatro estrategias en `chunking.py`:

| Estrategia | Cómo corta |
| --- | --- |
| `sentence` | Por oraciones, sin cortar en abreviaturas como "Dr." o "Ing.". |
| `paragraph` | Por párrafos. |
| `fixed` | Por longitud fija. |
| `llm` | Un modelo de Ollama decide dónde termina cada idea. Es la que se usa por defecto. |

Se cambia con las variables `HRI_CHUNKING` (estrategia) y `HRI_CHUNK_MODEL` (modelo para `llm`). Normalmente no se corre a mano: `people.sh` indexa el documento al guardarlo y tiene una opción para reindexar con otra estrategia.

## Archivos

| Archivo | Contenido |
| --- | --- |
| `rag_service_node.py` | Nodo con el servicio `/rag/query`. |
| `ingest_documents.py` | Indexa documentos en ChromaDB. |
| `chunking.py` | Las estrategias de chunking. |
| `retrieval.py` | Selección del contexto y armado del prompt. |
