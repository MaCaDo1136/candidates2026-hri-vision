"""Helpers to turn ChromaDB results into the context given to the LLM."""


def select_context(results, max_distance):
    """Returns (context_parts, distances) from a ChromaDB query result.

    Keeps chunks within max_distance. If a chunk has a 'parent' (the original
    text it came from), the parent is used instead, without duplicates.
    """
    documents = results['documents'][0] if results['documents'] else []
    metadatas = results['metadatas'][0] if results.get('metadatas') else []
    distances = results['distances'][0] if results['distances'] else []
    parts = []
    for i, (doc, dist) in enumerate(zip(documents, distances)):
        if dist > max_distance:
            continue
        meta = metadatas[i] if i < len(metadatas) else None
        text = (meta or {}).get('parent') or doc
        if text not in parts:
            parts.append(text)
    return parts, distances


def build_prompt(context_parts, question):
    """Builds the grounded prompt for the answering model."""
    context = '\n\n'.join(context_parts)
    return (
        'Responde la PREGUNTA usando solo el CONTEXTO. Reglas:\n'
        '- Responde en español, en una o dos oraciones, hablando de la '
        'persona en tercera persona.\n'
        '- Si la respuesta aparece en el CONTEXTO, aunque la pregunta use '
        'otras palabras, dila.\n'
        '- Solo si de verdad no aparece, responde exactamente: '
        '"No tengo esa información."\n\n'
        f'CONTEXTO:\n{context}\n\n'
        f'PREGUNTA: {question}\n\n'
        'RESPUESTA:')
