"""Chunking strategies for the RAG documents: sentence, paragraph, fixed, llm."""

import re
import unicodedata

STRATEGIES = ('sentence', 'paragraph', 'fixed', 'llm')
DEFAULT_MAX_CHARS = {'sentence': 300,
                     'paragraph': 600, 'fixed': 400, 'llm': 300}
ABBREVIATIONS = ('Dr.', 'Dra.', 'Sr.', 'Sra.', 'Srta.', 'Ing.', 'Lic.',
                 'Mtro.', 'Mtra.', 'Prof.', 'núm.', 'vs.')

LLM_PROMPT = (
    'Eres un asistente que prepara texto para un sistema de búsqueda.\n'
    'Reescribe el TEXTO como una lista de afirmaciones cortas y '
    'autocontenidas.\n'
    'Reglas:\n'
    '- Una afirmación por línea, sin viñetas ni numeración.\n'
    '- Cada afirmación debe mencionar el nombre de la persona: @NAME@. '
    'Reemplaza pronombres y referencias ("ella", "su", "ahí") por ese '
    'nombre.\n'
    '- No omitas ningún dato: conserva lugares, motivos, cantidades, fechas '
    'y nombres propios.\n'
    '- Mantén el mismo tiempo verbal y el mismo sujeto que el texto.\n'
    '- Usa solo información que esté en el TEXTO. No inventes nada.\n'
    '- No incluyas introducciones ni comentarios, solo las afirmaciones.\n\n'
    'TEXTO:\n@TEXT@')


def split_sentences(text):
    """Splits text into sentences (each line is also a boundary)."""
    sentences = []
    for line in text.splitlines():
        line = re.sub(r'\s+', ' ', line).strip()
        if not line:
            continue
        parts = re.split(r'(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÑ¿¡"“(])', line)
        for i, part in enumerate(parts):
            part = part.strip()
            if not part:
                continue
            if i > 0 and sentences and sentences[-1].endswith(ABBREVIATIONS):
                sentences[-1] += ' ' + part  # abbreviation, not a real end
            else:
                sentences.append(part)
    return sentences


def chunk_sentences(text, max_chars=300):
    """Groups consecutive sentences until max_chars."""
    chunks, current = [], ''
    for sentence in split_sentences(text):
        if current and len(current) + 1 + len(sentence) > max_chars:
            chunks.append(current)
            current = ''
        current = f'{current} {sentence}'.strip()
    if current:
        chunks.append(current)
    return chunks


def chunk_paragraphs(text, max_chars=600):
    """One chunk per paragraph (blank-line separated); long ones by sentences."""
    chunks = []
    for paragraph in re.split(r'\n\s*\n', text):
        flat = re.sub(r'\s+', ' ', paragraph).strip()
        if not flat:
            continue
        if len(flat) <= max_chars:
            chunks.append(flat)
        else:
            chunks.extend(chunk_sentences(paragraph, max_chars))
    return chunks


def chunk_fixed(text, size=400, overlap=80):
    """Sliding window of `size` characters with `overlap`, cut at spaces."""
    text = re.sub(r'\s+', ' ', text).strip()
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            cut = text.rfind(' ', start + size // 2, end)
            if cut != -1:
                end = cut
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        nxt = max(end - overlap, start + 1)
        space = text.find(' ', nxt, end)
        start = space + 1 if space != -1 else nxt
    return [c for c in chunks if c]


# ----------------------------------------------------------- LLM chunking

def _words(text):
    """Lowercase accent-free words with at least 4 letters."""
    norm = unicodedata.normalize('NFD', text.lower())
    norm = ''.join(c for c in norm if unicodedata.category(c) != 'Mn')
    return {w for w in re.findall(r'[a-z0-9]+', norm) if len(w) >= 4}


def _grounded(statement, source_words, name_words, min_ratio=0.6):
    """Cheap hallucination guard: most content words must come from the source."""
    words = _words(statement) - name_words
    if not words:
        return False
    return len(words & source_words) / len(words) >= min_ratio


def _parse_statements(content):
    statements = []
    for line in content.splitlines():
        line = re.sub(r'^\s*(?:[-*•]+|\d+[.)])\s*', '', line).strip()
        if 15 <= len(line) <= 400:
            statements.append(line)
    return statements


def _sections(text, max_chars=1200):
    """Groups paragraphs into sections the LLM can handle in one call."""
    pieces = []
    for paragraph in re.split(r'\n\s*\n', text):
        flat = re.sub(r'\s+', ' ', paragraph).strip()
        if not flat:
            continue
        if len(flat) > max_chars:
            pieces.extend(chunk_sentences(paragraph, max_chars))
        else:
            pieces.append(flat)
    sections, current = [], ''
    for piece in pieces:
        if current and len(current) + 1 + len(piece) > max_chars:
            sections.append(current)
            current = ''
        current = f'{current} {piece}'.strip()
    if current:
        sections.append(current)
    return sections


def _facts_lost(sentence, statements_text, statement_words):
    """True if numbers, proper nouns or 2+ content words are missing."""
    for number in re.findall(r'\d+(?:[.,]\d+)?', sentence):
        if number not in statements_text:
            return True
    tokens = re.findall(r'\w+', sentence)
    for token in tokens[1:]:  # first token is just the sentence start
        words = _words(token)
        if token[0].isupper() and words and not words <= statement_words:
            return True
    return len(_words(sentence) - statement_words) >= 2


def chunk_llm(text, name='la persona', model='qwen2.5:1.5b', log=print,
              chat=None, section_chars=500):
    """LLM chunking with parent-child retrieval. Returns (chunk, parent) pairs.

    - The LLM rewrites each short section as self-contained statements
      (these are what gets embedded and searched).
    - The ORIGINAL section is kept as the parent: it is what the answering
      model reads, so nothing the LLM drops or distorts reaches the answer.
    - Statements are prefixed with the person's name when it is missing.
    - Original sentences whose facts (numbers, proper nouns, 2+ content
      words) are missing from all statements are added as extra chunks.
    - Hallucination guard + fallback to sentence chunks when the LLM fails.
    """
    if chat is None:
        import ollama
        chat = ollama.chat
    name_words = _words(name)
    first_name = name.split()[0]
    stats = {'sections': 0, 'kept': 0, 'dropped': 0, 'recovered': 0,
             'fallback': 0}
    pairs = []
    for section in _sections(text, section_chars):
        stats['sections'] += 1
        source_words = _words(section)
        try:
            prompt = LLM_PROMPT.replace(
                '@NAME@', name).replace('@TEXT@', section)
            result = chat(model=model,
                          messages=[{'role': 'user', 'content': prompt}],
                          options={'temperature': 0})
            raw = _parse_statements(result['message']['content'])
        except Exception as e:
            log(f'  [llm] falló una sección ({e}); uso oraciones')
            raw = []
        statements = [s for s in raw if _grounded(s, source_words, name_words)]
        stats['dropped'] += len(raw) - len(statements)
        sentences = split_sentences(section)
        if len(statements) < max(1, len(sentences) // 2):
            stats['fallback'] += 1
            pairs.extend((c, section) for c in chunk_sentences(section))
            continue
        statements = [s if first_name.lower() in s.lower()
                      else f'{first_name}: {s}' for s in statements]
        stats['kept'] += len(statements)
        joined = ' '.join(statements)
        covered = _words(joined)
        for sentence in sentences:
            if _facts_lost(sentence, joined, covered):
                statements.append(sentence)
                stats['recovered'] += 1
        pairs.extend((s, section) for s in statements)
    log(f"  [llm] secciones={stats['sections']} afirmaciones={stats['kept']} "
        f"descartadas={stats['dropped']} recuperadas={stats['recovered']} "
        f"con respaldo={stats['fallback']}")
    return pairs


def chunk(text, strategy='sentence', name='la persona', max_chars=None,
          model='qwen2.5:1.5b', log=print):
    """Returns a list of (chunk, parent) pairs for the chosen strategy."""
    if strategy not in STRATEGIES:
        raise ValueError(f'estrategia desconocida: {strategy}')
    max_chars = max_chars or DEFAULT_MAX_CHARS[strategy]
    if strategy == 'llm':
        return chunk_llm(text, name, model, log)
    if strategy == 'sentence':
        chunks = chunk_sentences(text, max_chars)
    elif strategy == 'paragraph':
        chunks = chunk_paragraphs(text, max_chars)
    else:
        chunks = chunk_fixed(text, max_chars, max_chars // 5)
    return [(c, c) for c in chunks]
