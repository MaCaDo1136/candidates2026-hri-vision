import os

import chromadb
from sentence_transformers import SentenceTransformer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.expanduser('~/rag_db')
DOC_PATH = os.path.join(SCRIPT_DIR, 'mi_documento.txt')
PERSON_ID = 'mario'


def chunk_text(text, max_chars=120):
    """Split text into chunks of roughly max_chars, cutting at sentence ends."""
    sentences = [s.strip()
                 for s in text.replace('\n', ' ').split('.') if s.strip()]
    chunks, current = [], ''
    for sentence in sentences:
        if current and len(current) + len(sentence) > max_chars:
            chunks.append(current.strip())
            current = ''
        current += sentence + '. '
    if current.strip():
        chunks.append(current.strip())
    return chunks


with open(DOC_PATH, encoding='utf-8') as f:
    chunks = chunk_text(f.read())

model = SentenceTransformer('intfloat/multilingual-e5-small')
embeddings = model.encode(
    ['passage: ' + c for c in chunks], normalize_embeddings=True).tolist()

client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_or_create_collection(
    'documents', metadata={'hnsw:space': 'cosine'})

collection.upsert(
    ids=[f'{PERSON_ID}_chunk_{i}' for i in range(len(chunks))],
    embeddings=embeddings,
    documents=chunks,
    metadatas=[{'person_id': PERSON_ID} for _ in chunks],
)
print(f'Indexados {len(chunks)} chunks para {PERSON_ID} en {DB_PATH}')
