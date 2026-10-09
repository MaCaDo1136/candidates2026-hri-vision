import argparse
import os

import chromadb
from sentence_transformers import SentenceTransformer

from hri_common.people import DOC_FILE, load_people, person_path
from rag_service.chunking import STRATEGIES, chunk

DB_PATH = os.path.expanduser('~/rag_db')
DEFAULT_STRATEGY = os.environ.get('HRI_CHUNKING', 'llm')
DEFAULT_LLM = os.environ.get('HRI_CHUNK_MODEL', 'qwen2.5:1.5b')


def index_person(person_id, model, collection, strategy='sentence',
                 max_chars=None, llm_model=DEFAULT_LLM, dry_run=False):
    """Chunks, embeds and indexes one person's document."""
    doc_path = person_path(person_id, DOC_FILE)
    if not os.path.isfile(doc_path):
        print(f'[skip] {person_id}: no existe {doc_path}')
        return
    with open(doc_path, encoding='utf-8') as f:
        text = f.read()
    display = load_people().get(person_id, person_id)
    pairs = chunk(text, strategy, display, max_chars, llm_model)
    if not pairs:
        print(f'[skip] {person_id}: documento vacío')
        return
    if dry_run:
        parents = {p for _, p in pairs}
        print(f'--- {person_id} ({strategy}): {len(pairs)} chunks, '
              f'{len(parents)} contextos ---')
        for i, (c, _) in enumerate(pairs):
            print(f'{i:>3}. {c}')
        return
    texts = [c for c, _ in pairs]
    embeddings = model.encode(
        ['passage: ' + c for c in texts], normalize_embeddings=True).tolist()
    collection.delete(where={'person_id': person_id})
    collection.upsert(
        ids=[f'{person_id}_chunk_{i}' for i in range(len(texts))],
        embeddings=embeddings,
        documents=texts,
        metadatas=[{'person_id': person_id, 'strategy': strategy,
                    **({'parent': p} if p != c else {})}
                   for c, p in pairs],
    )
    print(f'Indexados {len(texts)} chunks para {person_id} ({strategy})')


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--name', help='person id, e.g. mario')
    group.add_argument('--all', action='store_true',
                       help='index every person in people.yaml')
    parser.add_argument('--strategy', choices=STRATEGIES,
                        default=DEFAULT_STRATEGY)
    parser.add_argument('--max-chars', type=int)
    parser.add_argument('--llm-model', default=DEFAULT_LLM)
    parser.add_argument('--dry-run', action='store_true',
                        help='print the chunks without indexing')
    args = parser.parse_args()

    people = load_people()
    targets = list(people) if args.all else [args.name.lower()]
    missing = [p for p in targets if p not in people]
    if missing:
        raise SystemExit(f'No registrados en people.yaml: {missing}')

    model = None if args.dry_run else SentenceTransformer(
        'intfloat/multilingual-e5-small')
    collection = None
    if not args.dry_run:
        client = chromadb.PersistentClient(path=DB_PATH)
        collection = client.get_or_create_collection(
            'documents', metadata={'hnsw:space': 'cosine'})
    for person_id in targets:
        index_person(person_id, model, collection, args.strategy,
                     args.max_chars, args.llm_model, args.dry_run)


if __name__ == '__main__':
    main()
