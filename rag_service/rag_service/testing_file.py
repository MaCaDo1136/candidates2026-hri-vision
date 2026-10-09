import os

import chromadb
from sentence_transformers import SentenceTransformer

client = chromadb.PersistentClient(path=os.path.expanduser('~/rag_db'))
col = client.get_collection('documents')
model = SentenceTransformer('intfloat/multilingual-e5-small')

q = model.encode('query: ¿Que estudia Mario?',
                 normalize_embeddings=True).tolist()
res = col.query(query_embeddings=[q], n_results=2,
                where={'person_id': 'mario'})
print(res['documents'])
print(res['distances'])

preguntas = [
    # deben encontrarse
    '¿Qué estudia Mario?',
    '¿Cuándo se gradúa Mario?',
    '¿Cómo se llaman sus perras?',
    '¿Cuál es su color favorito?',
    '¿De qué equipo de fútbol es?',
    '¿Qué servicios corre en su homelab?',
    # NO deben encontrarse
    '¿Cuál es la capital de Francia?',
    '¿Cómo se prepara una pizza?',
    '¿Cuánto mide la torre Eiffel?',
    '¿Quién ganó el mundial de 2018?',
]
for p in preguntas:
    q = model.encode('query: ' + p, normalize_embeddings=True).tolist()
    res = col.query(query_embeddings=[q], n_results=1,
                    where={'person_id': 'mario'})
    print(f'{res["distances"][0][0]:.3f}  {p}  ->  {res["documents"][0][0][:45]}')
