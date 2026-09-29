#! /usr/bin/env python3

"""
Description:
    Answers questions about a person using retrieval over that person's
    private documents (ChromaDB, filtered by person_id).

------------------------------
Services:
    /rag/query - RagQuery

------------------------------
Author: Mario Casas Donjuan
Date: September 28, 2026
"""

import os

import chromadb
import rclpy
import ollama

from rclpy.node import Node
from sentence_transformers import SentenceTransformer
from hri_vision_interfaces.srv import RagQuery


class RagServiceNode(Node):
    """Retrieval service over per-person documents."""

    def __init__(self):
        """Loads the embedding model, opens ChromaDB and creates the service."""
        super().__init__('rag_service_node')

        # max distance for the retrieved chunks to be considered relevant
        self.declare_parameter('max_distance', 0.19)

        self.model = SentenceTransformer('intfloat/multilingual-e5-small')
        client = chromadb.PersistentClient(
            path=os.path.expanduser('~/rag_db'))
        self.col = client.get_collection('documents')

        self.srv = self.create_service(
            RagQuery, '/rag/query', self.query_callback)
        self.get_logger().info('RAG service ready.')

    def query_callback(self, request, response):
        """Handles a RagQuery request."""
        max_distance = self.get_parameter('max_distance').value

        query_embedding = self.model.encode(
            'query: ' + request.question, normalize_embeddings=True).tolist()

        results = self.col.query(
            query_embeddings=[query_embedding],
            n_results=3,
            where={'person_id': request.person_id}
        )

        documents = results['documents'][0]
        distances = results['distances'][0]
        relevant_chunks = [
            doc for doc, dist in zip(documents, distances) if dist <= max_distance
        ]

        if not relevant_chunks:
            response.found = False
            response.answer = 'No encontré información sobre eso.'
        else:
            context = '\n'.join(relevant_chunks)
            prompt = (
                f'Contexto sobre esta persona:\n{context}\n\n'
                f'Pregunta: {request.question}\n\n'
                'Responde SOLO con información del contexto. '
                'Si el contexto no contiene la respuesta, di que no lo sabes.'
            )
            try:
                result = ollama.chat(
                    model='qwen2.5:1.5b',
                    messages=[{'role': 'user', 'content': prompt}]
                )
                response.answer = result['message']['content']
                response.found = True
            except Exception as e:
                self.get_logger().warn(f'Ollama falló: {e}')
                response.answer = ' '.join(relevant_chunks)
                response.found = True

        return response


def main(args=None):
    """Entry point of the node."""
    rclpy.init(args=args)
    node = RagServiceNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
