#! /usr/bin/env python3

"""
Description:
    Answers questions about a person using retrieval over that person's
    private documents (ChromaDB, filtered by person_id) and Ollama.
    Retrieval matches small chunks; the model answers from the original
    text those chunks came from (parent context).

------------------------------
Services:
    /rag/query - RagQuery

------------------------------
Parameters:
    max_distance (float) - maximum cosine distance for a chunk to count
    n_results (int)      - chunks retrieved before filtering
    llm_model (str)      - Ollama model used to write the answer

------------------------------
Author: Mario Casas Donjuan
Date: October 8, 2026
"""

import os

import chromadb
import ollama
import rclpy

from rclpy.node import Node
from sentence_transformers import SentenceTransformer
from hri_vision_interfaces.srv import RagQuery
from rag_service.retrieval import build_prompt, select_context


class RagServiceNode(Node):
    """Retrieval service over per-person documents."""

    def __init__(self):
        """Loads the embedding model, opens ChromaDB and creates the service."""
        super().__init__('rag_service_node')

        self.declare_parameter('max_distance', 0.22)
        self.declare_parameter('n_results', 5)
        self.declare_parameter('llm_model', 'qwen2.5:1.5b')

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
        n_results = self.get_parameter('n_results').value
        person_id = request.person_id.lower()

        query_embedding = self.model.encode(
            'query: ' + request.question, normalize_embeddings=True).tolist()
        results = self.col.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where={'person_id': person_id}
        )
        context_parts, distances = select_context(results, max_distance)
        self.get_logger().info(
            f'{person_id}: distancias={[round(d, 3) for d in distances]} '
            f'contextos usados={len(context_parts)}')

        if not context_parts:
            response.found = False
            response.answer = 'No encontré información sobre eso.'
            return response

        try:
            result = ollama.chat(
                model=self.get_parameter('llm_model').value,
                messages=[{'role': 'user',
                           'content': build_prompt(context_parts,
                                                   request.question)}]
            )
            response.answer = result['message']['content']
        except Exception as e:
            self.get_logger().warn(f'Ollama falló: {e}')
            response.answer = ' '.join(context_parts)
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
