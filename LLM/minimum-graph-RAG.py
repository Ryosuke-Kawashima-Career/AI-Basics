# [Library Import]
import neo4j
from typing import List, Dict, Tuple, Any
from neo4j_graphrag.llm import OpenAILLM as LLM
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings as Embeddings
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.retrievers import VectorRetriever
from neo4j_graphrag.generation.graphrag import GraphRAG

neo4j_driver = neo4j.GraphDatabase.driver(NEO4J_URI,
                                          auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

class MinimumGraphRAG:
    def __init__(self, model_name: str, knowledge_graph: VectorRetriever):
        self.llm = LLM.from_pretrained(model_name)
        self.embeddings = Embeddings.from_pretrained(model_name)
        self.knowledge_graph = knowledge_graph


def main():
    # [LLM initiation]

    # [Embedding]

    # [Query Input]

    # [Graph Retrieval]

    # [Ingest Retrieval into the Query]

    # [Generation]

    # [Evaluation]

if __name__ == "__main__":
    main()
