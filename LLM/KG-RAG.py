from typing import Dict, Tuple, List, Any, Optional
from dataclasses import dataclass
import Neo4j

@dataclass
class Entity:
    name: str
    properties: Dict[str, Any] = {}

@dataclass
class Relation:
    name: str
    source: str
    target: str
    properties: Dict[str, Any] = {}

@dataclass
class KnowledgeGraph:
    def __init__(self):
        #entity -> relations -> entities
        self.nodes: Dict[str, Entity] = {}
        self.edges: List[Relation] = []
        
        # For quick lookups
        self.adj: Dict[str, List[Relation]] = {}

    def add_node(self, entity: Entity) -> None:
        if entity.name not in self.nodes:
            self.nodes[entity.name] = entity
            self.adj[entity.name] = []
    
    def add_edge(self, edge: Relation) -> None:
        if edge.source not in self.nodes:
            self.add_node(Entity(name=edge.source))
        if edge.target not in self.nodes:
            self.add_node(Entity(name=edge.target))
        self.edges.append(edge)
        self.adj[edge.source].append(edge)

class KnowledgeGraphRAG:
    def __init__(self, model_name: str, knowledge_graph: KnowledgeGraph):
        self.model_name = model_name
        self.knowledge_graph = knowledge_graph
        self.llm = AutoCausalLlm.from_pretrained(model_name)

    def rag_query(self, question: str) -> str:
        #query exttraction

        #retrieval from knowledge graph

        #context

    def fetch_graph_context(self, question: str) -> str:
