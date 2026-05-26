import re
import json
from typing import Dict, Tuple, List, Any, Optional
from dataclasses import dataclass
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer

# Neo4j connector for professional database operations
try:
    from neo4j import GraphDatabase, Driver
except ImportError:
    # Safe fallback if not installed, though we installed it in our env
    GraphDatabase = None
    Driver = None

class Neo4jConnector:
    def __init__(self, uri: str = "bolt://localhost:7687", auth: Tuple[str, str] = ("neo4j", "password")):
        self.uri = uri
        self.auth = auth
        self.driver = None
        self.connected = False
        if GraphDatabase is not None:
            self.connect()

    def connect(self):
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=self.auth)
            # Verify connectivity
            self.driver.verify_connectivity()
            self.connected = True
            print(f"Successfully connected to Neo4j database at {self.uri}!")
        except Exception as e:
            self.connected = False
            self.driver = None
            print(f"Warning: Could not connect to Neo4j database at {self.uri}. Reason: {e}")
            print("Falling back to local in-memory knowledge graph.")

    def close(self):
        if self.driver:
            self.driver.close()

    def run_query(self, query: str, parameters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        if not self.connected or not self.driver:
            raise RuntimeError("Neo4j is not connected.")
        with self.driver.session() as session:
            result = session.run(query, parameters or {})
            return [record.data() for record in result]

@dataclass
class Entity:
    name: str
    properties: Dict[str, Any] = None

    def __post_init__(self):
        if self.properties is None:
            self.properties = {}

@dataclass
class Relation:
    name: str
    source: str
    target: str
    properties: Dict[str, Any] = None

    def __post_init__(self):
        if self.properties is None:
            self.properties = {}

class KnowledgeGraph:
    def __init__(self, neo4j_conn: Optional[Neo4jConnector] = None):
        self.nodes: Dict[str, Entity] = {}
        self.edges: List[Relation] = []
        # For quick lookups
        self.adj: Dict[str, List[Relation]] = {}
        self.neo4j_conn = neo4j_conn

    def add_node(self, entity: Entity) -> None:
        if entity.name not in self.nodes:
            self.nodes[entity.name] = entity
            self.adj[entity.name] = []
        
        # Synchronize to Neo4j if available
        if self.neo4j_conn and self.neo4j_conn.connected:
            try:
                query = """
                MERGE (e:Entity {name: $name})
                SET e += $properties
                """
                self.neo4j_conn.run_query(query, {"name": entity.name, "properties": entity.properties})
            except Exception as e:
                print(f"Failed to sync node to Neo4j: {e}")
    
    def add_edge(self, edge: Relation) -> None:
        if edge.source not in self.nodes:
            self.add_node(Entity(name=edge.source))
        if edge.target not in self.nodes:
            self.add_node(Entity(name=edge.target))
            
        self.edges.append(edge)
        self.adj[edge.source].append(edge)
        
        # Synchronize to Neo4j if available
        if self.neo4j_conn and self.neo4j_conn.connected:
            try:
                # Sanitize relation name for safe Cypher type insertion
                rel_type = re.sub(r'[^a-zA-Z0-9_]', '_', edge.name).upper()
                query = f"""
                MERGE (s:Entity {{name: $source}})
                MERGE (t:Entity {{name: $target}})
                MERGE (s)-[r:{rel_type}]->(t)
                SET r += $properties
                """
                self.neo4j_conn.run_query(query, {
                    "source": edge.source,
                    "target": edge.target,
                    "properties": edge.properties
                })
            except Exception as e:
                print(f"Failed to sync edge to Neo4j: {e}")
    
    def get_context(self, query_entities: List[str]) -> str:
        """Retrieves graph context surrounding the given query entities"""
        if not query_entities:
            return "No matching entities found in the graph database."

        retrieved_facts = []
        
        # Try retrieving from Neo4j if connected
        if self.neo4j_conn and self.neo4j_conn.connected:
            try:
                query = """
                MATCH (n:Entity)-[r]->(m:Entity)
                WHERE n.name IN $entities OR m.name IN $entities
                RETURN n.name AS source, type(r) AS relation, m.name AS target, r AS properties
                """
                records = self.neo4j_conn.run_query(query, {"entities": query_entities})
                for rec in records:
                    props_str = f" with properties {rec['properties']}" if rec['properties'] else ""
                    retrieved_facts.append(f"{rec['source']} -[{rec['relation']}]-> {rec['target']}{props_str}")
            except Exception as e:
                print(f"Error reading from Neo4j: {e}. Falling back to memory search...")
        
        # Local in-memory search fallback (or primary search if no Neo4j)
        if not retrieved_facts:
            for entity in query_entities:
                # Search edges starting from this entity
                if entity in self.adj:
                    for edge in self.adj[entity]:
                        props_str = f" with properties {edge.properties}" if edge.properties else ""
                        retrieved_facts.append(f"{edge.source} -[{edge.name}]-> {edge.target}{props_str}")
                # Search edges ending at this entity
                for edge in self.edges:
                    if edge.target == entity and edge.source != entity:
                        props_str = f" with properties {edge.properties}" if edge.properties else ""
                        retrieved_facts.append(f"{edge.source} -[{edge.name}]-> {edge.target}{props_str}")

        # Unique facts formatted neatly
        unique_facts = list(set(retrieved_facts))
        if not unique_facts:
            return "No relationships found for the extracted entities."
            
        return "The following facts were retrieved from the Graph Database:\n" + "\n".join(unique_facts)

class KnowledgeGraphRAG:
    def __init__(self, model_name: str, knowledge_graph: KnowledgeGraph):
        self.model_name = model_name
        self.knowledge_graph = knowledge_graph
        print(f"Loading Causal LLM: {model_name}...")
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def extract_entities(self, question: str) -> List[str]:
        """Extract key entities from the user's question using the LLM with a fallback keyword matching parser"""
        prompt = f"""Identify all key entity names, proper nouns, or topics mentioned in the following question. Return ONLY a valid JSON list of strings. Do not add any conversational text.
Question: "{question}"
Answer:"""
        
        prompt_str = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False
        )
        
        input_tokens = self.tokenizer(prompt_str, return_tensors="pt").to(self.llm.device)
        with torch.no_grad():
            generated_tokens = self.llm.generate(**input_tokens, max_new_tokens=64)
            
        prompt_len = input_tokens["input_ids"].shape[1]
        raw_output = self.tokenizer.decode(generated_tokens[0][prompt_len:], skip_special_tokens=True).strip()
        
        # Try parsing JSON output
        try:
            # Regex to find JSON list
            match = re.search(r'\[\s*".*?"\s*(?:,\s*".*?"\s*)*\]', raw_output)
            if match:
                return json.loads(match.group(0))
            items = json.loads(raw_output)
            if isinstance(items, list):
                return [str(x) for x in items]
        except Exception:
            pass
            
        # Robust regex-based keyword extraction fallback
        print("Fallback keyword matching activated for entity extraction.")
        quoted = re.findall(r'"([^"]+)"|\'([^\']+)\'', question)
        extracted = [q[0] or q[1] for q in quoted if q[0] or q[1]]
        if not extracted:
            words = re.findall(r'[a-zA-Z]{3,}', question)
            stopwords = {"what", "who", "where", "when", "how", "the", "and", "relationship", "between", "in", "april", "sum", "maximum", "temperature", "is", "of", "are"}
            extracted = list(set([w for w in words if w.lower() not in stopwords]))
        return extracted

    def rag_query(self, question: str) -> str:
        # Step 1: Extract entities using LLM / Fallback
        extracted_entities = self.extract_entities(question)
        print(f"Extracted Entities: {extracted_entities}")
        
        # Step 2: Retrieve context from the knowledge graph
        context = self.knowledge_graph.get_context(extracted_entities)
        print(f"Retrieved Graph Context:\n{context}\n")
        
        # Step 3: Generate a response from Causal LLM
        prompt = f"""Use the following facts from a Graph Database to accurately answer the question. If the facts are not sufficient, answer as best as you can.

Facts:
{context}

Question: {question}
Answer:"""

        prompt_str = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False
        )
        
        input_tokens = self.tokenizer(prompt_str, return_tensors="pt").to(self.llm.device)
        with torch.no_grad():
            response = self.llm.generate(**input_tokens, max_new_tokens=256)
            
        prompt_len = input_tokens["input_ids"].shape[1]
        result = self.tokenizer.decode(response[0][prompt_len:], skip_special_tokens=True).strip()
        return result

def main():
    # 1. Connect to Neo4j (attempts connection, falls back to memory safely)
    connector = Neo4jConnector(uri="bolt://localhost:7687", auth=("neo4j", "password"))
    
    # 2. Build the Knowledge Graph
    kg = KnowledgeGraph(neo4j_conn=connector)
    
    # 3. Populate sample weather data for testing
    print("Populating graph database with test data...")
    kg.add_node(Entity("New York", {"region": "East Coast"}))
    kg.add_node(Entity("California", {"region": "West Coast"}))
    kg.add_node(Entity("April 2026", {"season": "Spring"}))
    
    kg.add_edge(Relation("HAS_TEMPERATURE", "New York", "April 2026", {"max_temp": 18.5}))
    kg.add_edge(Relation("HAS_TEMPERATURE", "California", "April 2026", {"max_temp": 24.0}))
    print("Graph database population complete.")
    
    # 4. Load CPU-friendly LLM (Qwen 2.5 0.5B Instruct)
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    kg_rag = KnowledgeGraphRAG(model_name, kg)
    
    # 5. Run Graph RAG Query
    question = "What is the sum of the maximum temperatures in New York and California in April 2026?"
    print(f"\nAsking Graph RAG Question: '{question}'")
    answer = kg_rag.rag_query(question)
    
    print("\n=== Final KG-RAG Answer ===")
    print(answer)
    print("==========================")
    
    # Clean up connection
    connector.close()

if __name__ == "__main__":
    main()
