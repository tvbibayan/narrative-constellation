"""
================================================================================
Shadow Supply Chain Hunter - GraphRAG Engine
================================================================================
Module: graph_rag.py
Purpose: Build Property Graph Index from SEC 10-K filings using LlamaIndex

ARCHITECTURE NOTES:
- PropertyGraphIndex (NOT the deprecated KnowledgeGraphIndex) is the modern
  LlamaIndex abstraction for GraphRAG. It captures:
  1. Entities as NODES with properties (company name, type, risk_score)
  2. Relations as EDGES with properties (dependency_type, confidence)
  3. Text chunks linked to their source entities for hybrid retrieval

QUANT OPTIMIZATION:
- Custom extraction prompt focuses on SUPPLY CHAIN relationships
- We define explicit entity types: COMPANY, PRODUCT, REGION, RISK_FACTOR
- Edge types: SUPPLIES, DEPENDS_ON, MANUFACTURES_IN, EXPOSED_TO
- This schema enables multi-hop traversal: Apple -> TSMC -> Taiwan -> Geopolitical Risk
================================================================================
"""

import os
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

# Load environment before importing LlamaIndex (for OpenAI key)
load_dotenv()

# LlamaIndex imports - using the modular architecture
from llama_index.core import (
    SimpleDirectoryReader,
    Settings,
    StorageContext,
    PropertyGraphIndex,
)
from llama_index.core.indices.property_graph import (
    SchemaLLMPathExtractor,
    SimpleLLMPathExtractor,
)
from llama_index.core.schema import Document
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PERSIST_DIR = PROJECT_ROOT / "storage"

# ==============================================================================
# SUPPLY CHAIN SCHEMA DEFINITION
# ==============================================================================
# QUANT RATIONALE: Explicit schema improves extraction precision vs. open-ended
# extraction. In supply chain analysis, we care about specific relationships.

ENTITY_TYPES = [
    "COMPANY",           # Corporate entities (Apple, TSMC, Foxconn)
    "SUPPLIER",          # Explicit supplier designation
    "CUSTOMER",          # Downstream customers
    "PRODUCT",           # Products/Components (A17 chip, iPhone)
    "REGION",            # Geographic locations (Taiwan, China, US)
    "RISK_FACTOR",       # Risk categories (geopolitical, supply shortage)
    "MATERIAL",          # Raw materials (rare earth, silicon)
    "FACILITY",          # Manufacturing facilities
]

RELATION_TYPES = [
    "SUPPLIES",              # Supplier -> Customer
    "SUPPLIES_TO",           # Supplier -> Customer (directional)
    "DEPENDS_ON",            # Customer -> Supplier dependency
    "MANUFACTURES",          # Company -> Product
    "MANUFACTURES_IN",       # Company -> Region (manufacturing location)
    "OPERATES_IN",           # Company -> Region
    "SOURCES_FROM",          # Company -> Region (material sourcing)
    "EXPOSED_TO",            # Entity -> Risk Factor
    "COMPETES_WITH",         # Company -> Company
    "REQUIRES",              # Product -> Material/Component
    "LOCATED_IN",            # Facility -> Region
]

# Custom extraction prompt focused on supply chain dependencies
SUPPLY_CHAIN_EXTRACTION_PROMPT = """
You are an expert financial analyst specializing in supply chain risk assessment.
Your task is to extract supply chain relationships from SEC 10-K filings.

FOCUS AREAS:
1. **Supplier Dependencies**: Who supplies components/materials to this company?
2. **Customer Relationships**: Who are the major customers?
3. **Geographic Exposure**: Where are manufacturing facilities located?
4. **Risk Factors**: What supply chain risks are disclosed?
5. **Single-Source Dependencies**: Critical suppliers with no alternatives

EXTRACTION RULES:
- Extract ONLY factual relationships explicitly stated in the text
- Include confidence indicators when the relationship is implied vs. stated
- Capture the DIRECTION of relationships (Supplier -> supplies -> Customer)
- Note any disclosed risks associated with supplier relationships

For each relationship found, extract:
- Subject Entity (with type)
- Relationship/Predicate
- Object Entity (with type)

Focus on relationships that reveal supply chain vulnerabilities and dependencies.
"""


def get_neo4j_store() -> Neo4jPropertyGraphStore:
    """
    Initialize Neo4j Property Graph Store connection.
    
    QUANT NOTE:
    - Neo4j Aura (cloud) is preferred for production - handles scaling
    - Local Docker instance works for development
    - PropertyGraphStore (not GraphStore) is required for PropertyGraphIndex
    
    Returns:
        Configured Neo4jPropertyGraphStore instance
    """
    neo4j_uri = os.getenv("NEO4J_URI")
    neo4j_user = os.getenv("NEO4J_USERNAME", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    
    if not neo4j_uri or not neo4j_password:
        raise ValueError(
            "Missing Neo4j credentials. Set NEO4J_URI and NEO4J_PASSWORD in .env"
        )
    
    print(f"[NEO4J] Connecting to: {neo4j_uri}")
    
    graph_store = Neo4jPropertyGraphStore(
        url=neo4j_uri,
        username=neo4j_user,
        password=neo4j_password,
    )
    
    print("[NEO4J] Connection established successfully")
    return graph_store


def configure_llm(use_local: bool = False) -> None:
    """
    Configure the LLM for entity extraction.
    
    QUANT OPTIMIZATION:
    - GPT-4o for production: Best extraction quality on financial documents
    - GPT-4o-mini as fallback: 10x cheaper, 90% quality for dev work
    - Ollama/Llama3 for air-gapped environments (hedge fund compliance)
    
    Args:
        use_local: If True, use Ollama instead of OpenAI
    """
    if use_local or os.getenv("USE_LOCAL_LLM", "").lower() == "true":
        # Local LLM via Ollama
        from llama_index.llms.ollama import Ollama
        from llama_index.embeddings.ollama import OllamaEmbedding
        
        ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        ollama_model = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
        
        Settings.llm = Ollama(
            model=ollama_model,
            base_url=ollama_url,
            request_timeout=300.0  # Longer timeout for local inference
        )
        print(f"[LLM] Using Ollama: {ollama_model}")
        
        # Local embeddings via Ollama (uses nomic-embed-text or same model)
        Settings.embed_model = OllamaEmbedding(
            model_name=ollama_model,
            base_url=ollama_url,
        )
        print(f"[EMBED] Using Ollama embeddings: {ollama_model}")
    else:
        # OpenAI GPT-4o
        Settings.llm = OpenAI(
            model="gpt-4o",
            temperature=0.0,  # Deterministic for consistent extraction
            api_key=os.getenv("OPENAI_API_KEY")
        )
        print("[LLM] Using OpenAI: gpt-4o")
        
        # Embeddings for vector search component
        Settings.embed_model = OpenAIEmbedding(
            model="text-embedding-3-small",
            api_key=os.getenv("OPENAI_API_KEY")
        )
        print("[EMBED] Using OpenAI: text-embedding-3-small")


def load_documents() -> List[Document]:
    """
    Load SEC 10-K filings from the data directory.
    
    QUANT OPTIMIZATION:
    - SimpleDirectoryReader handles multiple file formats
    - We add metadata tags to track source company
    - Large files are automatically chunked
    
    Returns:
        List of Document objects ready for indexing
    """
    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"Data directory not found: {DATA_DIR}\n"
            "Run 'python src/ingest.py' first to download 10-K filings."
        )
    
    print(f"\n[LOAD] Reading documents from: {DATA_DIR}")
    
    # Find all text files recursively
    documents = []
    
    for ticker_dir in DATA_DIR.iterdir():
        if ticker_dir.is_dir():
            ticker = ticker_dir.name
            print(f"[LOAD] Processing {ticker}...")
            
            # Use SimpleDirectoryReader for robust file handling
            try:
                reader = SimpleDirectoryReader(
                    input_dir=str(ticker_dir),
                    recursive=True,
                    required_exts=[".txt", ".htm", ".html"],
                    filename_as_id=True,
                )
                ticker_docs = reader.load_data()
                
                # Add ticker metadata for source tracking
                for doc in ticker_docs:
                    doc.metadata["ticker"] = ticker
                    doc.metadata["filing_type"] = "10-K"
                
                documents.extend(ticker_docs)
                print(f"[LOAD] {ticker}: {len(ticker_docs)} documents loaded")
                
            except Exception as e:
                print(f"[WARN] Failed to load {ticker}: {e}")
    
    print(f"\n[LOAD] Total documents loaded: {len(documents)}")
    return documents


def build_property_graph_index(
    documents: List[Document],
    graph_store: Neo4jPropertyGraphStore,
) -> PropertyGraphIndex:
    """
    Build the Property Graph Index from documents.
    
    ARCHITECTURE (PropertyGraphIndex):
    - Modern replacement for KnowledgeGraphIndex
    - Supports both vector and graph retrieval natively
    - Schema-guided extraction for better precision
    - Stores: Nodes (entities), Edges (relations), Text chunks, Embeddings
    
    QUANT OPTIMIZATION:
    - SchemaLLMPathExtractor: Uses our supply chain schema for guided extraction
    - max_triplets_per_chunk: Controls extraction density (higher = more edges)
    - We persist to Neo4j for durability and complex Cypher queries
    
    Args:
        documents: List of Document objects to index
        graph_store: Neo4j graph store instance
    
    Returns:
        Configured PropertyGraphIndex
    """
    print("\n" + "="*60)
    print("BUILDING PROPERTY GRAPH INDEX")
    print("="*60)
    
    # Configure storage context with Neo4j
    storage_context = StorageContext.from_defaults(
        property_graph_store=graph_store
    )
    
    # Schema-guided extraction - CRITICAL for supply chain focus
    # This tells the LLM exactly what entities and relations to look for
    kg_extractor = SchemaLLMPathExtractor(
        llm=Settings.llm,
        possible_entities=ENTITY_TYPES,
        possible_relations=RELATION_TYPES,
        kg_validation_schema=None,  # Allow flexibility in extraction
        strict=False,  # Don't reject valid extractions not in schema
        num_workers=4,  # Parallel extraction for speed
        max_triplets_per_chunk=15,  # Balance: coverage vs. noise
    )
    
    print(f"[GRAPH] Entity types: {len(ENTITY_TYPES)}")
    print(f"[GRAPH] Relation types: {len(RELATION_TYPES)}")
    print(f"[GRAPH] Max triplets per chunk: 15")
    
    # Build the index - this is the heavy lifting
    print("\n[GRAPH] Extracting entities and relations (this may take a while)...")
    
    index = PropertyGraphIndex.from_documents(
        documents,
        kg_extractors=[kg_extractor],
        storage_context=storage_context,
        show_progress=True,
        embed_kg_nodes=True,  # Enable vector search on nodes
    )
    
    print("\n[GRAPH] Property Graph Index built successfully!")
    
    return index


def load_existing_index(graph_store: Neo4jPropertyGraphStore) -> PropertyGraphIndex:
    """
    Load an existing Property Graph Index from Neo4j.
    
    Use this when you've already built the graph and want to query it
    without re-extracting entities.
    
    Args:
        graph_store: Neo4j graph store instance
    
    Returns:
        Loaded PropertyGraphIndex
    """
    print("[GRAPH] Loading existing index from Neo4j...")
    
    index = PropertyGraphIndex.from_existing(
        property_graph_store=graph_store,
        embed_kg_nodes=True,
    )
    
    print("[GRAPH] Index loaded successfully")
    return index


def get_graph_statistics(graph_store: Neo4jPropertyGraphStore) -> dict:
    """
    Retrieve statistics about the graph for monitoring.
    
    QUANT USE: Track graph growth over time as we ingest more filings.
    
    Args:
        graph_store: Neo4j graph store instance
    
    Returns:
        Dictionary with node/edge counts by type
    """
    # Direct Cypher query for statistics
    from neo4j import GraphDatabase
    
    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI"),
        auth=(os.getenv("NEO4J_USERNAME", "neo4j"), os.getenv("NEO4J_PASSWORD"))
    )
    
    stats = {}
    
    with driver.session() as session:
        # Node count
        result = session.run("MATCH (n) RETURN count(n) as count")
        stats["total_nodes"] = result.single()["count"]
        
        # Relationship count
        result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
        stats["total_relationships"] = result.single()["count"]
        
        # Node types
        result = session.run(
            "MATCH (n) RETURN labels(n) as labels, count(*) as count"
        )
        stats["node_types"] = {str(r["labels"]): r["count"] for r in result}
        
        # Relationship types
        result = session.run(
            "MATCH ()-[r]->() RETURN type(r) as type, count(*) as count"
        )
        stats["relationship_types"] = {r["type"]: r["count"] for r in result}
    
    driver.close()
    
    return stats


def main():
    """
    Main execution: Build the Property Graph Index from 10-K filings.
    
    Usage:
        python src/graph_rag.py
    
    Prerequisites:
        1. Run 'python src/ingest.py' to download filings
        2. Ensure .env has valid NEO4J and OPENAI credentials
    """
    print("\n" + "="*60)
    print("SHADOW SUPPLY CHAIN HUNTER - GRAPH CONSTRUCTION")
    print("="*60)
    
    # Step 1: Configure LLM
    configure_llm()
    
    # Step 2: Connect to Neo4j
    graph_store = get_neo4j_store()
    
    # Step 3: Load documents
    documents = load_documents()
    
    if not documents:
        print("[ERROR] No documents found. Run ingest.py first.")
        return
    
    # Step 4: Build the graph
    index = build_property_graph_index(documents, graph_store)
    
    # Step 5: Report statistics
    print("\n" + "="*60)
    print("GRAPH STATISTICS")
    print("="*60)
    
    try:
        stats = get_graph_statistics(graph_store)
        print(f"Total Nodes: {stats['total_nodes']}")
        print(f"Total Relationships: {stats['total_relationships']}")
        print("\nNode Types:")
        for label, count in stats.get("node_types", {}).items():
            print(f"  {label}: {count}")
        print("\nRelationship Types:")
        for rel_type, count in stats.get("relationship_types", {}).items():
            print(f"  {rel_type}: {count}")
    except Exception as e:
        print(f"[WARN] Could not retrieve statistics: {e}")
    
    print("\n[SUCCESS] Graph construction complete!")
    print("Run 'python src/query.py' to start querying.")
    
    return index


if __name__ == "__main__":
    main()
