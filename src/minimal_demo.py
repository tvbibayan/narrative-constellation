"""
================================================================================
Shadow Supply Chain Hunter - Minimal Demo (Works with Local Ollama)
================================================================================
A minimal demo that uses SimpleLLMPathExtractor for reliable local LLM support.
================================================================================
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from llama_index.core import (
    Settings,
    StorageContext,
    PropertyGraphIndex,
    Document,
)
from llama_index.core.indices.property_graph import SimpleLLMPathExtractor
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def get_sample_text() -> list[Document]:
    """
    Create sample documents with key supply chain facts.
    This ensures the demo works quickly while still showing the concept.
    """
    samples = [
        Document(
            text="""
            Apple Inc. is a major customer of Taiwan Semiconductor Manufacturing Company (TSMC).
            TSMC manufactures Apple's custom silicon chips including the A17 Pro and M3 processors.
            TSMC's primary manufacturing facilities are located in Taiwan, specifically in Hsinchu and Tainan.
            Apple depends on TSMC for its most advanced chip production using 3nm and 5nm process technology.
            This creates supply chain concentration risk for Apple related to Taiwan's geopolitical situation.
            Apple also sources components from suppliers in China, Japan, and South Korea.
            Foxconn, based in Taiwan and operating factories in China, assembles most iPhone devices.
            """,
            metadata={"ticker": "AAPL", "section": "Supply Chain Summary"}
        ),
        Document(
            text="""
            NVIDIA Corporation designs graphics processing units (GPUs) and AI accelerators.
            NVIDIA relies on TSMC for manufacturing its most advanced chips including the H100 and H200.
            TSMC produces NVIDIA's datacenter GPUs using advanced 4nm and 5nm process nodes.
            NVIDIA faces supply constraints due to limited advanced semiconductor manufacturing capacity.
            Samsung Foundry serves as a secondary manufacturing partner for some NVIDIA products.
            Geopolitical tensions between China and Taiwan pose risks to NVIDIA's supply chain.
            NVIDIA's products are critical for AI infrastructure at companies like Microsoft, Google, and Amazon.
            """,
            metadata={"ticker": "NVDA", "section": "Supply Chain Summary"}
        ),
        Document(
            text="""
            Taiwan Semiconductor Manufacturing Company (TSMC) is the world's largest contract chip manufacturer.
            TSMC operates major fabrication facilities in Taiwan including Fab 18 in Tainan.
            The company is building new fabs in Arizona, USA and Kumamoto, Japan for geographic diversification.
            TSMC's customers include Apple, NVIDIA, AMD, Qualcomm, and many other semiconductor companies.
            TSMC's leading-edge processes (3nm, 5nm) are concentrated in Taiwan facilities.
            Natural disasters in Taiwan, including earthquakes and typhoons, pose operational risks.
            Water supply constraints in Taiwan have affected TSMC operations during droughts.
            US-China tensions and potential conflict over Taiwan represent significant geopolitical risk.
            """,
            metadata={"ticker": "TSM", "section": "Risk Factors Summary"}
        ),
    ]
    return samples


def main():
    print("\n" + "="*60)
    print("SHADOW SUPPLY CHAIN HUNTER - MINIMAL DEMO")
    print("="*60)
    
    # Configure Ollama
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    
    Settings.llm = Ollama(
        model=ollama_model,
        base_url=ollama_url,
        request_timeout=600.0,
        temperature=0.0,
    )
    Settings.embed_model = OllamaEmbedding(
        model_name=ollama_model,
        base_url=ollama_url,
    )
    Settings.chunk_size = 256
    Settings.chunk_overlap = 20
    
    print(f"[LLM] Using Ollama: {ollama_model}")
    
    # Connect to Neo4j
    print("[NEO4J] Connecting...")
    graph_store = Neo4jPropertyGraphStore(
        url=os.getenv("NEO4J_URI"),
        username=os.getenv("NEO4J_USERNAME", "neo4j"),
        password=os.getenv("NEO4J_PASSWORD"),
    )
    print("[NEO4J] Connected")
    
    # Get sample documents
    documents = get_sample_text()
    print(f"[LOAD] {len(documents)} sample documents ready")
    
    # Build graph with simple extractor (more reliable with local LLMs)
    print("\n[GRAPH] Building knowledge graph...")
    print("        (This takes 3-5 minutes with local Ollama)")
    
    storage_context = StorageContext.from_defaults(
        property_graph_store=graph_store
    )
    
    # SimpleLLMPathExtractor works better with local LLMs
    kg_extractor = SimpleLLMPathExtractor(
        llm=Settings.llm,
        max_paths_per_chunk=10,
        num_workers=1,
    )
    
    index = PropertyGraphIndex.from_documents(
        documents,
        kg_extractors=[kg_extractor],
        storage_context=storage_context,
        show_progress=True,
        embed_kg_nodes=False,  # Disable embeddings to avoid async issues
    )
    
    print("\n" + "="*60)
    print("[SUCCESS] Knowledge graph built!")
    print("="*60)
    
    # Run test queries
    query_engine = index.as_query_engine(
        include_text=True,
        similarity_top_k=5,
    )
    
    questions = [
        "What are the geopolitical risks for Apple based on its suppliers?",
        "Which companies depend on TSMC?",
        "What risks does Taiwan pose to the semiconductor supply chain?",
    ]
    
    for q in questions:
        print(f"\n{'='*60}")
        print(f"Q: {q}")
        print("="*60)
        response = query_engine.query(q)
        print(f"\nA: {response.response}\n")
    
    # Show graph stats
    print("\n" + "="*60)
    print("GRAPH STATISTICS")
    print("="*60)
    print(f"View your graph at: http://localhost:7474")
    print("Username: neo4j")
    print("Password: supplychain123")
    print("\nTry this Cypher query:")
    print("  MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 50")
    
    return index


if __name__ == "__main__":
    main()
