"""
================================================================================
Shadow Supply Chain Hunter - Groq Demo (Fast & Free)
================================================================================
Uses Groq's blazing fast LLM inference for entity extraction.
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
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

PROJECT_ROOT = Path(__file__).parent.parent


def get_sample_text() -> list[Document]:
    """Sample documents with key supply chain facts."""
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
    print("SHADOW SUPPLY CHAIN HUNTER - GROQ DEMO")
    print("="*60)
    
    # Set Groq API key
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        print("\n[ERROR] No Groq API key found!")
        print("Set GROQ_API_KEY in your .env file")
        return
    
    # Configure Groq LLM (llama-3.3-70b is the latest)
    Settings.llm = Groq(
        model="llama-3.3-70b-versatile",
        api_key=groq_key,
        temperature=0.0,
    )
    
    # Use local HuggingFace embeddings (free, no API needed)
    print("[EMBED] Loading local embeddings model...")
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    Settings.chunk_size = 512
    Settings.chunk_overlap = 50
    
    print("[LLM] Using Groq llama-3.1-70b-versatile")
    print("[EMBED] Using sentence-transformers/all-MiniLM-L6-v2")
    
    # Connect to Neo4j
    print("[NEO4J] Connecting...")
    graph_store = Neo4jPropertyGraphStore(
        url=os.getenv("NEO4J_URI", "neo4j://localhost:7687"),
        username=os.getenv("NEO4J_USERNAME", "neo4j"),
        password=os.getenv("NEO4J_PASSWORD", "supplychain123"),
    )
    print("[NEO4J] Connected")
    
    # Get sample documents
    documents = get_sample_text()
    print(f"[LOAD] {len(documents)} sample documents ready")
    
    # Build graph
    print("\n[GRAPH] Building knowledge graph...")
    print("        (Groq is FAST - should take < 30 seconds)")
    
    storage_context = StorageContext.from_defaults(
        property_graph_store=graph_store
    )
    
    kg_extractor = SimpleLLMPathExtractor(
        llm=Settings.llm,
        max_paths_per_chunk=10,
        num_workers=1,  # Sequential to avoid rate limits
    )
    
    index = PropertyGraphIndex.from_documents(
        documents,
        kg_extractors=[kg_extractor],
        storage_context=storage_context,
        show_progress=True,
        embed_kg_nodes=True,
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
    print("GRAPH VISUALIZATION")
    print("="*60)
    print(f"View your graph at: http://localhost:7474")
    print("Username: neo4j")
    print("Password: supplychain123")
    print("\nTry this Cypher query to see the supply chain:")
    print("  MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 50")
    
    return index


if __name__ == "__main__":
    main()
