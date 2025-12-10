"""
================================================================================
Shadow Supply Chain Hunter - Quick Demo (Optimized for Local LLM)
================================================================================
This script demonstrates the GraphRAG pipeline with a smaller dataset
optimized for local Ollama inference.
================================================================================
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from llama_index.core import (
    SimpleDirectoryReader,
    Settings,
    StorageContext,
    PropertyGraphIndex,
    Document,
)
from llama_index.core.indices.property_graph import SchemaLLMPathExtractor
from llama_index.core.node_parser import SentenceSplitter
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Supply chain schema (simplified for demo)
ENTITY_TYPES = ["COMPANY", "SUPPLIER", "PRODUCT", "REGION", "RISK_FACTOR"]
RELATION_TYPES = ["SUPPLIES", "DEPENDS_ON", "MANUFACTURES_IN", "EXPOSED_TO", "OPERATES_IN"]


def extract_key_sections(file_path: Path, max_chars: int = 50000) -> str:
    """Extract the most relevant sections from a 10-K filing."""
    content = file_path.read_text(errors='ignore')
    
    # Find Risk Factors section (Item 1A)
    keywords = [
        "RISK FACTORS",
        "Item 1A",
        "supply chain",
        "suppliers",
        "manufacturing",
        "Taiwan",
        "China",
        "geopolitical",
        "semiconductor",
        "TSMC",
        "concentration",
        "single source"
    ]
    
    # Extract paragraphs containing keywords
    paragraphs = content.split('\n\n')
    relevant = []
    
    for para in paragraphs:
        if any(kw.lower() in para.lower() for kw in keywords):
            if len(para) > 100:  # Skip tiny fragments
                relevant.append(para[:2000])  # Cap each paragraph
    
    result = '\n\n'.join(relevant[:30])  # Max 30 relevant paragraphs
    return result[:max_chars]


def main():
    print("\n" + "="*60)
    print("SHADOW SUPPLY CHAIN HUNTER - QUICK DEMO")
    print("="*60)
    
    # Configure Ollama
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    
    Settings.llm = Ollama(
        model=ollama_model,
        base_url=ollama_url,
        request_timeout=300.0
    )
    Settings.embed_model = OllamaEmbedding(
        model_name=ollama_model,
        base_url=ollama_url,
    )
    
    # Use smaller chunks for faster processing
    Settings.chunk_size = 512
    Settings.chunk_overlap = 50
    
    print(f"[LLM] Using Ollama: {ollama_model}")
    
    # Connect to Neo4j
    graph_store = Neo4jPropertyGraphStore(
        url=os.getenv("NEO4J_URI"),
        username=os.getenv("NEO4J_USERNAME", "neo4j"),
        password=os.getenv("NEO4J_PASSWORD"),
    )
    print("[NEO4J] Connected")
    
    # Load only relevant sections from each filing
    print("\n[LOAD] Extracting relevant sections from filings...")
    documents = []
    
    for ticker_dir in DATA_DIR.iterdir():
        if ticker_dir.is_dir():
            ticker = ticker_dir.name
            print(f"  Processing {ticker}...")
            
            for file_path in ticker_dir.rglob("*.txt"):
                if "full-submission" in file_path.name:
                    content = extract_key_sections(file_path)
                    if content:
                        doc = Document(
                            text=content,
                            metadata={"ticker": ticker, "source": file_path.name}
                        )
                        documents.append(doc)
                        print(f"    → Extracted {len(content):,} chars from {ticker}")
    
    print(f"\n[LOAD] Total documents: {len(documents)}")
    
    # Build the graph with simplified extractor
    print("\n[GRAPH] Building knowledge graph...")
    
    storage_context = StorageContext.from_defaults(
        property_graph_store=graph_store
    )
    
    kg_extractor = SchemaLLMPathExtractor(
        llm=Settings.llm,
        possible_entities=ENTITY_TYPES,
        possible_relations=RELATION_TYPES,
        strict=False,
        num_workers=1,  # Single worker for local LLM
        max_triplets_per_chunk=5,  # Fewer triplets for speed
    )
    
    index = PropertyGraphIndex.from_documents(
        documents,
        kg_extractors=[kg_extractor],
        storage_context=storage_context,
        show_progress=True,
        embed_kg_nodes=True,
    )
    
    print("\n[SUCCESS] Graph built!")
    
    # Test query
    print("\n" + "="*60)
    print("TEST QUERY")
    print("="*60)
    
    query_engine = index.as_query_engine(
        include_text=True,
        similarity_top_k=5,
    )
    
    question = "What are the geopolitical risks for Apple based on its suppliers?"
    print(f"\nQ: {question}\n")
    
    response = query_engine.query(question)
    print(f"A: {response.response}")
    
    return index


if __name__ == "__main__":
    main()
