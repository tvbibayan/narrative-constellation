"""
================================================================================
Shadow Supply Chain Hunter - Hybrid Query Engine
================================================================================
Module: query.py
Purpose: Natural language reasoning over the supply chain knowledge graph

ARCHITECTURE - HYBRID RETRIEVAL:
This module implements a sophisticated two-stage retrieval strategy:

1. VECTOR SEARCH (Semantic Similarity):
   - Embeds the query and finds semantically similar text chunks
   - Good for: "What does Apple say about supply chain risks?"
   - Returns relevant passages from the original 10-K text

2. GRAPH TRAVERSAL (Structural Reasoning):
   - Navigates the knowledge graph following entity relationships
   - Good for: "Which companies depend on TSMC?" (multi-hop)
   - Returns connected entities and their relationships

QUANT OPTIMIZATION:
- Hybrid search combines both signals for comprehensive answers
- Graph traversal enables "hidden" dependency discovery
- Custom Cypher queries for complex supply chain analysis
================================================================================
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple

from dotenv import load_dotenv

# Load environment
load_dotenv()

# LlamaIndex imports
from llama_index.core import Settings
from llama_index.core.indices.property_graph import PropertyGraphIndex
from llama_index.core.retrievers import (
    QueryFusionRetriever,
)
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# Direct Neo4j for custom Cypher queries
from neo4j import GraphDatabase

# Project imports
from graph_rag import get_neo4j_store, configure_llm, load_existing_index

# Constants
PROJECT_ROOT = Path(__file__).parent.parent


class SupplyChainQueryEngine:
    """
    Hybrid Query Engine for Supply Chain Analysis.
    
    CAPABILITIES:
    1. Natural language questions -> Graph + Vector retrieval
    2. Direct Cypher queries for complex traversals
    3. Supply chain risk assessment queries
    4. Multi-hop relationship discovery
    """
    
    def __init__(self, index: PropertyGraphIndex = None):
        """
        Initialize the query engine.
        
        Args:
            index: Pre-loaded PropertyGraphIndex (optional)
        """
        # Configure LLM for response generation
        configure_llm()
        
        # Connect to Neo4j
        self.graph_store = get_neo4j_store()
        
        # Load or use provided index
        if index:
            self.index = index
        else:
            self.index = load_existing_index(self.graph_store)
        
        # Build query engine with hybrid retrieval
        self._setup_query_engine()
        
        # Direct Neo4j driver for custom queries
        self.neo4j_driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI"),
            auth=(os.getenv("NEO4J_USERNAME", "neo4j"), os.getenv("NEO4J_PASSWORD"))
        )
        
        print("[QUERY] Supply Chain Query Engine initialized")
    
    def _setup_query_engine(self):
        """
        Configure the hybrid retrieval query engine.
        
        ARCHITECTURE:
        - PropertyGraphIndex.as_query_engine() provides built-in hybrid search
        - It automatically combines vector similarity + graph traversal
        - Response synthesizer merges results into coherent answer
        """
        # Response synthesizer - how to combine retrieved info
        self.response_synthesizer = get_response_synthesizer(
            response_mode="tree_summarize",  # Hierarchical summarization
            use_async=False,
        )
        
        # The PropertyGraphIndex query engine handles hybrid retrieval internally
        self.query_engine = self.index.as_query_engine(
            include_text=True,      # Include source text chunks
            response_mode="tree_summarize",
            embedding_mode="hybrid",  # Vector + keyword
            similarity_top_k=10,    # Top chunks to retrieve
            graph_store_query_depth=2,  # Hops for graph traversal
        )
        
        print("[QUERY] Hybrid query engine configured")
        print("  - Vector search: similarity_top_k=10")
        print("  - Graph traversal: depth=2 hops")
    
    def query(self, question: str) -> str:
        """
        Execute a natural language query with hybrid retrieval.
        
        This is the main entry point for questions like:
        "What are the geopolitical risks for Apple based on its suppliers?"
        
        Args:
            question: Natural language question
        
        Returns:
            Synthesized answer with source attribution
        """
        print(f"\n{'='*60}")
        print(f"QUERY: {question}")
        print('='*60)
        
        response = self.query_engine.query(question)
        
        print(f"\n[ANSWER]\n{response.response}")
        
        # Show source nodes for transparency
        if hasattr(response, 'source_nodes') and response.source_nodes:
            print(f"\n[SOURCES] {len(response.source_nodes)} chunks retrieved")
            for i, node in enumerate(response.source_nodes[:3]):
                ticker = node.metadata.get('ticker', 'Unknown')
                print(f"  {i+1}. [{ticker}] Score: {node.score:.3f}")
        
        return response.response
    
    def find_suppliers(self, company: str) -> List[dict]:
        """
        Find all suppliers for a given company using graph traversal.
        
        QUANT USE CASE: Identify supply chain concentration risk
        
        Args:
            company: Company name (e.g., "Apple", "NVIDIA")
        
        Returns:
            List of supplier entities with relationship details
        """
        cypher = """
        MATCH (supplier)-[r:SUPPLIES|SUPPLIES_TO]->(company)
        WHERE toLower(company.name) CONTAINS toLower($company_name)
           OR toLower(company.id) CONTAINS toLower($company_name)
        RETURN supplier.name as supplier, 
               supplier.id as supplier_id,
               type(r) as relationship,
               company.name as customer
        LIMIT 50
        """
        
        return self._execute_cypher(cypher, {"company_name": company})
    
    def find_dependencies(self, company: str) -> List[dict]:
        """
        Find what a company depends on (reverse supply chain).
        
        QUANT USE CASE: Identify single-source dependencies
        
        Args:
            company: Company name
        
        Returns:
            List of dependencies
        """
        cypher = """
        MATCH (company)-[r:DEPENDS_ON|SOURCES_FROM|REQUIRES]->(dependency)
        WHERE toLower(company.name) CONTAINS toLower($company_name)
           OR toLower(company.id) CONTAINS toLower($company_name)
        RETURN company.name as company,
               type(r) as relationship,
               dependency.name as dependency,
               labels(dependency) as dependency_type
        LIMIT 50
        """
        
        return self._execute_cypher(cypher, {"company_name": company})
    
    def find_geographic_exposure(self, region: str) -> List[dict]:
        """
        Find all companies with exposure to a specific region.
        
        QUANT USE CASE: Geopolitical risk assessment
        "Which companies have manufacturing exposure to Taiwan?"
        
        Args:
            region: Geographic region (e.g., "Taiwan", "China")
        
        Returns:
            List of companies with regional exposure
        """
        cypher = """
        MATCH (company)-[r:MANUFACTURES_IN|OPERATES_IN|SOURCES_FROM|LOCATED_IN]->(region)
        WHERE toLower(region.name) CONTAINS toLower($region_name)
           OR toLower(region.id) CONTAINS toLower($region_name)
        RETURN company.name as company,
               type(r) as exposure_type,
               region.name as region
        LIMIT 50
        """
        
        return self._execute_cypher(cypher, {"region_name": region})
    
    def multi_hop_risk_analysis(self, company: str, hops: int = 2) -> List[dict]:
        """
        Perform multi-hop traversal to find hidden supply chain risks.
        
        QUANT ALPHA: This is where GraphRAG shines. We can discover:
        - Apple depends on TSMC (hop 1)
        - TSMC manufactures in Taiwan (hop 2)
        - Taiwan has geopolitical risk exposure (hop 3)
        
        This reveals risks not explicitly stated in Apple's own 10-K.
        
        Args:
            company: Starting company
            hops: Number of relationship hops (default: 2)
        
        Returns:
            List of risk paths discovered
        """
        # Variable-length path query
        cypher = f"""
        MATCH path = (start)-[*1..{hops}]->(risk:RISK_FACTOR)
        WHERE toLower(start.name) CONTAINS toLower($company_name)
           OR toLower(start.id) CONTAINS toLower($company_name)
        RETURN [node in nodes(path) | node.name] as path_nodes,
               [rel in relationships(path) | type(rel)] as path_rels,
               risk.name as risk_factor
        LIMIT 25
        """
        
        return self._execute_cypher(cypher, {"company_name": company})
    
    def find_supply_chain_path(self, source: str, target: str) -> List[dict]:
        """
        Find the shortest path between two entities in the supply chain.
        
        QUANT USE CASE: Trace dependency chains
        "How is Apple connected to rare earth mining?"
        
        Args:
            source: Starting entity
            target: Target entity
        
        Returns:
            Paths connecting the entities
        """
        cypher = """
        MATCH path = shortestPath(
            (source)-[*..5]-(target)
        )
        WHERE (toLower(source.name) CONTAINS toLower($source_name)
               OR toLower(source.id) CONTAINS toLower($source_name))
          AND (toLower(target.name) CONTAINS toLower($target_name)
               OR toLower(target.id) CONTAINS toLower($target_name))
        RETURN [node in nodes(path) | node.name] as path_nodes,
               [rel in relationships(path) | type(rel)] as path_rels,
               length(path) as path_length
        LIMIT 10
        """
        
        return self._execute_cypher(cypher, {
            "source_name": source,
            "target_name": target
        })
    
    def _execute_cypher(self, cypher: str, params: dict) -> List[dict]:
        """
        Execute a Cypher query and return results.
        
        Args:
            cypher: Cypher query string
            params: Query parameters
        
        Returns:
            List of result records as dictionaries
        """
        print(f"\n[CYPHER] Executing graph query...")
        
        with self.neo4j_driver.session() as session:
            result = session.run(cypher, params)
            records = [dict(record) for record in result]
        
        print(f"[CYPHER] Found {len(records)} results")
        return records
    
    def get_graph_summary(self) -> dict:
        """
        Get a summary of the knowledge graph contents.
        
        Returns:
            Dictionary with graph statistics
        """
        cypher_nodes = "MATCH (n) RETURN count(n) as count"
        cypher_rels = "MATCH ()-[r]->() RETURN count(r) as count"
        cypher_types = """
            MATCH (n) 
            WITH labels(n) as labels, count(*) as count 
            RETURN labels, count 
            ORDER BY count DESC
        """
        
        with self.neo4j_driver.session() as session:
            nodes = session.run(cypher_nodes).single()["count"]
            rels = session.run(cypher_rels).single()["count"]
            types = [dict(r) for r in session.run(cypher_types)]
        
        return {
            "total_nodes": nodes,
            "total_relationships": rels,
            "node_types": types
        }
    
    def close(self):
        """Clean up database connections."""
        self.neo4j_driver.close()


def interactive_mode(engine: SupplyChainQueryEngine):
    """
    Run an interactive query session.
    
    Allows the user to ask multiple questions without reloading.
    """
    print("\n" + "="*60)
    print("SHADOW SUPPLY CHAIN HUNTER - INTERACTIVE MODE")
    print("="*60)
    print("Type your question and press Enter.")
    print("Commands: 'quit' to exit, 'stats' for graph summary")
    print("="*60)
    
    while True:
        try:
            question = input("\n🔍 Question: ").strip()
            
            if not question:
                continue
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("\n[EXIT] Closing connections...")
                break
            
            if question.lower() == 'stats':
                summary = engine.get_graph_summary()
                print(f"\n[GRAPH STATS]")
                print(f"  Nodes: {summary['total_nodes']}")
                print(f"  Relationships: {summary['total_relationships']}")
                continue
            
            engine.query(question)
            
        except KeyboardInterrupt:
            print("\n[EXIT] Interrupted by user")
            break
        except Exception as e:
            print(f"[ERROR] {e}")
    
    engine.close()


def main():
    """
    Main execution: Run test queries against the supply chain graph.
    
    This demonstrates the hybrid retrieval capabilities:
    1. Natural language queries
    2. Direct graph traversal
    3. Multi-hop risk analysis
    """
    print("\n" + "="*60)
    print("SHADOW SUPPLY CHAIN HUNTER - REASONING ENGINE")
    print("="*60)
    
    # Initialize the query engine
    engine = SupplyChainQueryEngine()
    
    # Get graph summary first
    print("\n[INFO] Graph Summary:")
    try:
        summary = engine.get_graph_summary()
        print(f"  Total Nodes: {summary['total_nodes']}")
        print(f"  Total Relationships: {summary['total_relationships']}")
    except Exception as e:
        print(f"  Warning: Could not get summary - {e}")
    
    # ===========================================================================
    # TEST QUERY 1: Geopolitical Risk Assessment (Hybrid Retrieval)
    # ===========================================================================
    # This is the key test case - combines:
    # - Vector search to find relevant risk disclosure text
    # - Graph traversal to find supply chain dependencies
    # ===========================================================================
    
    print("\n" + "="*60)
    print("TEST QUERY 1: Geopolitical Risk Assessment")
    print("="*60)
    
    question = "What are the geopolitical risks for Apple based on its suppliers?"
    response = engine.query(question)
    
    # ===========================================================================
    # TEST QUERY 2: Direct Graph Traversal - Supplier Discovery
    # ===========================================================================
    
    print("\n" + "="*60)
    print("TEST QUERY 2: Supplier Discovery (Graph Traversal)")
    print("="*60)
    
    for company in ["Apple", "NVIDIA", "TSMC"]:
        print(f"\n[SUPPLIERS for {company}]")
        suppliers = engine.find_suppliers(company)
        if suppliers:
            for s in suppliers[:5]:
                print(f"  - {s.get('supplier', 'N/A')} -> {s.get('relationship', 'N/A')} -> {company}")
        else:
            print("  No suppliers found in graph")
    
    # ===========================================================================
    # TEST QUERY 3: Geographic Exposure Analysis
    # ===========================================================================
    
    print("\n" + "="*60)
    print("TEST QUERY 3: Geographic Exposure (Taiwan)")
    print("="*60)
    
    taiwan_exposure = engine.find_geographic_exposure("Taiwan")
    if taiwan_exposure:
        print(f"Found {len(taiwan_exposure)} entities with Taiwan exposure:")
        for e in taiwan_exposure[:10]:
            print(f"  - {e.get('company', 'N/A')} [{e.get('exposure_type', 'N/A')}]")
    else:
        print("No Taiwan exposure found in graph")
    
    # ===========================================================================
    # TEST QUERY 4: Multi-Hop Risk Path Discovery
    # ===========================================================================
    
    print("\n" + "="*60)
    print("TEST QUERY 4: Multi-Hop Risk Paths (Apple)")
    print("="*60)
    
    risk_paths = engine.multi_hop_risk_analysis("Apple", hops=3)
    if risk_paths:
        print(f"Found {len(risk_paths)} risk paths:")
        for path in risk_paths[:5]:
            nodes = " -> ".join(str(n) for n in path.get('path_nodes', []))
            print(f"  Path: {nodes}")
            print(f"  Risk: {path.get('risk_factor', 'N/A')}")
    else:
        print("No risk paths found")
    
    # ===========================================================================
    # TEST QUERY 5: Supply Chain Path Finding
    # ===========================================================================
    
    print("\n" + "="*60)
    print("TEST QUERY 5: Supply Chain Path (Apple -> TSMC)")
    print("="*60)
    
    paths = engine.find_supply_chain_path("Apple", "TSMC")
    if paths:
        print(f"Found {len(paths)} paths:")
        for p in paths:
            nodes = " -> ".join(str(n) for n in p.get('path_nodes', []))
            print(f"  {nodes}")
    else:
        print("No paths found between Apple and TSMC")
    
    # ===========================================================================
    # Optional: Interactive Mode
    # ===========================================================================
    
    print("\n" + "="*60)
    print("Would you like to enter interactive mode? (y/n)")
    print("="*60)
    
    try:
        choice = input().strip().lower()
        if choice == 'y':
            interactive_mode(engine)
        else:
            engine.close()
            print("\n[DONE] Query engine closed.")
    except:
        engine.close()


if __name__ == "__main__":
    main()
