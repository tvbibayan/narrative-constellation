"""
================================================================================
🕵️ SHADOW SUPPLY CHAIN HUNTER
================================================================================
Interactive GraphRAG Dashboard for SEC 10-K Supply Chain Analysis
================================================================================
"""

import streamlit as st
import os
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
from pyvis.network import Network
import tempfile
import streamlit.components.v1 as components

# Load environment
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

# Page config
st.set_page_config(
    page_title="Shadow Supply Chain Hunter",
    page_icon="🕵️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme
st.markdown("""
<style>
    /* Main container */
    .main {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
    }
    
    /* Headers */
    h1, h2, h3 {
        background: linear-gradient(90deg, #00d4ff, #7c3aed, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }
    
    /* Metric cards */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #1e1e3f 0%, #2d2d5a 100%);
        border: 1px solid #7c3aed;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 15px rgba(124, 58, 237, 0.3);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0f23 0%, #1a1a2e 100%);
        border-right: 1px solid #7c3aed;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #7c3aed, #00d4ff);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 25px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 20px rgba(124, 58, 237, 0.5);
    }
    
    /* Text input */
    .stTextInput>div>div>input {
        background: #1e1e3f;
        border: 1px solid #7c3aed;
        border-radius: 8px;
        color: white;
    }
    
    /* Success/Info boxes */
    .stAlert {
        background: linear-gradient(135deg, #1e1e3f 0%, #2d2d5a 100%);
        border-left: 4px solid #00d4ff;
        border-radius: 8px;
    }
    
    /* Code blocks */
    .stCodeBlock {
        background: #0f0f23 !important;
        border: 1px solid #7c3aed;
        border-radius: 8px;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: linear-gradient(90deg, #1e1e3f, #2d2d5a);
        border-radius: 8px;
    }
    
    /* Divider */
    hr {
        border-color: #7c3aed;
    }
    
    /* Card styling */
    .risk-card {
        background: linear-gradient(135deg, #2d1f3d 0%, #1a1a2e 100%);
        border: 1px solid #f472b6;
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
    }
    
    .supplier-card {
        background: linear-gradient(135deg, #1f2d3d 0%, #1a1a2e 100%);
        border: 1px solid #00d4ff;
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


def get_sample_documents() -> list[Document]:
    """Sample documents with supply chain facts."""
    return [
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
            metadata={"ticker": "AAPL", "company": "Apple Inc.", "section": "Supply Chain"}
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
            metadata={"ticker": "NVDA", "company": "NVIDIA Corporation", "section": "Supply Chain"}
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
            metadata={"ticker": "TSM", "company": "TSMC", "section": "Risk Factors"}
        ),
    ]


@st.cache_resource
def init_llm():
    """Initialize LLM and embeddings."""
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        return None, None
    
    llm = Groq(
        model="llama-3.3-70b-versatile",
        api_key=groq_key,
        temperature=0.0,
    )
    
    embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    return llm, embed_model


@st.cache_resource
def connect_neo4j():
    """Connect to Neo4j."""
    try:
        graph_store = Neo4jPropertyGraphStore(
            url=os.getenv("NEO4J_URI", "neo4j://localhost:7687"),
            username=os.getenv("NEO4J_USERNAME", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", "supplychain123"),
        )
        return graph_store
    except Exception as e:
        return None


def get_graph_data(graph_store):
    """Fetch graph data from Neo4j for visualization."""
    try:
        query = """
        MATCH (n)-[r]->(m)
        RETURN n.id as source, type(r) as relationship, m.id as target
        LIMIT 100
        """
        result = graph_store._driver.execute_query(query)
        edges = []
        for record in result.records:
            edges.append({
                "source": record["source"],
                "relationship": record["relationship"],
                "target": record["target"]
            })
        return edges
    except:
        return []


def create_pyvis_graph(edges):
    """Create interactive PyVis network graph."""
    net = Network(
        height="600px",
        width="100%",
        bgcolor="#0f0f23",
        font_color="white",
        directed=True
    )
    
    # Color mapping for different entity types
    colors = {
        "company": "#00d4ff",
        "location": "#f472b6",
        "product": "#7c3aed",
        "risk": "#ff6b6b",
        "default": "#4ade80"
    }
    
    nodes_added = set()
    
    for edge in edges:
        source = edge["source"] or "Unknown"
        target = edge["target"] or "Unknown"
        rel = edge["relationship"] or "RELATED_TO"
        
        # Determine node colors based on content
        def get_color(name):
            name_lower = name.lower() if name else ""
            if any(x in name_lower for x in ["apple", "nvidia", "tsmc", "samsung", "foxconn", "amd", "qualcomm"]):
                return colors["company"]
            elif any(x in name_lower for x in ["taiwan", "china", "japan", "arizona", "usa"]):
                return colors["location"]
            elif any(x in name_lower for x in ["chip", "gpu", "processor", "3nm", "5nm"]):
                return colors["product"]
            elif any(x in name_lower for x in ["risk", "tension", "conflict", "constraint"]):
                return colors["risk"]
            return colors["default"]
        
        if source not in nodes_added:
            net.add_node(source, label=source[:30], color=get_color(source), 
                        size=25, font={"size": 12})
            nodes_added.add(source)
        
        if target not in nodes_added:
            net.add_node(target, label=target[:30], color=get_color(target),
                        size=25, font={"size": 12})
            nodes_added.add(target)
        
        net.add_edge(source, target, title=rel, label=rel[:20], 
                    color="#7c3aed", arrows="to")
    
    # Physics settings for better layout
    net.set_options("""
    {
        "nodes": {
            "borderWidth": 2,
            "borderWidthSelected": 4,
            "shadow": true
        },
        "edges": {
            "color": {"inherit": false},
            "smooth": {"type": "curvedCW", "roundness": 0.2},
            "shadow": true
        },
        "physics": {
            "forceAtlas2Based": {
                "gravitationalConstant": -50,
                "centralGravity": 0.01,
                "springLength": 200,
                "springConstant": 0.08
            },
            "solver": "forceAtlas2Based",
            "stabilization": {"iterations": 100}
        },
        "interaction": {
            "hover": true,
            "navigationButtons": true,
            "keyboard": true
        }
    }
    """)
    
    return net


def main():
    # Header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("# 🕵️ Shadow Supply Chain Hunter")
        st.markdown("### *Uncover Hidden Dependencies in SEC 10-K Filings*")
    with col2:
        st.image("https://img.icons8.com/nolan/128/graph.png", width=100)
    
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        st.markdown("---")
        
        # Status indicators
        llm, embed_model = init_llm()
        graph_store = connect_neo4j()
        
        st.markdown("### 🔌 Connection Status")
        
        if llm:
            st.success("✅ Groq LLM Connected")
        else:
            st.error("❌ Groq API Key Missing")
        
        if graph_store:
            st.success("✅ Neo4j Connected")
        else:
            st.error("❌ Neo4j Not Available")
        
        st.markdown("---")
        
        st.markdown("### 📊 Legend")
        st.markdown("""
        <div style='padding: 10px;'>
            <span style='color: #00d4ff;'>●</span> Companies<br>
            <span style='color: #f472b6;'>●</span> Locations<br>
            <span style='color: #7c3aed;'>●</span> Products<br>
            <span style='color: #ff6b6b;'>●</span> Risks<br>
            <span style='color: #4ade80;'>●</span> Other
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 🏢 Tracked Companies")
        st.markdown("- 🍎 Apple (AAPL)")
        st.markdown("- 🎮 NVIDIA (NVDA)")
        st.markdown("- 🔧 TSMC (TSM)")
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Dashboard", 
        "🔍 Query Engine", 
        "🕸️ Graph Explorer",
        "📄 Build Graph"
    ])
    
    # Tab 1: Dashboard
    with tab1:
        st.markdown("## 📊 Supply Chain Overview")
        
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="🏢 Companies Tracked",
                value="3",
                delta="AAPL, NVDA, TSM"
            )
        
        with col2:
            st.metric(
                label="🔗 Relationships",
                value="~25+",
                delta="Supply chain links"
            )
        
        with col3:
            st.metric(
                label="⚠️ Risk Factors",
                value="5",
                delta="Geopolitical, Natural"
            )
        
        with col4:
            st.metric(
                label="🌍 Regions",
                value="4",
                delta="Taiwan, China, US, Japan"
            )
        
        st.markdown("---")
        
        # Key insights
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🔴 Critical Dependencies")
            st.markdown("""
            <div class='risk-card'>
                <h4>🇹🇼 Taiwan Concentration Risk</h4>
                <p>Both <b>Apple</b> and <b>NVIDIA</b> depend critically on <b>TSMC</b> 
                for advanced chip manufacturing (3nm, 5nm). Over 90% of leading-edge 
                chips are made in Taiwan.</p>
                <br>
                <b>Impacted Companies:</b> AAPL, NVDA, AMD, QCOM
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("### 🟢 Key Suppliers")
            st.markdown("""
            <div class='supplier-card'>
                <h4>🏭 Primary Supply Chain</h4>
                <ul>
                    <li><b>TSMC</b> → Chip Manufacturing</li>
                    <li><b>Foxconn</b> → Device Assembly</li>
                    <li><b>Samsung</b> → Secondary Foundry</li>
                </ul>
                <br>
                <b>Diversification:</b> Arizona (USA), Kumamoto (Japan)
            </div>
            """, unsafe_allow_html=True)
        
        # Risk matrix
        st.markdown("### ⚠️ Risk Assessment Matrix")
        
        risk_data = pd.DataFrame({
            "Risk Type": ["Geopolitical (Taiwan)", "Natural Disasters", "Water Shortage", "Trade Restrictions", "Capacity Constraints"],
            "Severity": [5, 4, 3, 4, 4],
            "Likelihood": [3, 3, 2, 4, 5],
            "Impact Score": [15, 12, 6, 16, 20],
            "Affected": ["AAPL, NVDA, AMD", "TSMC, AAPL", "TSMC", "NVDA, AMD", "All"]
        })
        
        st.dataframe(
            risk_data,
            use_container_width=True,
            hide_index=True
        )
    
    # Tab 2: Query Engine
    with tab2:
        st.markdown("## 🔍 Natural Language Query")
        st.markdown("*Ask questions about supply chain relationships and risks*")
        
        if not llm or not graph_store:
            st.warning("⚠️ Please ensure Groq API and Neo4j are connected.")
        else:
            # Preset queries
            st.markdown("### 💡 Example Queries")
            preset_queries = [
                "What are the geopolitical risks for Apple?",
                "Which companies depend on TSMC?",
                "What risks does Taiwan pose to semiconductors?",
                "How is NVIDIA's supply chain vulnerable?",
                "What natural disasters could affect chip production?"
            ]
            
            col1, col2 = st.columns(2)
            for i, q in enumerate(preset_queries):
                with col1 if i % 2 == 0 else col2:
                    if st.button(f"📌 {q}", key=f"preset_{i}"):
                        st.session_state.query = q
            
            st.markdown("---")
            
            # Custom query
            query = st.text_input(
                "🔎 Enter your question:",
                value=st.session_state.get("query", ""),
                placeholder="e.g., What companies manufacture chips for Apple?"
            )
            
            if st.button("🚀 Search", type="primary"):
                if query:
                    with st.spinner("🔍 Searching knowledge graph..."):
                        try:
                            # Set up LlamaIndex
                            Settings.llm = llm
                            Settings.embed_model = embed_model
                            
                            storage_context = StorageContext.from_defaults(
                                property_graph_store=graph_store
                            )
                            
                            # Load existing index
                            index = PropertyGraphIndex.from_existing(
                                property_graph_store=graph_store,
                                llm=llm,
                                embed_model=embed_model,
                            )
                            
                            query_engine = index.as_query_engine(
                                include_text=True,
                                similarity_top_k=5,
                            )
                            
                            response = query_engine.query(query)
                            
                            st.markdown("### 📋 Answer")
                            st.success(response.response)
                            
                            # Show source nodes if available
                            if hasattr(response, 'source_nodes') and response.source_nodes:
                                with st.expander("📚 Source Context"):
                                    for node in response.source_nodes[:3]:
                                        st.markdown(f"```\n{node.text[:500]}...\n```")
                        
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
                            st.info("💡 Try building the graph first in the 'Build Graph' tab.")
    
    # Tab 3: Graph Explorer
    with tab3:
        st.markdown("## 🕸️ Interactive Knowledge Graph")
        
        if graph_store:
            edges = get_graph_data(graph_store)
            
            if edges:
                st.markdown(f"*Showing {len(edges)} relationships*")
                
                # Create and display graph
                net = create_pyvis_graph(edges)
                
                # Save to temp file and display
                with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as f:
                    net.save_graph(f.name)
                    with open(f.name, 'r') as html_file:
                        html_content = html_file.read()
                    components.html(html_content, height=650, scrolling=True)
                
                # Graph statistics
                st.markdown("---")
                st.markdown("### 📊 Graph Statistics")
                
                unique_nodes = set()
                for e in edges:
                    if e["source"]:
                        unique_nodes.add(e["source"])
                    if e["target"]:
                        unique_nodes.add(e["target"])
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Nodes", len(unique_nodes))
                with col2:
                    st.metric("Total Edges", len(edges))
                with col3:
                    rel_types = set(e["relationship"] for e in edges if e["relationship"])
                    st.metric("Relationship Types", len(rel_types))
                
            else:
                st.info("📭 No graph data found. Build the graph first!")
                st.markdown("""
                **To build the graph:**
                1. Go to the **Build Graph** tab
                2. Click **Build Knowledge Graph**
                3. Return here to explore
                """)
        else:
            st.error("❌ Neo4j not connected")
    
    # Tab 4: Build Graph
    with tab4:
        st.markdown("## 🔧 Build Knowledge Graph")
        st.markdown("*Extract supply chain entities and relationships from documents*")
        
        if not llm or not graph_store:
            st.warning("⚠️ Please ensure Groq API and Neo4j are connected.")
        else:
            st.markdown("### 📄 Source Documents")
            
            docs = get_sample_documents()
            for doc in docs:
                with st.expander(f"📑 {doc.metadata['company']} ({doc.metadata['ticker']})"):
                    st.markdown(doc.text)
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                max_paths = st.slider("Max paths per chunk", 5, 20, 10)
            
            with col2:
                chunk_size = st.slider("Chunk size", 256, 1024, 512)
            
            if st.button("🚀 Build Knowledge Graph", type="primary"):
                progress = st.progress(0)
                status = st.empty()
                
                try:
                    status.text("⚙️ Configuring LLM...")
                    progress.progress(10)
                    
                    Settings.llm = llm
                    Settings.embed_model = embed_model
                    Settings.chunk_size = chunk_size
                    Settings.chunk_overlap = 50
                    
                    status.text("🔗 Setting up storage...")
                    progress.progress(20)
                    
                    storage_context = StorageContext.from_defaults(
                        property_graph_store=graph_store
                    )
                    
                    status.text("🧠 Creating entity extractor...")
                    progress.progress(30)
                    
                    kg_extractor = SimpleLLMPathExtractor(
                        llm=llm,
                        max_paths_per_chunk=max_paths,
                        num_workers=1,
                    )
                    
                    status.text("📊 Building knowledge graph (this may take a minute)...")
                    progress.progress(50)
                    
                    index = PropertyGraphIndex.from_documents(
                        docs,
                        kg_extractors=[kg_extractor],
                        storage_context=storage_context,
                        show_progress=True,
                        embed_kg_nodes=True,
                    )
                    
                    progress.progress(100)
                    status.empty()
                    
                    st.balloons()
                    st.success("✅ Knowledge graph built successfully!")
                    st.info("🕸️ Go to the **Graph Explorer** tab to visualize your graph!")
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.exception(e)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🕵️ Shadow Supply Chain Hunter | Built with LlamaIndex + Neo4j + Groq</p>
        <p>Analyze SEC 10-K filings for hidden supply chain dependencies</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
