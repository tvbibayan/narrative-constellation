"""
================================================================================
SHADOW SUPPLY CHAIN HUNTER - INSTITUTIONAL TERMINAL
================================================================================
Tier-1 GraphRAG Dashboard for SEC 10-K Supply Chain Intelligence
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
from datetime import datetime
import time

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

# ============================================================================
# PROFESSIONAL SVG ICONS (Lucide-style)
# ============================================================================
ICONS = {
    "search": '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"></circle><path d="m21 21-4.3-4.3"></path></svg>',
    "graph": '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="18" cy="18" r="3"></circle><circle cx="6" cy="6" r="3"></circle><circle cx="6" cy="18" r="3"></circle><line x1="6" y1="9" x2="6" y2="15"></line><path d="M18 15V9a6 6 0 0 0-6-6H6"></path></svg>',
    "dashboard": '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="7" height="9" x="3" y="3" rx="1"></rect><rect width="7" height="5" x="14" y="3" rx="1"></rect><rect width="7" height="9" x="14" y="12" rx="1"></rect><rect width="7" height="5" x="3" y="16" rx="1"></rect></svg>',
    "settings": '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z"></path><circle cx="12" cy="12" r="3"></circle></svg>',
    "alert": '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"></path><path d="M12 9v4"></path><path d="M12 17h.01"></path></svg>',
    "building": '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="16" height="20" x="4" y="2" rx="2" ry="2"></rect><path d="M9 22v-4h6v4"></path><path d="M8 6h.01"></path><path d="M16 6h.01"></path><path d="M12 6h.01"></path><path d="M12 10h.01"></path><path d="M12 14h.01"></path><path d="M16 10h.01"></path><path d="M16 14h.01"></path><path d="M8 10h.01"></path><path d="M8 14h.01"></path></svg>',
    "link": '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"></path><path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"></path></svg>',
    "globe": '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><path d="M12 2a14.5 14.5 0 0 0 0 20 14.5 14.5 0 0 0 0-20"></path><path d="M2 12h20"></path></svg>',
    "shield": '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10"></path></svg>',
    "cpu": '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="16" height="16" x="4" y="4" rx="2"></rect><rect width="6" height="6" x="9" y="9" rx="1"></rect><path d="M15 2v2"></path><path d="M15 20v2"></path><path d="M2 15h2"></path><path d="M2 9h2"></path><path d="M20 15h2"></path><path d="M20 9h2"></path><path d="M9 2v2"></path><path d="M9 20v2"></path></svg>',
    "factory": '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M2 20a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V8l-7 5V8l-7 5V4a2 2 0 0 0-2-2H4a2 2 0 0 0-2 2Z"></path><path d="M17 18h1"></path><path d="M12 18h1"></path><path d="M7 18h1"></path></svg>',
    "zap": '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon></svg>',
    "target": '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><circle cx="12" cy="12" r="6"></circle><circle cx="12" cy="12" r="2"></circle></svg>',
    "activity": '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 12h-4l-3 9L9 3l-3 9H2"></path></svg>',
    "database": '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><ellipse cx="12" cy="5" rx="9" ry="3"></ellipse><path d="M3 5V19a9 3 0 0 0 18 0V5"></path><path d="M3 12a9 3 0 0 0 18 0"></path></svg>',
    "file": '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z"></path><path d="M14 2v4a2 2 0 0 0 2 2h4"></path></svg>',
    "play": '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="6 3 20 12 6 21 6 3"></polygon></svg>',
    "radio": '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4.9 19.1C1 15.2 1 8.8 4.9 4.9"></path><path d="M7.8 16.2c-2.3-2.3-2.3-6.1 0-8.5"></path><circle cx="12" cy="12" r="2"></circle><path d="M16.2 7.8c2.3 2.3 2.3 6.1 0 8.5"></path><path d="M19.1 4.9C23 8.8 23 15.1 19.1 19"></path></svg>',
    "map": '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="3 6 9 3 15 6 21 3 21 18 15 21 9 18 3 21"></polygon><line x1="9" x2="9" y1="3" y2="18"></line><line x1="15" x2="15" y1="6" y2="21"></line></svg>',
    "check": '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6 9 17l-5-5"></path></svg>',
    "x": '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M18 6 6 18"></path><path d="m6 6 12 12"></path></svg>',
    "layers": '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="12 2 2 7 12 12 22 7 12 2"></polygon><polyline points="2 17 12 22 22 17"></polyline><polyline points="2 12 12 17 22 12"></polyline></svg>',
    "network": '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="16" y="16" width="6" height="6" rx="1"></rect><rect x="2" y="16" width="6" height="6" rx="1"></rect><rect x="9" y="2" width="6" height="6" rx="1"></rect><path d="M5 16v-3a1 1 0 0 1 1-1h12a1 1 0 0 1 1 1v3"></path><path d="M12 12V8"></path></svg>',
    "pin": '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="12" x2="12" y1="17" y2="22"></line><path d="M5 17h14v-1.76a2 2 0 0 0-1.11-1.79l-1.78-.9A2 2 0 0 1 15 10.76V6h1a2 2 0 0 0 0-4H8a2 2 0 0 0 0 4h1v4.76a2 2 0 0 1-1.11 1.79l-1.78.9A2 2 0 0 0 5 15.24Z"></path></svg>',
}

def icon(name, size=16, color="currentColor"):
    """Return an SVG icon with custom size and color."""
    svg = ICONS.get(name, ICONS["target"])
    svg = svg.replace('width="16"', f'width="{size}"')
    svg = svg.replace('height="16"', f'height="{size}"')
    svg = svg.replace('stroke="currentColor"', f'stroke="{color}"')
    return svg

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="Supply Chain Terminal",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# TIER-1 INSTITUTIONAL TERMINAL CSS
# ============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');
    
    :root {
        --bg-primary: #0E1117;
        --bg-secondary: #161B22;
        --bg-tertiary: #1C2128;
        --border-color: #30363D;
        --accent-green: #00FF9D;
        --accent-red: #FF4B4B;
        --accent-blue: #58A6FF;
        --accent-purple: #A371F7;
        --accent-cyan: #00D4FF;
        --text-primary: #E6EDF3;
        --text-secondary: #8B949E;
        --text-muted: #484F58;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    
    .stApp {
        background: var(--bg-primary);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 100%;
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0D1117 0%, #161B22 100%);
        border-right: 1px solid var(--border-color);
    }
    
    .glass-card {
        background: rgba(22, 27, 34, 0.8);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(48, 54, 61, 0.8);
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    .glass-card-green {
        background: rgba(22, 27, 34, 0.8);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(0, 255, 157, 0.3);
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 0 20px rgba(0, 255, 157, 0.1);
    }
    
    .glass-card-red {
        background: rgba(22, 27, 34, 0.8);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 75, 75, 0.3);
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 0 20px rgba(255, 75, 75, 0.1);
    }
    
    .glass-card-blue {
        background: rgba(22, 27, 34, 0.8);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(88, 166, 255, 0.3);
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 0 20px rgba(88, 166, 255, 0.1);
    }
    
    [data-testid="metric-container"] {
        background: rgba(22, 27, 34, 0.9);
        backdrop-filter: blur(10px);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    
    [data-testid="metric-container"] label {
        font-family: 'Inter', sans-serif !important;
        font-weight: 500 !important;
        color: var(--text-secondary) !important;
        font-size: 0.85rem !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace !important;
        font-weight: 600 !important;
        color: var(--text-primary) !important;
        font-size: 1.8rem !important;
    }
    
    h1, h2, h3 {
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        color: var(--text-primary) !important;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        background: var(--bg-secondary);
        border-radius: 10px;
        padding: 4px;
        gap: 4px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        color: var(--text-secondary);
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        padding: 10px 20px;
    }
    
    .stTabs [aria-selected="true"] {
        background: var(--bg-tertiary) !important;
        color: var(--accent-cyan) !important;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, var(--accent-cyan) 0%, var(--accent-purple) 100%);
        color: var(--bg-primary);
        border: none;
        border-radius: 8px;
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        padding: 12px 28px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 212, 255, 0.4);
    }
    
    .stTextInput > div > div > input {
        background: var(--bg-secondary) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 8px !important;
        color: var(--text-primary) !important;
        font-family: 'Inter', sans-serif !important;
        padding: 12px 16px !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: var(--accent-cyan) !important;
        box-shadow: 0 0 0 2px rgba(0, 212, 255, 0.2) !important;
    }
    
    .stAlert {
        background: var(--bg-secondary) !important;
        border-radius: 8px !important;
        border-left: 4px solid var(--accent-cyan) !important;
    }
    
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, var(--accent-cyan), var(--accent-purple)) !important;
    }
    
    hr {
        border-color: var(--border-color) !important;
        opacity: 0.5;
    }
    
    .terminal-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 15px 0;
        border-bottom: 1px solid var(--border-color);
        margin-bottom: 20px;
    }
    
    .status-indicator {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        font-family: 'Inter', sans-serif;
        font-size: 0.85rem;
        color: var(--text-secondary);
    }
    
    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        animation: pulse 2s infinite;
    }
    
    .status-dot.online { background: var(--accent-green); box-shadow: 0 0 10px var(--accent-green); }
    .status-dot.offline { background: var(--accent-red); box-shadow: 0 0 10px var(--accent-red); }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    .ticker-card {
        background: rgba(22, 27, 34, 0.9);
        border: 1px solid var(--border-color);
        border-left: 3px solid var(--accent-cyan);
        border-radius: 8px;
        padding: 16px;
        margin: 8px 0;
        transition: all 0.2s ease;
    }
    
    .ticker-card:hover {
        border-left-color: var(--accent-green);
        transform: translateX(4px);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    }
    
    .ticker-card-title {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        color: var(--text-primary);
        font-size: 0.95rem;
        margin-bottom: 6px;
    }
    
    .ticker-card-meta {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        color: var(--text-muted);
    }
    
    .risk-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-family: 'Inter', sans-serif;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
    }
    
    .risk-high { background: rgba(255, 75, 75, 0.2); color: var(--accent-red); border: 1px solid rgba(255, 75, 75, 0.3); }
    .risk-medium { background: rgba(255, 193, 7, 0.2); color: #FFC107; border: 1px solid rgba(255, 193, 7, 0.3); }
    .risk-low { background: rgba(0, 255, 157, 0.2); color: var(--accent-green); border: 1px solid rgba(0, 255, 157, 0.3); }
    
    .icon-label {
        display: inline-flex;
        align-items: center;
        gap: 8px;
    }
    
    .section-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 15px;
        color: var(--text-primary);
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        font-size: 1.1rem;
    }
    
    .mono-number {
        font-family: 'JetBrains Mono', monospace !important;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# DATA FUNCTIONS (UNCHANGED)
# ============================================================================
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
    llm = Groq(model="llama-3.3-70b-versatile", api_key=groq_key, temperature=0.0)
    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return llm, embed_model


@st.cache_resource
def connect_neo4j():
    """Connect to Neo4j."""
    try:
        return Neo4jPropertyGraphStore(
            url=os.getenv("NEO4J_URI", "neo4j://localhost:7687"),
            username=os.getenv("NEO4J_USERNAME", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", "supplychain123"),
        )
    except:
        return None


def get_graph_data(graph_store):
    """Fetch graph data from Neo4j for visualization."""
    try:
        query = "MATCH (n)-[r]->(m) RETURN n.id as source, type(r) as relationship, m.id as target LIMIT 100"
        result = graph_store._driver.execute_query(query)
        return [{"source": r["source"], "relationship": r["relationship"], "target": r["target"]} for r in result.records]
    except:
        return []


def create_pyvis_graph(edges):
    """Create interactive PyVis network graph."""
    net = Network(height="600px", width="100%", bgcolor="#0E1117", font_color="#E6EDF3", directed=True)
    colors = {"company": "#00D4FF", "location": "#A371F7", "product": "#58A6FF", "risk": "#FF4B4B", "default": "#00FF9D"}
    nodes_added = set()
    
    for edge in edges:
        source, target, rel = edge["source"] or "Unknown", edge["target"] or "Unknown", edge["relationship"] or "RELATED_TO"
        
        def get_color(name):
            n = (name or "").lower()
            if any(x in n for x in ["apple", "nvidia", "tsmc", "samsung", "foxconn", "amd", "qualcomm"]): return colors["company"]
            elif any(x in n for x in ["taiwan", "china", "japan", "arizona", "usa"]): return colors["location"]
            elif any(x in n for x in ["chip", "gpu", "processor", "3nm", "5nm"]): return colors["product"]
            elif any(x in n for x in ["risk", "tension", "conflict", "constraint"]): return colors["risk"]
            return colors["default"]
        
        if source not in nodes_added:
            net.add_node(source, label=source[:25], color=get_color(source), size=30, font={"size": 14})
            nodes_added.add(source)
        if target not in nodes_added:
            net.add_node(target, label=target[:25], color=get_color(target), size=30, font={"size": 14})
            nodes_added.add(target)
        net.add_edge(source, target, title=rel, label=rel[:15], color="#30363D", arrows="to", width=2)
    
    net.set_options('{"physics": {"forceAtlas2Based": {"gravitationalConstant": -80}, "solver": "forceAtlas2Based"}, "interaction": {"hover": true}}')
    return net


# ============================================================================
# MAIN APPLICATION
# ============================================================================
def main():
    llm, embed_model = init_llm()
    graph_store = connect_neo4j()
    
    # ========================================================================
    # SIDEBAR
    # ========================================================================
    with st.sidebar:
        st.markdown(f"""
        <div style='text-align: center; padding: 20px 0; border-bottom: 1px solid #30363D; margin-bottom: 20px;'>
            <div style='margin-bottom: 10px;'>{icon('network', 32, '#00D4FF')}</div>
            <h2 style='margin: 0; font-size: 1.3rem; color: #E6EDF3;'>SUPPLY CHAIN</h2>
            <p style='margin: 5px 0 0 0; font-size: 0.7rem; color: #8B949E; letter-spacing: 2px;'>INTELLIGENCE TERMINAL</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"<div class='section-header'>{icon('activity', 18, '#8B949E')} SYSTEM STATUS</div>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            status = "online" if llm else "offline"
            st.markdown(f"<div class='status-indicator'><span class='status-dot {status}'></span><span>LLM</span></div>", unsafe_allow_html=True)
        with col2:
            status = "online" if graph_store else "offline"
            st.markdown(f"<div class='status-indicator'><span class='status-dot {status}'></span><span>NEO4J</span></div>", unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"<div class='section-header'>{icon('building', 18, '#8B949E')} TRACKED ASSETS</div>", unsafe_allow_html=True)
        
        for asset in [{"ticker": "AAPL", "name": "Apple Inc.", "risk": "HIGH"}, {"ticker": "NVDA", "name": "NVIDIA Corp.", "risk": "HIGH"}, {"ticker": "TSM", "name": "TSMC", "risk": "CRITICAL"}]:
            risk_class = "risk-high" if asset["risk"] in ["HIGH", "CRITICAL"] else "risk-medium"
            st.markdown(f"""
            <div class='ticker-card'>
                <div class='ticker-card-title'>{asset['ticker']}</div>
                <div style='display: flex; justify-content: space-between; align-items: center;'>
                    <span class='ticker-card-meta'>{asset['name']}</span>
                    <span class='risk-badge {risk_class}'>{asset['risk']}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"<div class='section-header'>{icon('layers', 18, '#8B949E')} NODE LEGEND</div>", unsafe_allow_html=True)
        st.markdown("""
        <div style='font-size: 0.85rem; color: #8B949E;'>
            <div style='margin: 8px 0;'><span style='color: #00D4FF;'>●</span> Companies</div>
            <div style='margin: 8px 0;'><span style='color: #A371F7;'>●</span> Locations</div>
            <div style='margin: 8px 0;'><span style='color: #58A6FF;'>●</span> Products</div>
            <div style='margin: 8px 0;'><span style='color: #FF4B4B;'>●</span> Risks</div>
            <div style='margin: 8px 0;'><span style='color: #00FF9D;'>●</span> Other</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div style='font-family: JetBrains Mono; font-size: 0.7rem; color: #484F58; text-align: center; margin-top: 30px;'>
            {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC
        </div>
        """, unsafe_allow_html=True)
    
    # ========================================================================
    # MAIN HEADER
    # ========================================================================
    st.markdown(f"""
    <div class='terminal-header'>
        <div style='display: flex; align-items: center; gap: 15px;'>
            {icon('target', 28, '#00D4FF')}
            <div>
                <h1 style='margin: 0; font-size: 1.6rem;'>SHADOW SUPPLY CHAIN HUNTER</h1>
                <p style='margin: 5px 0 0 0; color: #8B949E; font-size: 0.85rem;'>
                    Real-time GraphRAG Intelligence · SEC 10-K Analysis · Risk Monitoring
                </p>
            </div>
        </div>
        <div class='status-indicator' style='font-size: 0.9rem;'>
            <span class='status-dot online'></span>
            <span style='color: #00FF9D; font-weight: 500;'>SYSTEM ONLINE</span>
            <span style='color: #484F58; margin-left: 15px;'>|</span>
            <span style='color: #8B949E; margin-left: 15px; font-family: JetBrains Mono;'>{datetime.now().strftime('%H:%M:%S')}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # ========================================================================
    # KPI ROW
    # ========================================================================
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1: st.metric(label="ENTITIES TRACKED", value="3", delta="Active")
    with col2: st.metric(label="RELATIONSHIPS", value="25+", delta="Mapped")
    with col3: st.metric(label="RISK FACTORS", value="5", delta="Identified")
    with col4: st.metric(label="REGIONS", value="4", delta="Monitored")
    with col5: st.metric(label="ALERT LEVEL", value="HIGH", delta="Taiwan")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ========================================================================
    # TABS
    # ========================================================================
    tab1, tab2, tab3, tab4 = st.tabs(["DASHBOARD", "QUERY ENGINE", "GRAPH EXPLORER", "BUILD GRAPH"])
    
    # TAB 1: DASHBOARD
    with tab1:
        col_left, col_right = st.columns([1.2, 1])
        
        with col_left:
            st.markdown(f"<div class='section-header'>{icon('alert', 20, '#FF4B4B')} CRITICAL DEPENDENCIES</div>", unsafe_allow_html=True)
            st.markdown(f"""
            <div class='glass-card-red'>
                <div style='display: flex; justify-content: space-between; align-items: start;'>
                    <div>
                        <h4 style='margin: 0 0 10px 0; color: #FF4B4B; display: flex; align-items: center; gap: 8px;'>
                            {icon('globe', 18, '#FF4B4B')} TAIWAN CONCENTRATION RISK
                        </h4>
                        <p style='color: #8B949E; font-size: 0.9rem; line-height: 1.6;'>
                            Both <span style='color: #00D4FF; font-weight: 600;'>Apple</span> and 
                            <span style='color: #00D4FF; font-weight: 600;'>NVIDIA</span> depend critically on 
                            <span style='color: #00FF9D; font-weight: 600;'>TSMC</span> for advanced chip manufacturing. 
                            Over <span class='mono-number' style='color: #FF4B4B; font-weight: 700;'>90%</span> 
                            of leading-edge chips are produced in Taiwan.
                        </p>
                    </div>
                    <span class='risk-badge risk-high'>CRITICAL</span>
                </div>
                <div style='margin-top: 15px; padding-top: 15px; border-top: 1px solid #30363D;'>
                    <span style='color: #8B949E; font-size: 0.8rem;'>IMPACTED:</span>
                    <span style='font-family: JetBrains Mono; color: #E6EDF3; margin-left: 10px;'>AAPL · NVDA · AMD · QCOM</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"<div class='section-header'>{icon('factory', 20, '#00FF9D')} PRIMARY SUPPLY CHAIN</div>", unsafe_allow_html=True)
            st.markdown(f"""
            <div class='glass-card-green'>
                <h4 style='margin: 0 0 15px 0; color: #00FF9D; display: flex; align-items: center; gap: 8px;'>
                    {icon('cpu', 18, '#00FF9D')} MANUFACTURING PARTNERS
                </h4>
                <div style='display: grid; gap: 12px;'>
                    <div style='display: flex; justify-content: space-between; padding: 10px; background: rgba(0,0,0,0.2); border-radius: 6px;'>
                        <span style='color: #E6EDF3; font-weight: 500;'>TSMC</span>
                        <span style='color: #8B949E;'>Chip Manufacturing · 3nm/5nm</span>
                    </div>
                    <div style='display: flex; justify-content: space-between; padding: 10px; background: rgba(0,0,0,0.2); border-radius: 6px;'>
                        <span style='color: #E6EDF3; font-weight: 500;'>Foxconn</span>
                        <span style='color: #8B949E;'>Device Assembly · China</span>
                    </div>
                    <div style='display: flex; justify-content: space-between; padding: 10px; background: rgba(0,0,0,0.2); border-radius: 6px;'>
                        <span style='color: #E6EDF3; font-weight: 500;'>Samsung</span>
                        <span style='color: #8B949E;'>Secondary Foundry · Korea</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_right:
            st.markdown(f"<div class='section-header'>{icon('shield', 20, '#FFC107')} RISK MATRIX</div>", unsafe_allow_html=True)
            risk_data = pd.DataFrame({
                "RISK TYPE": ["Geopolitical", "Natural Disaster", "Water Shortage", "Trade War", "Capacity"],
                "SEV": [5, 4, 3, 4, 4], "PROB": [3, 3, 2, 4, 5], "SCORE": [15, 12, 6, 16, 20],
                "AFFECTED": ["AAPL,NVDA", "TSMC", "TSMC", "NVDA", "ALL"]
            })
            st.dataframe(risk_data, use_container_width=True, hide_index=True,
                column_config={"SCORE": st.column_config.ProgressColumn("SCORE", min_value=0, max_value=25, format="%d")})
            
            st.markdown(f"<div class='section-header'>{icon('radio', 20, '#58A6FF')} RECENT ALERTS</div>", unsafe_allow_html=True)
            for alert in [
                {"time": "14:32:01", "type": "GEOPOLITICAL", "msg": "US-China tensions escalate over Taiwan strait", "color": "#FF4B4B"},
                {"time": "13:45:22", "type": "SUPPLY", "msg": "TSMC reports 3nm capacity constraints", "color": "#FFC107"},
                {"time": "11:20:45", "type": "WEATHER", "msg": "Typhoon warning issued for Taiwan region", "color": "#58A6FF"},
            ]:
                st.markdown(f"""
                <div class='ticker-card' style='border-left-color: {alert['color']};'>
                    <span class='ticker-card-title'>{alert['msg']}</span>
                    <div class='ticker-card-meta' style='margin-top: 8px;'>
                        <span style='color: {alert['color']};'>{alert['type']}</span> · {alert['time']} UTC
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    # TAB 2: QUERY ENGINE
    with tab2:
        st.markdown(f"<div class='section-header'>{icon('search', 20, '#00D4FF')} NATURAL LANGUAGE QUERY</div>", unsafe_allow_html=True)
        
        if not llm or not graph_store:
            st.warning("LLM or Neo4j not connected. Check sidebar for status.")
        else:
            st.markdown(f"<div class='section-header' style='font-size: 0.9rem;'>{icon('zap', 16, '#8B949E')} QUICK QUERIES</div>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            queries = ["What are geopolitical risks for Apple?", "Which companies depend on TSMC?", "What natural disasters threaten supply?",
                       "How is NVIDIA's supply chain vulnerable?", "What are Taiwan concentration risks?", "List all critical suppliers"]
            for i, q in enumerate(queries):
                with [col1, col2, col3][i % 3]:
                    if st.button(f"{q[:35]}...", key=f"q_{i}", use_container_width=True):
                        st.session_state.query = q
            
            st.markdown("---")
            query = st.text_input("ENTER QUERY", value=st.session_state.get("query", ""), placeholder="e.g., What companies manufacture chips for Apple?", label_visibility="collapsed")
            
            col1, col2, col3 = st.columns([1, 1, 3])
            with col1:
                if st.button("EXECUTE QUERY", type="primary", use_container_width=True) and query:
                    with st.spinner("Querying..."):
                        try:
                            Settings.llm, Settings.embed_model = llm, embed_model
                            index = PropertyGraphIndex.from_existing(property_graph_store=graph_store, llm=llm, embed_model=embed_model)
                            response = index.as_query_engine(include_text=True, similarity_top_k=5).query(query)
                            st.markdown(f"<div class='section-header'>{icon('file', 18, '#00FF9D')} RESPONSE</div>", unsafe_allow_html=True)
                            st.markdown(f"<div class='glass-card-blue'><p style='color: #E6EDF3; line-height: 1.7;'>{response.response}</p></div>", unsafe_allow_html=True)
                        except Exception as e:
                            st.error(f"Query failed: {str(e)}")
    
    # TAB 3: GRAPH EXPLORER
    with tab3:
        st.markdown(f"<div class='section-header'>{icon('graph', 20, '#A371F7')} KNOWLEDGE GRAPH EXPLORER</div>", unsafe_allow_html=True)
        
        if graph_store:
            edges = get_graph_data(graph_store)
            if edges:
                unique_nodes = set(e["source"] for e in edges if e["source"]) | set(e["target"] for e in edges if e["target"])
                col1, col2, col3, col4 = st.columns(4)
                with col1: st.metric("NODES", len(unique_nodes))
                with col2: st.metric("EDGES", len(edges))
                with col3: st.metric("REL TYPES", len(set(e["relationship"] for e in edges if e["relationship"])))
                with col4: st.metric("DENSITY", f"{len(edges)/max(len(unique_nodes),1):.1f}")
                
                net = create_pyvis_graph(edges)
                with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as f:
                    net.save_graph(f.name)
                    components.html(open(f.name).read(), height=650, scrolling=True)
            else:
                st.markdown("<div class='glass-card' style='text-align: center; padding: 60px;'><h3 style='color: #8B949E;'>NO GRAPH DATA</h3><p style='color: #484F58;'>Build the knowledge graph first.</p></div>", unsafe_allow_html=True)
        else:
            st.error("Neo4j not connected")
    
    # TAB 4: BUILD GRAPH
    with tab4:
        st.markdown(f"<div class='section-header'>{icon('settings', 20, '#A371F7')} BUILD KNOWLEDGE GRAPH</div>", unsafe_allow_html=True)
        
        if not llm or not graph_store:
            st.warning("LLM or Neo4j not connected.")
        else:
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown(f"<div class='section-header' style='font-size: 0.9rem;'>{icon('file', 16, '#8B949E')} SOURCE DOCUMENTS</div>", unsafe_allow_html=True)
                for doc in get_sample_documents():
                    with st.expander(f"{doc.metadata['company']} ({doc.metadata['ticker']})"):
                        st.markdown(f"<p style='color: #8B949E; font-size: 0.9rem;'>{doc.text}</p>", unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"<div class='section-header' style='font-size: 0.9rem;'>{icon('settings', 16, '#8B949E')} PARAMETERS</div>", unsafe_allow_html=True)
                max_paths = st.slider("Max Paths/Chunk", 5, 20, 10)
                chunk_size = st.slider("Chunk Size", 256, 1024, 512)
                
                if st.button("BUILD GRAPH", type="primary", use_container_width=True):
                    progress = st.progress(0)
                    try:
                        Settings.llm, Settings.embed_model, Settings.chunk_size, Settings.chunk_overlap = llm, embed_model, chunk_size, 50
                        progress.progress(20)
                        storage_context = StorageContext.from_defaults(property_graph_store=graph_store)
                        progress.progress(40)
                        kg_extractor = SimpleLLMPathExtractor(llm=llm, max_paths_per_chunk=max_paths, num_workers=1)
                        progress.progress(60)
                        PropertyGraphIndex.from_documents(get_sample_documents(), kg_extractors=[kg_extractor], storage_context=storage_context, show_progress=True, embed_kg_nodes=True)
                        progress.progress(100)
                        st.balloons()
                        st.success("Knowledge graph built successfully!")
                    except Exception as e:
                        st.error(f"Build failed: {str(e)}")
    
    # FOOTER
    st.markdown("---")
    st.markdown(f"""
    <div style='text-align: center; padding: 20px 0;'>
        <p style='color: #484F58; font-size: 0.8rem; font-family: JetBrains Mono;'>
            SHADOW SUPPLY CHAIN HUNTER v1.0 · GROQ + LLAMAINDEX + NEO4J · {datetime.now().strftime('%Y-%m-%d')}
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
