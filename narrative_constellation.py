"""
╔══════════════════════════════════════════════════════════════════════════════╗
║            NARRATIVE CONSTELLATION TERMINAL v4.0                             ║
║         Graph-Theoretic Financial Narrative Analysis + Alpha Signals         ║
╚══════════════════════════════════════════════════════════════════════════════╝

A sophisticated narrative analysis system that visualizes headline relationships
using graph theory, revealing hidden narrative clusters and contagion patterns.

NOVEL ALPHA SIGNALS (Not available in existing tools):
    1. NFI (Narrative Fragmentation Index) - Measures narrative coherence
       Formula: NFI = 1 - (largest_cluster_size / total_nodes)
       High NFI = fragmented narratives = volatility ahead
       
    2. SPDS (Sentiment-Price Dislocation Score) - Price vs sentiment divergence
       Formula: SPDS = Rolling_Correlation(Sentiment, Returns, 5d)
       Negative SPDS = dislocation = mean reversion opportunity

Core Innovation:
    - Headlines as Nodes in a semantic network
    - Keyword overlap creates Edges (narrative connections)
    - Sentiment colors the constellation (Green=Greed, Red=Fear)
    - Viral Potential sizes the nodes (bigger = more contagious)

Architecture:
    ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
    │  RSS Ingestion  │ --> │  SQLite + NLP   │ --> │  NetworkX Graph │
    │  (Google News)  │     │  (VADER + KW)   │     │  (Constellation)│
    └─────────────────┘     └─────────────────┘     └─────────────────┘
                                    │
                                    v
                          ┌─────────────────┐
                          │  Alpha Signals  │
                          │  NFI + SPDS     │
                          └─────────────────┘

Author: Financial Engineering Team
Version: 4.0.0 Alpha Signals Edition
"""

import streamlit as st
import yfinance as yf
import feedparser
import sqlite3
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import networkx as nx
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple, Set
import hashlib
import re
from contextlib import contextmanager
from dataclasses import dataclass
import time

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

class Config:
    """Centralized configuration."""
    DB_PATH = "narrative_constellation.db"
    RSS_BASE_URL = "https://news.google.com/rss/search?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en"
    DEFAULT_TICKERS = ["AAPL", "NVDA", "TSLA", "MSFT", "GOOGL", "AMZN", "META", "AMD", "TSM", "INTC"]
    LOOKBACK_DAYS = 30
    MIN_KEYWORD_LENGTH = 3
    MIN_SHARED_KEYWORDS = 1  # Minimum shared keywords to create an edge

# Stop words for keyword extraction
STOP_WORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
    'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been', 'be', 'have', 'has', 'had',
    'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must',
    'shall', 'can', 'need', 'dare', 'ought', 'used', 'it', 'its', 'this', 'that',
    'these', 'those', 'i', 'you', 'he', 'she', 'we', 'they', 'what', 'which', 'who',
    'whom', 'whose', 'where', 'when', 'why', 'how', 'all', 'each', 'every', 'both',
    'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
    'same', 'so', 'than', 'too', 'very', 'just', 'also', 'now', 'here', 'there', 'then',
    'once', 'new', 'after', 'says', 'said', 'stock', 'stocks', 'shares', 'share',
    'market', 'markets', 'price', 'prices', 'today', 'report', 'reports', 'news',
    'company', 'companies', 'inc', 'corp', 'ltd', 'llc', 'amid', 'over', 'into', 'about'
}

# ══════════════════════════════════════════════════════════════════════════════
# SVG ICON SYSTEM (No Emojis - Enterprise Terminal Aesthetic)
# ══════════════════════════════════════════════════════════════════════════════

class Icons:
    """Professional SVG icon library."""
    
    @staticmethod
    def arrow_up_right(color: str = "#00FF88", size: int = 16) -> str:
        return f'''<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><line x1="7" y1="17" x2="17" y2="7"></line><polyline points="7 7 17 7 17 17"></polyline></svg>'''
    
    @staticmethod
    def arrow_down_right(color: str = "#FF3366", size: int = 16) -> str:
        return f'''<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><line x1="7" y1="7" x2="17" y2="17"></line><polyline points="17 7 17 17 7 17"></polyline></svg>'''
    
    @staticmethod
    def minus_circle(color: str = "#64748B", size: int = 16) -> str:
        return f'''<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="8" y1="12" x2="16" y2="12"></line></svg>'''
    
    @staticmethod
    def network(color: str = "#3B82F6", size: int = 16) -> str:
        return f'''<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="5" r="3"></circle><circle cx="5" cy="19" r="3"></circle><circle cx="19" cy="19" r="3"></circle><line x1="12" y1="8" x2="5" y2="16"></line><line x1="12" y1="8" x2="19" y2="16"></line></svg>'''
    
    @staticmethod
    def search(color: str = "#94A3B8", size: int = 16) -> str:
        return f'''<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg>'''
    
    @staticmethod
    def database(color: str = "#8B5CF6", size: int = 16) -> str:
        return f'''<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><ellipse cx="12" cy="5" rx="9" ry="3"></ellipse><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"></path><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"></path></svg>'''
    
    @staticmethod
    def zap(color: str = "#FFB800", size: int = 16) -> str:
        return f'''<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon></svg>'''
    
    @staticmethod
    def refresh(color: str = "#00D9FF", size: int = 16) -> str:
        return f'''<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 4 23 10 17 10"></polyline><polyline points="1 20 1 14 7 14"></polyline><path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"></path></svg>'''
    
    @staticmethod
    def candlestick(color: str = "#00FF88", size: int = 16) -> str:
        return f'''<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="8" width="4" height="8" rx="1"></rect><line x1="5" y1="4" x2="5" y2="8"></line><line x1="5" y1="16" x2="5" y2="20"></line><rect x="10" y="6" width="4" height="10" rx="1"></rect><line x1="12" y1="2" x2="12" y2="6"></line><line x1="12" y1="16" x2="12" y2="22"></line><rect x="17" y="10" width="4" height="6" rx="1"></rect><line x1="19" y1="6" x2="19" y2="10"></line><line x1="19" y1="16" x2="19" y2="18"></line></svg>'''
    
    @staticmethod
    def constellation(color: str = "#00D9FF", size: int = 24) -> str:
        return f'''<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="6" cy="6" r="2" fill="{color}"></circle><circle cx="18" cy="8" r="2" fill="{color}"></circle><circle cx="12" cy="12" r="3" fill="{color}"></circle><circle cx="4" cy="18" r="2" fill="{color}"></circle><circle cx="20" cy="18" r="2" fill="{color}"></circle><line x1="8" y1="6" x2="10" y2="10" opacity="0.5"></line><line x1="16" y1="8" x2="14" y2="10" opacity="0.5"></line><line x1="10" y1="14" x2="6" y2="16" opacity="0.5"></line><line x1="14" y1="14" x2="18" y2="16" opacity="0.5"></line></svg>'''
    
    @staticmethod
    def insight(color: str = "#FFB800", size: int = 16) -> str:
        return f'''<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2v4"></path><path d="M12 18v4"></path><path d="M4.93 4.93l2.83 2.83"></path><path d="M16.24 16.24l2.83 2.83"></path><path d="M2 12h4"></path><path d="M18 12h4"></path><path d="M4.93 19.07l2.83-2.83"></path><path d="M16.24 7.76l2.83-2.83"></path></svg>'''
    
    @staticmethod
    def get_sentiment_icon(sentiment: float, size: int = 18) -> str:
        """Get appropriate icon based on sentiment."""
        if sentiment > 0.15:
            return Icons.arrow_up_right("#00FF88", size)
        elif sentiment < -0.15:
            return Icons.arrow_down_right("#FF3366", size)
        else:
            return Icons.minus_circle("#64748B", size)
    
    @staticmethod
    def radar(color: str = "#8B5CF6", size: int = 16) -> str:
        return f'''<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><polygon points="12 2 15.09 9 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 9" fill="{color}" opacity="0.3"></polygon><line x1="12" y1="12" x2="12" y2="2"></line></svg>'''
    
    @staticmethod
    def divergence(color: str = "#F59E0B", size: int = 16) -> str:
        return f'''<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M16 3h5v5"></path><path d="M8 21H3v-5"></path><path d="M21 3l-9 9"></path><path d="M3 21l9-9"></path></svg>'''
    
    @staticmethod
    def pulse(color: str = "#10B981", size: int = 16) -> str:
        return f'''<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline></svg>'''
    
    @staticmethod
    def shield_alert(color: str = "#EF4444", size: int = 16) -> str:
        return f'''<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path><path d="M12 8v4"></path><path d="M12 16h.01"></path></svg>'''
    
    @staticmethod
    def target(color: str = "#06B6D4", size: int = 16) -> str:
        return f'''<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><circle cx="12" cy="12" r="6"></circle><circle cx="12" cy="12" r="2"></circle></svg>'''

# ══════════════════════════════════════════════════════════════════════════════
# DATA LAYER - SQLite with Keywords
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class HeadlineRecord:
    """Data class for headline with all metrics."""
    ticker: str
    headline: str
    source: str
    published: datetime
    sentiment: float
    viral_score: float
    keywords: str  # Comma-separated keywords
    headline_hash: str

class DatabaseService:
    """SQLite backend with keyword storage for graph construction."""
    
    def __init__(self, db_path: str = Config.DB_PATH):
        self.db_path = db_path
        self._init_schema()
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def _init_schema(self):
        """Initialize database schema with keywords column."""
        with self.get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS headlines (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    ticker TEXT NOT NULL,
                    headline TEXT NOT NULL,
                    source TEXT,
                    sentiment REAL NOT NULL,
                    viral_score REAL NOT NULL,
                    keywords TEXT,
                    headline_hash TEXT UNIQUE NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ticker ON headlines(ticker)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON headlines(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_viral ON headlines(viral_score)")
    
    def headline_exists(self, headline_hash: str) -> bool:
        """Check for duplicate headlines."""
        with self.get_connection() as conn:
            result = conn.execute(
                "SELECT 1 FROM headlines WHERE headline_hash = ? LIMIT 1",
                (headline_hash,)
            ).fetchone()
            return result is not None
    
    def insert_record(self, record: HeadlineRecord) -> bool:
        """Insert a headline record."""
        if self.headline_exists(record.headline_hash):
            return False
        
        with self.get_connection() as conn:
            conn.execute("""
                INSERT INTO headlines 
                (timestamp, ticker, headline, source, sentiment, viral_score, keywords, headline_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.published.isoformat(),
                record.ticker,
                record.headline,
                record.source,
                record.sentiment,
                record.viral_score,
                record.keywords,
                record.headline_hash
            ))
        return True
    
    def get_ticker_headlines(self, ticker: str, days: int = 30) -> pd.DataFrame:
        """Fetch all headlines for a ticker."""
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        
        with self.get_connection() as conn:
            df = pd.read_sql_query("""
                SELECT id, timestamp, headline, source, sentiment, viral_score, keywords
                FROM headlines
                WHERE ticker = ? AND timestamp >= ?
                ORDER BY timestamp DESC
            """, conn, params=(ticker, cutoff))
        
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    
    def get_daily_aggregates(self, ticker: str, days: int = 30) -> pd.DataFrame:
        """Get daily aggregated metrics."""
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        
        with self.get_connection() as conn:
            df = pd.read_sql_query("""
                SELECT 
                    DATE(timestamp) as date,
                    AVG(sentiment) as avg_sentiment,
                    SUM(viral_score) as total_viral,
                    MAX(viral_score) as max_viral,
                    COUNT(*) as news_count
                FROM headlines
                WHERE ticker = ? AND timestamp >= ?
                GROUP BY DATE(timestamp)
                ORDER BY date
            """, conn, params=(ticker, cutoff))
        
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
        return df
    
    def search_headlines(self, query: str, ticker: str = None, limit: int = 10) -> pd.DataFrame:
        """Search headlines by keywords, ordered by viral score."""
        words = query.lower().split()
        conditions = " OR ".join([f"LOWER(headline) LIKE '%{w}%'" for w in words if len(w) > 2])
        
        if not conditions:
            return pd.DataFrame()
        
        ticker_clause = f"AND ticker = '{ticker}'" if ticker else ""
        
        with self.get_connection() as conn:
            df = pd.read_sql_query(f"""
                SELECT timestamp, ticker, headline, source, sentiment, viral_score
                FROM headlines
                WHERE ({conditions}) {ticker_clause}
                ORDER BY viral_score DESC
                LIMIT ?
            """, conn, params=(limit,))
        
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    
    def get_stats(self) -> Dict:
        """Get database statistics."""
        with self.get_connection() as conn:
            total = conn.execute("SELECT COUNT(*) FROM headlines").fetchone()[0]
            tickers = conn.execute("SELECT COUNT(DISTINCT ticker) FROM headlines").fetchone()[0]
            latest = conn.execute("SELECT MAX(created_at) FROM headlines").fetchone()[0]
            avg_viral = conn.execute("SELECT AVG(viral_score) FROM headlines").fetchone()[0]
        
        return {
            'total_records': total,
            'unique_tickers': tickers,
            'last_update': datetime.fromisoformat(latest) if latest else None,
            'avg_viral_score': avg_viral or 0
        }

# ══════════════════════════════════════════════════════════════════════════════
# NLP ENGINE - VADER + Keyword Extraction
# ══════════════════════════════════════════════════════════════════════════════

class NLPEngine:
    """Sentiment analysis and keyword extraction."""
    
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        self._enhance_lexicon()
    
    def _enhance_lexicon(self):
        """Add financial terms to VADER."""
        financial_terms = {
            'bullish': 2.5, 'bearish': -2.5, 'upgrade': 2.0, 'downgrade': -2.0,
            'beat': 1.5, 'miss': -1.5, 'surge': 2.0, 'plunge': -2.5,
            'breakout': 1.8, 'crash': -3.0, 'rally': 2.0, 'selloff': -2.0,
            'soar': 2.5, 'tank': -2.5, 'squeeze': 1.5, 'fraud': -3.0,
            'lawsuit': -1.5, 'bankruptcy': -3.0, 'acquisition': 1.5, 'merger': 1.0
        }
        self.analyzer.lexicon.update(financial_terms)
    
    def analyze_sentiment(self, text: str) -> float:
        """Get VADER compound sentiment score."""
        if not text:
            return 0.0
        return self.analyzer.polarity_scores(text)['compound']
    
    def calculate_viral_score(self, headline: str, sentiment: float) -> float:
        """
        Calculate viral potential score.
        Formula: viral_score = abs(sentiment) * (100 if len < 80 else 80)
        """
        intensity = abs(sentiment)
        brevity_bonus = 100 if len(headline) < 80 else 80
        return intensity * brevity_bonus
    
    def extract_keywords(self, headline: str) -> Set[str]:
        """Extract significant keywords (non-stopwords) from headline."""
        # Clean and tokenize
        words = re.findall(r'\b[a-zA-Z]+\b', headline.lower())
        
        # Filter: not stopword, length >= 3
        keywords = {
            w for w in words 
            if w not in STOP_WORDS and len(w) >= Config.MIN_KEYWORD_LENGTH
        }
        
        return keywords

# ══════════════════════════════════════════════════════════════════════════════
# GRAPH ENGINE - NetworkX Constellation Builder
# ══════════════════════════════════════════════════════════════════════════════

class ConstellationEngine:
    """Builds narrative constellation graph from headlines."""
    
    def __init__(self):
        self.graph = nx.Graph()
    
    def build_graph(self, df: pd.DataFrame) -> nx.Graph:
        """
        Build a network graph where:
        - Nodes = Headlines (with sentiment, viral_score attributes)
        - Edges = Shared keywords between headlines
        """
        self.graph = nx.Graph()
        
        if df.empty:
            return self.graph
        
        # Add nodes
        for idx, row in df.iterrows():
            self.graph.add_node(
                row['id'],
                headline=row['headline'][:60] + ('...' if len(row['headline']) > 60 else ''),
                full_headline=row['headline'],
                sentiment=row['sentiment'],
                viral_score=row['viral_score'],
                keywords=set(row['keywords'].split(',')) if row['keywords'] else set(),
                source=row['source'],
                timestamp=str(row['timestamp'])[:16]
            )
        
        # Add edges based on shared keywords
        nodes = list(self.graph.nodes(data=True))
        for i, (node1, data1) in enumerate(nodes):
            for node2, data2 in nodes[i+1:]:
                shared = data1['keywords'] & data2['keywords']
                if len(shared) >= Config.MIN_SHARED_KEYWORDS:
                    self.graph.add_edge(
                        node1, node2,
                        weight=len(shared),
                        shared_keywords=list(shared)
                    )
        
        return self.graph
    
    def get_graph_stats(self) -> Dict:
        """Get graph statistics."""
        if self.graph.number_of_nodes() == 0:
            return {'nodes': 0, 'edges': 0, 'clusters': 0, 'density': 0}
        
        return {
            'nodes': self.graph.number_of_nodes(),
            'edges': self.graph.number_of_edges(),
            'clusters': nx.number_connected_components(self.graph),
            'density': nx.density(self.graph)
        }
    
    def calculate_nfi(self) -> float:
        """
        Calculate Narrative Fragmentation Index (NFI).
        
        NFI = 1 - (largest_cluster_size / total_nodes)
        
        - High NFI (>0.7) = Fragmented narrative = Uncertainty/Volatility coming
        - Low NFI (<0.3) = Coherent narrative = Trend continuation likely
        - NFI ~0.5 = Mixed signals = Neutral
        
        This is a NOVEL metric - not available in any existing tool.
        """
        if self.graph.number_of_nodes() == 0:
            return 0.5  # Neutral when no data
        
        total_nodes = self.graph.number_of_nodes()
        
        # Find the largest connected component
        if total_nodes == 1:
            return 0.0  # Single node = perfectly coherent
        
        components = list(nx.connected_components(self.graph))
        if not components:
            return 1.0  # No connections = maximum fragmentation
        
        largest_cluster_size = max(len(c) for c in components)
        
        # NFI = 1 - (dominance of largest cluster)
        nfi = 1.0 - (largest_cluster_size / total_nodes)
        
        return round(nfi, 3)
    
    def get_cluster_breakdown(self) -> List[Dict]:
        """Get detailed cluster breakdown for visualization."""
        if self.graph.number_of_nodes() == 0:
            return []
        
        components = list(nx.connected_components(self.graph))
        breakdown = []
        
        for i, component in enumerate(sorted(components, key=len, reverse=True)[:10]):
            nodes_data = [self.graph.nodes[n] for n in component]
            avg_sentiment = np.mean([n['sentiment'] for n in nodes_data])
            avg_viral = np.mean([n['viral_score'] for n in nodes_data])
            
            # Get dominant keywords in cluster
            all_keywords = []
            for n in nodes_data:
                all_keywords.extend(list(n['keywords']))
            
            from collections import Counter
            top_keywords = [kw for kw, _ in Counter(all_keywords).most_common(3)]
            
            breakdown.append({
                'cluster_id': i + 1,
                'size': len(component),
                'avg_sentiment': avg_sentiment,
                'avg_viral': avg_viral,
                'theme': ', '.join(top_keywords) if top_keywords else 'misc'
            })
        
        return breakdown


class AlphaSignalsEngine:
    """
    Novel Alpha Signal Generation Engine.
    
    Generates unique trading signals not available in existing tools:
    1. NFI (Narrative Fragmentation Index) - from ConstellationEngine
    2. SPDS (Sentiment-Price Dislocation Score) - price vs sentiment divergence
    3. Combined Alpha Score
    """
    
    @staticmethod
    def calculate_spds(price_df: pd.DataFrame, sentiment_df: pd.DataFrame, window: int = 5) -> Dict:
        """
        Calculate Sentiment-Price Dislocation Score (SPDS).
        
        SPDS = Rolling correlation between sentiment and price returns
        
        - SPDS near +1.0 = Sentiment and price moving together (normal)
        - SPDS near 0 = Dislocation beginning (watch closely)
        - SPDS negative = Full dislocation (mean reversion opportunity)
        
        This is a NOVEL metric - not available in any existing tool.
        """
        if price_df.empty or sentiment_df.empty:
            return {'current_spds': 0.0, 'signal': 'NEUTRAL', 'history': []}
        
        # Prepare price returns
        price_df = price_df.copy()
        price_df['date'] = pd.to_datetime(price_df['Date']).dt.date
        price_df['returns'] = price_df['Close'].pct_change()
        
        # Prepare sentiment
        sentiment_df = sentiment_df.copy()
        sentiment_df['date'] = pd.to_datetime(sentiment_df['date']).dt.date
        
        # Merge on date
        merged = pd.merge(
            price_df[['date', 'returns']],
            sentiment_df[['date', 'avg_sentiment']],
            on='date',
            how='inner'
        ).dropna()
        
        if len(merged) < window:
            return {'current_spds': 0.0, 'signal': 'INSUFFICIENT_DATA', 'history': []}
        
        # Calculate rolling correlation
        merged['spds'] = merged['returns'].rolling(window).corr(merged['avg_sentiment'])
        merged = merged.dropna()
        
        if merged.empty:
            return {'current_spds': 0.0, 'signal': 'NEUTRAL', 'history': []}
        
        current_spds = merged['spds'].iloc[-1]
        
        # Generate signal
        if current_spds < -0.3:
            signal = 'DISLOCATION'
        elif current_spds < 0.2:
            signal = 'DIVERGING'
        elif current_spds > 0.6:
            signal = 'ALIGNED'
        else:
            signal = 'NEUTRAL'
        
        history = merged[['date', 'spds', 'returns', 'avg_sentiment']].to_dict('records')
        
        return {
            'current_spds': round(current_spds, 3),
            'signal': signal,
            'history': history
        }
    
    @staticmethod
    def generate_alpha_score(nfi: float, spds: float) -> Dict:
        """
        Generate combined Alpha Score from NFI and SPDS.
        
        Alpha Score interpretation:
        - High NFI + Negative SPDS = Strong contrarian signal (volatility + dislocation)
        - Low NFI + Positive SPDS = Trend following signal (coherent + aligned)
        - Mixed = Neutral, wait for clearer signal
        """
        # Contrarian score: high when fragmented AND dislocated
        contrarian = (nfi * 0.6) + (max(0, -spds) * 0.4)
        
        # Trend score: high when coherent AND aligned
        trend = ((1 - nfi) * 0.6) + (max(0, spds) * 0.4)
        
        if contrarian > 0.5:
            signal = 'CONTRARIAN'
            action = 'Consider mean-reversion plays'
            confidence = min(contrarian * 100, 95)
        elif trend > 0.5:
            signal = 'TREND'
            action = 'Consider momentum plays'
            confidence = min(trend * 100, 95)
        else:
            signal = 'NEUTRAL'
            action = 'Wait for clearer signal'
            confidence = 50
        
        return {
            'signal': signal,
            'action': action,
            'confidence': round(confidence, 1),
            'contrarian_score': round(contrarian, 3),
            'trend_score': round(trend, 3)
        }

# ══════════════════════════════════════════════════════════════════════════════
# INGESTION SERVICE
# ══════════════════════════════════════════════════════════════════════════════

class IngestionService:
    """RSS feed ingestion with NLP processing."""
    
    def __init__(self, db: DatabaseService, nlp: NLPEngine):
        self.db = db
        self.nlp = nlp
    
    @staticmethod
    def _generate_hash(headline: str, ticker: str) -> str:
        content = f"{ticker}:{headline.lower().strip()}"
        return hashlib.sha256(content.encode()).hexdigest()[:32]
    
    @staticmethod
    def _parse_date(entry: dict) -> datetime:
        if hasattr(entry, 'published_parsed') and entry.published_parsed:
            try:
                return datetime(*entry.published_parsed[:6])
            except:
                pass
        return datetime.now()
    
    @staticmethod
    def _extract_source(entry: dict) -> str:
        if hasattr(entry, 'source') and hasattr(entry.source, 'title'):
            return entry.source.title
        title = entry.get('title', '')
        if ' - ' in title:
            return title.split(' - ')[-1].strip()
        return "Unknown"
    
    def ingest(self, ticker: str) -> Tuple[int, int, str]:
        """Ingest news for a ticker."""
        url = Config.RSS_BASE_URL.format(ticker=ticker)
        new_count = 0
        dup_count = 0
        
        try:
            feed = feedparser.parse(url)
            
            if feed.bozo and not feed.entries:
                return 0, 0, f"Feed error"
            
            if not feed.entries:
                return 0, 0, "No news found"
            
            for entry in feed.entries:
                headline = entry.get('title', '').strip()
                if not headline:
                    continue
                
                # Clean headline
                if ' - ' in headline:
                    headline = ' - '.join(headline.split(' - ')[:-1])
                
                headline_hash = self._generate_hash(headline, ticker)
                
                if self.db.headline_exists(headline_hash):
                    dup_count += 1
                    continue
                
                # NLP processing
                sentiment = self.nlp.analyze_sentiment(headline)
                viral_score = self.nlp.calculate_viral_score(headline, sentiment)
                keywords = self.nlp.extract_keywords(headline)
                
                record = HeadlineRecord(
                    ticker=ticker,
                    headline=headline,
                    source=self._extract_source(entry),
                    published=self._parse_date(entry),
                    sentiment=sentiment,
                    viral_score=viral_score,
                    keywords=','.join(keywords),
                    headline_hash=headline_hash
                )
                
                if self.db.insert_record(record):
                    new_count += 1
            
            return new_count, dup_count, "Success"
            
        except Exception as e:
            return 0, 0, f"Error: {str(e)}"

# ══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class VisualizationEngine:
    """Plotly visualization engine."""
    
    @staticmethod
    def create_constellation_chart(graph: nx.Graph) -> go.Figure:
        """
        Create interactive constellation visualization.
        Nodes colored by sentiment, sized by viral score.
        """
        fig = go.Figure()
        
        if graph.number_of_nodes() == 0:
            # Empty state
            fig.add_annotation(
                text="No Data Available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=20, color='#64748B')
            )
            fig.add_annotation(
                text="Click 'Ingest Data' to fetch headlines",
                xref="paper", yref="paper",
                x=0.5, y=0.4, showarrow=False,
                font=dict(size=14, color='#475569')
            )
        else:
            # Calculate layout
            pos = nx.spring_layout(graph, k=2, iterations=50, seed=42)
            
            # Draw edges first
            edge_x = []
            edge_y = []
            for edge in graph.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
            
            fig.add_trace(go.Scatter(
                x=edge_x, y=edge_y,
                mode='lines',
                line=dict(width=0.8, color='rgba(100, 116, 139, 0.3)'),
                hoverinfo='none',
                name='Connections'
            ))
            
            # Draw nodes
            node_x = []
            node_y = []
            node_colors = []
            node_sizes = []
            node_texts = []
            node_hovers = []
            
            for node in graph.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                
                data = graph.nodes[node]
                sentiment = data['sentiment']
                viral = data['viral_score']
                
                # Color by sentiment
                if sentiment > 0.15:
                    color = '#00FF88'
                elif sentiment < -0.15:
                    color = '#FF3366'
                else:
                    color = '#64748B'
                
                node_colors.append(color)
                node_sizes.append(max(15, min(50, viral / 2)))  # Scale size
                node_texts.append('')
                
                hover = f"<b>{data['headline']}</b><br>"
                hover += f"Sentiment: {sentiment:+.3f}<br>"
                hover += f"Viral Score: {viral:.1f}<br>"
                hover += f"Source: {data['source']}<br>"
                hover += f"Time: {data['timestamp']}"
                node_hovers.append(hover)
            
            fig.add_trace(go.Scatter(
                x=node_x, y=node_y,
                mode='markers',
                marker=dict(
                    size=node_sizes,
                    color=node_colors,
                    opacity=0.8,
                    line=dict(width=1, color='rgba(255,255,255,0.3)')
                ),
                text=node_hovers,
                hoverinfo='text',
                name='Headlines'
            ))
        
        # Layout
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='#0E1117',
            plot_bgcolor='#0E1117',
            height=550,
            margin=dict(l=20, r=20, t=40, b=20),
            showlegend=False,
            xaxis=dict(
                showgrid=False, zeroline=False, showticklabels=False,
                showline=False
            ),
            yaxis=dict(
                showgrid=False, zeroline=False, showticklabels=False,
                showline=False
            ),
            title=dict(
                text='Narrative Constellation',
                font=dict(size=16, color='#F1F5F9'),
                x=0.5
            ),
            hoverlabel=dict(
                bgcolor='rgba(15, 23, 42, 0.95)',
                bordercolor='rgba(100, 116, 139, 0.3)',
                font=dict(color='#F1F5F9', size=12)
            )
        )
        
        return fig
    
    @staticmethod
    @st.cache_data(ttl=300)
    def fetch_price_data(ticker: str, days: int = 30) -> pd.DataFrame:
        """Fetch OHLCV data from yfinance."""
        try:
            end = datetime.now()
            start = end - timedelta(days=days)
            df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
            if df.empty:
                return pd.DataFrame()
            df = df.reset_index()
            df.columns = df.columns.get_level_values(0)
            return df
        except:
            return pd.DataFrame()
    
    @staticmethod
    def create_price_chart(price_df: pd.DataFrame, viral_df: pd.DataFrame, ticker: str) -> go.Figure:
        """Create dual-axis price + viral chart."""
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            row_heights=[0.7, 0.3],
            specs=[[{"secondary_y": True}], [{"secondary_y": False}]]
        )
        
        if not price_df.empty:
            # Candlestick
            fig.add_trace(
                go.Candlestick(
                    x=price_df['Date'],
                    open=price_df['Open'],
                    high=price_df['High'],
                    low=price_df['Low'],
                    close=price_df['Close'],
                    name='Price',
                    increasing_line_color='#00FF88',
                    decreasing_line_color='#FF3366',
                    increasing_fillcolor='rgba(0, 255, 136, 0.3)',
                    decreasing_fillcolor='rgba(255, 51, 102, 0.3)'
                ),
                row=1, col=1, secondary_y=False
            )
        
        if not viral_df.empty:
            # Viral bars
            colors = ['#00FF88' if s > 0.1 else '#FF3366' if s < -0.1 else '#64748B' 
                     for s in viral_df['avg_sentiment']]
            
            fig.add_trace(
                go.Bar(
                    x=viral_df['date'],
                    y=viral_df['total_viral'],
                    name='Viral Volume',
                    marker_color=colors,
                    opacity=0.5
                ),
                row=1, col=1, secondary_y=True
            )
            
            # Sentiment line
            fig.add_trace(
                go.Scatter(
                    x=viral_df['date'],
                    y=viral_df['avg_sentiment'],
                    mode='lines+markers',
                    name='Sentiment',
                    line=dict(color='#3B82F6', width=2),
                    marker=dict(size=5),
                    fill='tozeroy',
                    fillcolor='rgba(59, 130, 246, 0.1)'
                ),
                row=2, col=1
            )
            
            fig.add_hline(y=0, line_dash="dash", line_color="rgba(148, 163, 184, 0.3)", row=2, col=1)
        
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='#0E1117',
            plot_bgcolor='rgba(15, 23, 42, 0.5)',
            height=550,
            margin=dict(l=60, r=60, t=40, b=40),
            showlegend=True,
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                bgcolor='rgba(0,0,0,0)', font=dict(color='#94A3B8', size=11)
            ),
            xaxis_rangeslider_visible=False,
            title=dict(text=f'{ticker} Price Action vs Narrative Contagion', font=dict(size=16, color='#F1F5F9'), x=0.5)
        )
        
        fig.update_xaxes(gridcolor='rgba(100, 116, 139, 0.1)')
        fig.update_yaxes(gridcolor='rgba(100, 116, 139, 0.1)')
        fig.update_yaxes(title_text="Price ($)", row=1, col=1, secondary_y=False)
        fig.update_yaxes(title_text="Viral Vol", row=1, col=1, secondary_y=True)
        fig.update_yaxes(title_text="Sentiment", row=2, col=1, range=[-1, 1])
        
        return fig
    
    @staticmethod
    def create_nfi_gauge(nfi: float) -> go.Figure:
        """Create NFI gauge visualization."""
        # Determine color and interpretation
        if nfi > 0.7:
            color = "#EF4444"  # Red - high fragmentation
            interpretation = "HIGH FRAGMENTATION"
        elif nfi > 0.4:
            color = "#F59E0B"  # Amber - moderate
            interpretation = "MODERATE"
        else:
            color = "#10B981"  # Green - coherent
            interpretation = "COHERENT"
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=nfi,
            number={'suffix': '', 'font': {'size': 36, 'color': '#F1F5F9'}},
            title={'text': f"NFI<br><span style='font-size:0.7em;color:{color}'>{interpretation}</span>", 
                   'font': {'size': 14, 'color': '#94A3B8'}},
            gauge={
                'axis': {'range': [0, 1], 'tickwidth': 1, 'tickcolor': '#64748B'},
                'bar': {'color': color, 'thickness': 0.75},
                'bgcolor': 'rgba(30, 41, 59, 0.5)',
                'borderwidth': 2,
                'bordercolor': 'rgba(100, 116, 139, 0.3)',
                'steps': [
                    {'range': [0, 0.3], 'color': 'rgba(16, 185, 129, 0.2)'},
                    {'range': [0.3, 0.7], 'color': 'rgba(245, 158, 11, 0.2)'},
                    {'range': [0.7, 1], 'color': 'rgba(239, 68, 68, 0.2)'}
                ],
                'threshold': {
                    'line': {'color': '#F1F5F9', 'width': 2},
                    'thickness': 0.75,
                    'value': nfi
                }
            }
        ))
        
        fig.update_layout(
            paper_bgcolor='#0E1117',
            plot_bgcolor='#0E1117',
            height=250,
            margin=dict(l=30, r=30, t=60, b=20)
        )
        
        return fig
    
    @staticmethod
    def create_spds_gauge(spds: float, signal: str) -> go.Figure:
        """Create SPDS gauge visualization."""
        # Color based on signal
        colors = {
            'DISLOCATION': '#EF4444',
            'DIVERGING': '#F59E0B',
            'NEUTRAL': '#64748B',
            'ALIGNED': '#10B981',
            'INSUFFICIENT_DATA': '#475569'
        }
        color = colors.get(signal, '#64748B')
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=spds,
            number={'font': {'size': 36, 'color': '#F1F5F9'}},
            title={'text': f"SPDS<br><span style='font-size:0.7em;color:{color}'>{signal}</span>", 
                   'font': {'size': 14, 'color': '#94A3B8'}},
            gauge={
                'axis': {'range': [-1, 1], 'tickwidth': 1, 'tickcolor': '#64748B'},
                'bar': {'color': color, 'thickness': 0.75},
                'bgcolor': 'rgba(30, 41, 59, 0.5)',
                'borderwidth': 2,
                'bordercolor': 'rgba(100, 116, 139, 0.3)',
                'steps': [
                    {'range': [-1, -0.3], 'color': 'rgba(239, 68, 68, 0.2)'},
                    {'range': [-0.3, 0.3], 'color': 'rgba(100, 116, 139, 0.2)'},
                    {'range': [0.3, 1], 'color': 'rgba(16, 185, 129, 0.2)'}
                ],
                'threshold': {
                    'line': {'color': '#F1F5F9', 'width': 2},
                    'thickness': 0.75,
                    'value': spds
                }
            }
        ))
        
        fig.update_layout(
            paper_bgcolor='#0E1117',
            plot_bgcolor='#0E1117',
            height=250,
            margin=dict(l=30, r=30, t=60, b=20)
        )
        
        return fig
    
    @staticmethod
    def create_spds_history_chart(spds_history: List[Dict]) -> go.Figure:
        """Create SPDS history line chart."""
        fig = go.Figure()
        
        if not spds_history:
            fig.add_annotation(
                text="Insufficient Data for SPDS History",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color='#64748B')
            )
        else:
            dates = [h['date'] for h in spds_history]
            spds_vals = [h['spds'] for h in spds_history]
            
            # SPDS line
            fig.add_trace(go.Scatter(
                x=dates, y=spds_vals,
                mode='lines+markers',
                name='SPDS',
                line=dict(color='#8B5CF6', width=2),
                marker=dict(size=6),
                fill='tozeroy',
                fillcolor='rgba(139, 92, 246, 0.1)'
            ))
            
            # Reference lines
            fig.add_hline(y=0, line_dash="solid", line_color="rgba(148, 163, 184, 0.5)", line_width=1)
            fig.add_hline(y=0.3, line_dash="dash", line_color="rgba(16, 185, 129, 0.5)", line_width=1,
                         annotation_text="Aligned", annotation_position="right")
            fig.add_hline(y=-0.3, line_dash="dash", line_color="rgba(239, 68, 68, 0.5)", line_width=1,
                         annotation_text="Dislocation", annotation_position="right")
        
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='#0E1117',
            plot_bgcolor='rgba(15, 23, 42, 0.5)',
            height=300,
            margin=dict(l=50, r=80, t=40, b=40),
            title=dict(text='Sentiment-Price Dislocation History', font=dict(size=14, color='#F1F5F9'), x=0.5),
            yaxis=dict(title='SPDS', range=[-1.1, 1.1], gridcolor='rgba(100, 116, 139, 0.1)'),
            xaxis=dict(gridcolor='rgba(100, 116, 139, 0.1)')
        )
        
        return fig
    
    @staticmethod
    def create_cluster_breakdown_chart(clusters: List[Dict]) -> go.Figure:
        """Create cluster breakdown bar chart."""
        fig = go.Figure()
        
        if not clusters:
            fig.add_annotation(
                text="No Clusters to Display",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color='#64748B')
            )
        else:
            # Prepare data
            labels = [f"C{c['cluster_id']}: {c['theme'][:15]}" for c in clusters]
            sizes = [c['size'] for c in clusters]
            sentiments = [c['avg_sentiment'] for c in clusters]
            colors = ['#10B981' if s > 0.1 else '#EF4444' if s < -0.1 else '#64748B' for s in sentiments]
            
            fig.add_trace(go.Bar(
                x=sizes,
                y=labels,
                orientation='h',
                marker_color=colors,
                text=[f"{s} nodes" for s in sizes],
                textposition='auto',
                hovertemplate='<b>%{y}</b><br>Size: %{x} nodes<extra></extra>'
            ))
        
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='#0E1117',
            plot_bgcolor='rgba(15, 23, 42, 0.5)',
            height=300,
            margin=dict(l=120, r=40, t=40, b=40),
            title=dict(text='Narrative Cluster Breakdown', font=dict(size=14, color='#F1F5F9'), x=0.5),
            xaxis=dict(title='Headlines', gridcolor='rgba(100, 116, 139, 0.1)'),
            yaxis=dict(gridcolor='rgba(100, 116, 139, 0.1)')
        )
        
        return fig

# ══════════════════════════════════════════════════════════════════════════════
# PRESENTATION LAYER - STREAMLIT UI
# ══════════════════════════════════════════════════════════════════════════════

def apply_theme():
    """Apply professional dark theme."""
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');
        
        .stApp {
            background: linear-gradient(135deg, #0E1117 0%, #1A1F2E 50%, #0E1117 100%);
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }
        
        #MainMenu, footer, header, .stDeployButton {visibility: hidden;}
        
        .glass-card {
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.6), rgba(15, 23, 42, 0.8));
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 1px solid rgba(100, 116, 139, 0.2);
            border-radius: 16px;
            padding: 1.5rem;
            margin: 0.75rem 0;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        }
        
        .glass-card-sm {
            background: rgba(30, 41, 59, 0.4);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(100, 116, 139, 0.15);
            border-radius: 12px;
            padding: 1rem;
            margin: 0.5rem 0;
        }
        
        .metric-label {
            color: #64748B;
            font-size: 0.7rem;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.1em;
        }
        
        .metric-value {
            color: #F1F5F9;
            font-size: 1.5rem;
            font-weight: 700;
            font-family: 'JetBrains Mono', monospace;
        }
        
        .insight-card {
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.5), rgba(15, 23, 42, 0.6));
            backdrop-filter: blur(10px);
            border-radius: 12px;
            padding: 1rem;
            margin-bottom: 0.75rem;
            border-left: 3px solid;
        }
        
        .insight-headline {
            color: #E2E8F0;
            font-size: 0.9rem;
            font-weight: 500;
            line-height: 1.4;
            margin-bottom: 0.5rem;
        }
        
        .insight-meta {
            color: #64748B;
            font-size: 0.75rem;
        }
        
        .section-header {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            color: #F1F5F9;
            font-size: 1rem;
            font-weight: 600;
            margin: 1.25rem 0 0.75rem 0;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid rgba(100, 116, 139, 0.2);
        }
        
        .status-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.35rem 0.75rem;
            background: rgba(0, 255, 136, 0.1);
            border: 1px solid rgba(0, 255, 136, 0.3);
            border-radius: 20px;
            color: #00FF88;
            font-size: 0.75rem;
            font-weight: 500;
        }
        
        .status-dot {
            width: 6px;
            height: 6px;
            background: #00FF88;
            border-radius: 50%;
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0F172A 0%, #1E293B 100%);
            border-right: 1px solid rgba(100, 116, 139, 0.2);
            min-width: 300px !important;
            width: 300px !important;
        }
        
        section[data-testid="stSidebar"] > div {
            padding-top: 1rem;
        }
        
        [data-testid="stSidebarCollapsedControl"] {
            display: none;
        }
        
        section[data-testid="stSidebar"] .stButton > button {
            width: 100%;
            background: linear-gradient(135deg, #3B82F6, #1D4ED8);
            color: white;
            border: none;
            border-radius: 10px;
            padding: 0.7rem 1rem;
            font-weight: 600;
            font-size: 0.85rem;
            transition: all 0.3s ease;
        }
        
        section[data-testid="stSidebar"] .stButton > button:hover {
            background: linear-gradient(135deg, #60A5FA, #3B82F6);
            transform: translateY(-2px);
            box-shadow: 0 4px 15px rgba(59, 130, 246, 0.4);
        }
        
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background: transparent;
        }
        
        .stTabs [data-baseweb="tab"] {
            background: rgba(30, 41, 59, 0.5);
            border-radius: 8px;
            padding: 8px 16px;
            color: #94A3B8;
            border: 1px solid rgba(100, 116, 139, 0.2);
        }
        
        .stTabs [aria-selected="true"] {
            background: rgba(59, 130, 246, 0.2);
            color: #F1F5F9;
            border-color: #3B82F6;
        }
        
        ::-webkit-scrollbar { width: 6px; height: 6px; }
        ::-webkit-scrollbar-track { background: rgba(15, 23, 42, 0.5); }
        ::-webkit-scrollbar-thumb { background: rgba(100, 116, 139, 0.5); border-radius: 3px; }
    </style>
    """, unsafe_allow_html=True)

def render_metric(label: str, value: str, icon_html: str, color: str = "#F1F5F9"):
    """Render a metric display."""
    st.markdown(f"""
    <div class="glass-card" style="padding: 1rem;">
        <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.4rem;">
            {icon_html}
            <span class="metric-label">{label}</span>
        </div>
        <div class="metric-value" style="color: {color};">{value}</div>
    </div>
    """, unsafe_allow_html=True)

def render_insight_card(headline: str, sentiment: float, viral: float, source: str, time_str: str):
    """Render an insight card from query results."""
    if sentiment > 0.15:
        color = "#00FF88"
        icon = Icons.arrow_up_right("#00FF88", 16)
    elif sentiment < -0.15:
        color = "#FF3366"
        icon = Icons.arrow_down_right("#FF3366", 16)
    else:
        color = "#64748B"
        icon = Icons.minus_circle("#64748B", 16)
    
    st.markdown(f"""
    <div class="insight-card" style="border-left-color: {color};">
        <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
            {icon}
            <span style="color: {color}; font-size: 0.7rem; text-transform: uppercase; font-weight: 600;">
                {sentiment:+.3f}
            </span>
            <span style="color: #FFB800; font-size: 0.7rem; margin-left: auto; font-family: 'JetBrains Mono', monospace;">
                VP: {viral:.1f}
            </span>
        </div>
        <div class="insight-headline">{headline}</div>
        <div class="insight-meta">{source} | {time_str}</div>
    </div>
    """, unsafe_allow_html=True)

def main():
    """Main application entry point."""
    st.set_page_config(
        page_title="Narrative Constellation Terminal",
        page_icon="chart_with_upwards_trend",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    apply_theme()
    
    # Initialize services
    db = DatabaseService()
    nlp = NLPEngine()
    ingestion = IngestionService(db, nlp)
    constellation = ConstellationEngine()
    viz = VisualizationEngine()
    
    # Initialize shared UI state
    if 'selected_ticker' not in st.session_state:
        st.session_state['selected_ticker'] = Config.DEFAULT_TICKERS[0]

    # ══════════════════════════════════════════════════════════════════════════
    # SIDEBAR
    # ══════════════════════════════════════════════════════════════════════════
    
    with st.sidebar:
        st.markdown(f"""
        <div style="text-align: center; padding: 1rem 0;">
            <div style="margin-bottom: 0.5rem;">{Icons.constellation('#00D9FF', 36)}</div>
            <h1 style="color: #F1F5F9; font-size: 1.1rem; margin: 0; font-weight: 700;">
                Narrative Constellation
            </h1>
            <p style="color: #64748B; font-size: 0.75rem; margin-top: 0.3rem;">
                Alpha Signals Edition v4.0
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Ticker selection
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.4rem;">
            {Icons.candlestick('#00FF88', 14)}
            <span style="color: #94A3B8; font-size: 0.8rem; font-weight: 500;">Select Ticker</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.session_state['selected_ticker'] = st.selectbox(
            "Ticker",
            Config.DEFAULT_TICKERS,
            index=Config.DEFAULT_TICKERS.index(st.session_state['selected_ticker']) if st.session_state['selected_ticker'] in Config.DEFAULT_TICKERS else 0,
            label_visibility="collapsed",
            key="ticker_select_sidebar"
        )
        custom = st.text_input("Custom", placeholder="e.g., COIN", label_visibility="collapsed", key="custom_ticker_sidebar")
        if custom:
            st.session_state['selected_ticker'] = custom.upper().strip()
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Ingest controls
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.4rem;">
            {Icons.refresh('#00D9FF', 14)}
            <span style="color: #94A3B8; font-size: 0.8rem; font-weight: 500;">Data Controls</span>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Ingest Data", key="ingest"):
            with st.spinner(f"Fetching {st.session_state['selected_ticker']}..."):
                new, dup, status = ingestion.ingest(selected_ticker)
                if status == "Success":
                    st.success(f"{new} new headlines")
                else:
                    st.warning(status)
        
        if st.button("Ingest All Tickers", key="ingest_all"):
            progress = st.progress(0)
            total = 0
            for i, t in enumerate(Config.DEFAULT_TICKERS):
                new, _, _ = ingestion.ingest(t)
                total += new
                progress.progress((i + 1) / len(Config.DEFAULT_TICKERS))
                time.sleep(0.2)
            st.success(f"{total} total headlines")
        
        st.markdown("---")
        
        # Tactical Query
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.4rem;">
            {Icons.search('#94A3B8', 14)}
            <span style="color: #94A3B8; font-size: 0.8rem; font-weight: 500;">Tactical Query</span>
        </div>
        """, unsafe_allow_html=True)
        
        query = st.text_input("Ask the Data", placeholder="Why is the stock down?", label_visibility="collapsed", key="query_sidebar")
        
        if query:
            results = db.search_headlines(query, st.session_state['selected_ticker'], limit=3)
            if not results.empty:
                st.markdown(f"""
                <div style="display: flex; align-items: center; gap: 0.5rem; margin: 0.75rem 0 0.5rem 0;">
                    {Icons.insight('#FFB800', 14)}
                    <span style="color: #FFB800; font-size: 0.75rem; font-weight: 600;">TOP INSIGHTS</span>
                </div>
                """, unsafe_allow_html=True)
                
                for _, row in results.iterrows():
                    render_insight_card(
                        row['headline'],
                        row['sentiment'],
                        row['viral_score'],
                        row['source'],
                        str(row['timestamp'])[:16]
                    )
            else:
                st.markdown("""
                <div class="glass-card-sm" style="text-align: center;">
                    <p style="color: #64748B; font-size: 0.8rem; margin: 0;">No matching headlines</p>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # System status
        stats = db.get_stats()
        
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
            {Icons.database('#8B5CF6', 14)}
            <span style="color: #94A3B8; font-size: 0.8rem; font-weight: 500;">System Status</span>
        </div>
        <div class="glass-card-sm">
            <div class="status-badge">
                <span class="status-dot"></span>
                Online
            </div>
            <div style="margin-top: 0.75rem; color: #94A3B8; font-size: 0.75rem; line-height: 1.8;">
                <div style="display: flex; justify-content: space-between;">
                    <span>Records</span>
                    <span style="color: #F1F5F9; font-family: 'JetBrains Mono', monospace;">{stats['total_records']:,}</span>
                </div>
                <div style="display: flex; justify-content: space-between;">
                    <span>Tickers</span>
                    <span style="color: #F1F5F9; font-family: 'JetBrains Mono', monospace;">{stats['unique_tickers']}</span>
                </div>
                <div style="display: flex; justify-content: space-between;">
                    <span>Avg Viral</span>
                    <span style="color: #FFB800; font-family: 'JetBrains Mono', monospace;">{stats['avg_viral_score']:.1f}</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # ══════════════════════════════════════════════════════════════════════════
    # MAIN CONTENT
    # ══════════════════════════════════════════════════════════════════════════

    # Toggle popover for controls (non-breaking addition)
    pop_cols = st.columns([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    with pop_cols[-1]:
        pop = st.popover("Controls")
    with pop:
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.6rem;">
            {Icons.constellation('#00D9FF', 18)}
            <span style="color: #F1F5F9; font-size: 0.95rem; font-weight: 600;">Quick Controls</span>
        </div>
        """, unsafe_allow_html=True)

        # Ticker selection (synced with sidebar)
        st.session_state['selected_ticker'] = st.selectbox(
            "Ticker",
            Config.DEFAULT_TICKERS,
            index=Config.DEFAULT_TICKERS.index(st.session_state['selected_ticker']) if st.session_state['selected_ticker'] in Config.DEFAULT_TICKERS else 0,
            key="ticker_select_popover"
        )
        custom_pop = st.text_input("Custom", placeholder="e.g., COIN", key="custom_ticker_popover")
        if custom_pop:
            st.session_state['selected_ticker'] = custom_pop.upper().strip()

        st.markdown("<br>", unsafe_allow_html=True)

        # Ingest controls (shared services)
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Ingest Data", key="ingest_pop"):
                with st.spinner(f"Fetching {st.session_state['selected_ticker']}..."):
                    new, dup, status = ingestion.ingest(st.session_state['selected_ticker'])
                    if status == "Success":
                        st.success(f"{new} new headlines")
                    else:
                        st.warning(status)
        with c2:
            if st.button("Ingest All", key="ingest_all_pop"):
                progress = st.progress(0)
                total = 0
                for i, t in enumerate(Config.DEFAULT_TICKERS):
                    new, _, _ = ingestion.ingest(t)
                    total += new
                    progress.progress((i + 1) / len(Config.DEFAULT_TICKERS))
                    time.sleep(0.2)
                st.success(f"{total} total headlines")

        st.markdown("<br>", unsafe_allow_html=True)

        # Tactical Query (shared)
        query_pop = st.text_input("Ask the Data", placeholder="Why is the stock down?", key="query_popover")
        if query_pop:
            results = db.search_headlines(query_pop, st.session_state['selected_ticker'], limit=3)
            if not results.empty:
                for _, row in results.iterrows():
                    render_insight_card(
                        row['headline'],
                        row['sentiment'],
                        row['viral_score'],
                        row['source'],
                        str(row['timestamp'])[:16]
                    )
            else:
                st.info("No matching headlines")
    
    # Header
    st.markdown(f"""
    <div style="margin-bottom: 1.5rem;">
        <h1 style="color: #F1F5F9; font-size: 1.5rem; margin: 0; font-weight: 700;">
            {st.session_state['selected_ticker']} Narrative Analysis
        </h1>
        <p style="color: #64748B; font-size: 0.85rem; margin-top: 0.3rem;">
            Graph-theoretic visualization of narrative clusters and contagion patterns
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Fetch data
    selected_ticker = st.session_state['selected_ticker']
    headlines_df = db.get_ticker_headlines(selected_ticker)
    daily_df = db.get_daily_aggregates(selected_ticker)
    price_df = viz.fetch_price_data(selected_ticker)
    
    # Build graph
    graph = constellation.build_graph(headlines_df)
    graph_stats = constellation.get_graph_stats()
    
    # ══════════════════════════════════════════════════════════════════════════
    # METRICS ROW
    # ══════════════════════════════════════════════════════════════════════════
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if not headlines_df.empty:
            avg_sent = headlines_df['sentiment'].mean()
            color = "#00FF88" if avg_sent > 0.1 else "#FF3366" if avg_sent < -0.1 else "#64748B"
            render_metric("Avg Sentiment", f"{avg_sent:+.3f}", Icons.get_sentiment_icon(avg_sent), color)
        else:
            render_metric("Avg Sentiment", "N/A", Icons.minus_circle('#64748B'), "#64748B")
    
    with col2:
        render_metric("Graph Nodes", str(graph_stats['nodes']), Icons.network('#3B82F6', 16), "#3B82F6")
    
    with col3:
        render_metric("Connections", str(graph_stats['edges']), Icons.constellation('#00D9FF', 16), "#00D9FF")
    
    with col4:
        render_metric("Clusters", str(graph_stats['clusters']), Icons.zap('#FFB800', 16), "#FFB800")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ══════════════════════════════════════════════════════════════════════════
    # TABS
    # ══════════════════════════════════════════════════════════════════════════
    
    # Calculate Alpha Signals
    nfi = constellation.calculate_nfi()
    spds_result = AlphaSignalsEngine.calculate_spds(price_df, daily_df)
    alpha_score = AlphaSignalsEngine.generate_alpha_score(nfi, spds_result['current_spds'])
    cluster_breakdown = constellation.get_cluster_breakdown()
    
    tab1, tab2, tab3 = st.tabs(["Narrative Constellation", "Price Action", "Alpha Signals"])
    
    with tab1:
        st.markdown(f"""
        <div class="section-header">
            {Icons.constellation('#00D9FF', 18)}
            <span>Headline Network Graph</span>
            <span style="margin-left: auto; color: #64748B; font-size: 0.75rem; font-weight: 400;">
                Nodes = Headlines | Edges = Shared Keywords | Size = Viral Score | Color = Sentiment
            </span>
        </div>
        """, unsafe_allow_html=True)
        
        constellation_fig = viz.create_constellation_chart(graph)
        st.plotly_chart(constellation_fig, width="stretch")
    
    with tab2:
        st.markdown(f"""
        <div class="section-header">
            {Icons.candlestick('#00FF88', 18)}
            <span>Price vs Narrative Contagion</span>
        </div>
        """, unsafe_allow_html=True)
        
        price_fig = viz.create_price_chart(price_df, daily_df, selected_ticker)
        st.plotly_chart(price_fig, width="stretch")
    
    with tab3:
        # Alpha Signals Tab - Novel Metrics
        st.markdown(f"""
        <div class="section-header">
            {Icons.radar('#8B5CF6', 18)}
            <span>Alpha Signals - Novel Metrics</span>
            <span style="margin-left: auto; color: #64748B; font-size: 0.75rem; font-weight: 400;">
                NFI + SPDS = Unique trading signals not available elsewhere
            </span>
        </div>
        """, unsafe_allow_html=True)
        
        # Alpha Score Banner
        signal_colors = {'CONTRARIAN': '#EF4444', 'TREND': '#10B981', 'NEUTRAL': '#64748B'}
        signal_color = signal_colors.get(alpha_score['signal'], '#64748B')
        
        st.markdown(f"""
        <div class="glass-card" style="text-align: center; padding: 1.5rem;">
            <div style="display: flex; align-items: center; justify-content: center; gap: 0.75rem; margin-bottom: 0.75rem;">
                {Icons.target(signal_color, 24)}
                <span style="color: {signal_color}; font-size: 1.5rem; font-weight: 700; font-family: 'JetBrains Mono', monospace;">
                    {alpha_score['signal']} SIGNAL
                </span>
            </div>
            <p style="color: #94A3B8; font-size: 0.9rem; margin: 0.5rem 0;">
                {alpha_score['action']}
            </p>
            <div style="display: flex; justify-content: center; gap: 2rem; margin-top: 1rem;">
                <div>
                    <span style="color: #64748B; font-size: 0.7rem;">CONFIDENCE</span><br>
                    <span style="color: #F1F5F9; font-size: 1.2rem; font-weight: 600;">{alpha_score['confidence']:.0f}%</span>
                </div>
                <div>
                    <span style="color: #64748B; font-size: 0.7rem;">CONTRARIAN</span><br>
                    <span style="color: #EF4444; font-size: 1.2rem; font-weight: 600;">{alpha_score['contrarian_score']:.2f}</span>
                </div>
                <div>
                    <span style="color: #64748B; font-size: 0.7rem;">TREND</span><br>
                    <span style="color: #10B981; font-size: 1.2rem; font-weight: 600;">{alpha_score['trend_score']:.2f}</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Gauges Row
        gauge_col1, gauge_col2 = st.columns(2)
        
        with gauge_col1:
            st.markdown(f"""
            <div class="glass-card-sm">
                <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
                    {Icons.divergence('#F59E0B', 16)}
                    <span style="color: #F1F5F9; font-size: 0.85rem; font-weight: 600;">Narrative Fragmentation Index</span>
                </div>
                <p style="color: #64748B; font-size: 0.75rem; margin: 0;">
                    Measures narrative coherence. High NFI = fragmented stories = uncertainty ahead.
                </p>
            </div>
            """, unsafe_allow_html=True)
            nfi_fig = viz.create_nfi_gauge(nfi)
            st.plotly_chart(nfi_fig, width="stretch")
        
        with gauge_col2:
            st.markdown(f"""
            <div class="glass-card-sm">
                <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
                    {Icons.pulse('#10B981', 16)}
                    <span style="color: #F1F5F9; font-size: 0.85rem; font-weight: 600;">Sentiment-Price Dislocation</span>
                </div>
                <p style="color: #64748B; font-size: 0.75rem; margin: 0;">
                    Correlation between sentiment and returns. Negative = mean reversion opportunity.
                </p>
            </div>
            """, unsafe_allow_html=True)
            spds_fig = viz.create_spds_gauge(spds_result['current_spds'], spds_result['signal'])
            st.plotly_chart(spds_fig, width="stretch")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # History Charts Row
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            spds_history_fig = viz.create_spds_history_chart(spds_result['history'])
            st.plotly_chart(spds_history_fig, width="stretch")
        
        with chart_col2:
            cluster_fig = viz.create_cluster_breakdown_chart(cluster_breakdown)
            st.plotly_chart(cluster_fig, width="stretch")
        
        # Methodology Explanation
        with st.expander("Methodology - How These Signals Work"):
            st.markdown("""
            ### Narrative Fragmentation Index (NFI)
            
            **Formula:** `NFI = 1 - (largest_cluster_size / total_nodes)`
            
            **Interpretation:**
            - **NFI > 0.7 (High):** Multiple conflicting narratives. Expect volatility and uncertainty.
            - **NFI 0.3-0.7 (Moderate):** Mixed signals. Market is digesting information.
            - **NFI < 0.3 (Low):** Dominant narrative. Trend continuation likely.
            
            ---
            
            ### Sentiment-Price Dislocation Score (SPDS)
            
            **Formula:** `SPDS = Rolling_Correlation(Sentiment, Returns, window=5)`
            
            **Interpretation:**
            - **SPDS > 0.6 (Aligned):** Sentiment and price moving together. Normal market behavior.
            - **SPDS 0.2-0.6 (Neutral):** Weak correlation. Watch for divergence.
            - **SPDS < 0.2 (Diverging):** Sentiment and price decoupling. Potential trade setup forming.
            - **SPDS < -0.3 (Dislocation):** Strong divergence. Mean reversion opportunity.
            
            ---
            
            ### Combined Alpha Score
            
            The system combines NFI and SPDS to generate:
            - **CONTRARIAN Signal:** High fragmentation + negative SPDS = consider mean-reversion plays
            - **TREND Signal:** Low fragmentation + positive SPDS = consider momentum plays
            - **NEUTRAL:** Mixed signals = wait for clarity
            
            *These metrics are novel and not available in existing financial tools.*
            """)
    
    # ══════════════════════════════════════════════════════════════════════════
    # FOOTER
    # ══════════════════════════════════════════════════════════════════════════
    
    st.markdown("""
    <div style="text-align: center; padding: 1.5rem 0; margin-top: 1rem; border-top: 1px solid rgba(100, 116, 139, 0.2);">
        <p style="color: #475569; font-size: 0.7rem;">
            Narrative Constellation Terminal v4.0 | Alpha Signals Edition<br>
            Novel Metrics: NFI (Narrative Fragmentation Index) + SPDS (Sentiment-Price Dislocation Score)<br>
            Data: Google News RSS + Yahoo Finance | NLP: VADER | Graph: NetworkX
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
