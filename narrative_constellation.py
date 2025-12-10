"""
╔══════════════════════════════════════════════════════════════════════════════╗
║            NARRATIVE CONSTELLATION TERMINAL v3.0                             ║
║         Graph-Theoretic Financial Narrative Analysis Platform                ║
╚══════════════════════════════════════════════════════════════════════════════╝

A sophisticated narrative analysis system that visualizes headline relationships
using graph theory, revealing hidden narrative clusters and contagion patterns.

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
                          │  Plotly Render  │
                          │  (Interactive)  │
                          └─────────────────┘

Author: Financial Engineering Team
Version: 3.0.0 Constellation
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
                Graph-Theoretic Analysis v3.0
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
        
        selected_ticker = st.selectbox("Ticker", Config.DEFAULT_TICKERS, label_visibility="collapsed")
        custom = st.text_input("Custom", placeholder="e.g., COIN", label_visibility="collapsed")
        if custom:
            selected_ticker = custom.upper().strip()
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Ingest controls
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.4rem;">
            {Icons.refresh('#00D9FF', 14)}
            <span style="color: #94A3B8; font-size: 0.8rem; font-weight: 500;">Data Controls</span>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Ingest Data", key="ingest"):
            with st.spinner(f"Fetching {selected_ticker}..."):
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
        
        query = st.text_input("Ask the Data", placeholder="Why is the stock down?", label_visibility="collapsed")
        
        if query:
            results = db.search_headlines(query, selected_ticker, limit=3)
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
    
    # Header
    st.markdown(f"""
    <div style="margin-bottom: 1.5rem;">
        <h1 style="color: #F1F5F9; font-size: 1.5rem; margin: 0; font-weight: 700;">
            {selected_ticker} Narrative Analysis
        </h1>
        <p style="color: #64748B; font-size: 0.85rem; margin-top: 0.3rem;">
            Graph-theoretic visualization of narrative clusters and contagion patterns
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Fetch data
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
    
    tab1, tab2 = st.tabs(["Narrative Constellation", "Price Action"])
    
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
        st.plotly_chart(constellation_fig, width='stretch')
    
    with tab2:
        st.markdown(f"""
        <div class="section-header">
            {Icons.candlestick('#00FF88', 18)}
            <span>Price vs Narrative Contagion</span>
        </div>
        """, unsafe_allow_html=True)
        
        price_fig = viz.create_price_chart(price_df, daily_df, selected_ticker)
        st.plotly_chart(price_fig, width='stretch')
    
    # ══════════════════════════════════════════════════════════════════════════
    # FOOTER
    # ══════════════════════════════════════════════════════════════════════════
    
    st.markdown("""
    <div style="text-align: center; padding: 1.5rem 0; margin-top: 1rem; border-top: 1px solid rgba(100, 116, 139, 0.2);">
        <p style="color: #475569; font-size: 0.7rem;">
            Narrative Constellation Terminal v3.0 | Graph-Theoretic Financial Analysis<br>
            Data: Google News RSS + Yahoo Finance | NLP: VADER | Graph: NetworkX
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
