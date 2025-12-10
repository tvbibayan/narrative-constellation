#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║            NARRATIVE CONSTELLATION v3.0 - FLET NATIVE APP                    ║
║         Graph-Theoretic Financial Narrative Analysis Platform                ║
╚══════════════════════════════════════════════════════════════════════════════╝

Production-grade native desktop application built with Flet (Python + Flutter).
Features candlestick charts, sentiment analysis, network graphs, and query interface.

Author: Senior Python Frontend Developer
Version: 3.0.0
"""

import flet as ft
from flet.matplotlib_chart import MatplotlibChart
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mplfinance.original_flavor import candlestick_ohlc
import hashlib
import sqlite3
import json
import re
from datetime import datetime, timedelta
from collections import Counter
from typing import Optional, List, Dict, Any, Tuple
import threading
import numpy as np

# Third-party imports
try:
    import feedparser
    FEEDPARSER_AVAILABLE = True
except ImportError:
    FEEDPARSER_AVAILABLE = False

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False

try:
    import yfinance as yf
    import pandas as pd
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False


# =============================================================================
# Configuration
# =============================================================================

RSS_FEEDS = {
    "Reuters Business": "https://feeds.reuters.com/reuters/businessNews",
    "Yahoo Finance": "https://finance.yahoo.com/news/rssindex",
    "MarketWatch": "https://feeds.marketwatch.com/marketwatch/topstories/",
    "CNBC Top News": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114",
    "Bloomberg Markets": "https://feeds.bloomberg.com/markets/news.rss",
    "Seeking Alpha": "https://seekingalpha.com/market_currents.xml",
    "Investing.com": "https://www.investing.com/rss/news.rss",
    "Benzinga": "https://www.benzinga.com/feed",
}

GOOGLE_NEWS_URL = "https://news.google.com/rss/search?q={query}+stock&hl=en-US&gl=US&ceid=US:en"

DEFAULT_TICKERS = ["SPY", "QQQ", "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA"]

STOP_WORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
    'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those', 'it',
    'its', "it's", 'they', 'their', 'them', 'we', 'our', 'us', 'you', 'your',
    'he', 'she', 'his', 'her', 'him', 'who', 'which', 'what', 'when', 'where',
    'how', 'why', 'all', 'each', 'every', 'both', 'few', 'more', 'most',
    'other', 'some', 'such', 'no', 'not', 'only', 'same', 'so', 'than',
    'too', 'very', 'just', 'also', 'now', 'here', 'there', 'then', 'once',
    'says', 'said', 'new', 'first', 'last', 'one', 'two', 'three', 'year',
    'years', 'day', 'days', 'week', 'weeks', 'month', 'months', 'time',
    'after', 'before', 'over', 'under', 'between', 'through', 'during',
    'into', 'about', 'against', 'above', 'below', 'up', 'down', 'out',
    'off', 'back', 'still', 'get', 'got', 'reuters', 'ap', 'afp', 'report',
    'reports', 'reported', 'according', 'source', 'sources', 'news', 'update',
    'stock', 'stocks', 'shares', 'share', 'market', 'markets', 'price', 'prices'
}


# =============================================================================
# Database Layer
# =============================================================================

class DatabaseManager:
    """SQLite database manager for headline persistence."""
    
    def __init__(self, db_path: str = "narrative_flet.db"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS headlines (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    hash TEXT UNIQUE NOT NULL,
                    headline TEXT NOT NULL,
                    source TEXT NOT NULL,
                    ticker TEXT,
                    published_at TIMESTAMP,
                    ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    sentiment_compound REAL,
                    sentiment_pos REAL,
                    sentiment_neg REAL,
                    sentiment_neu REAL,
                    viral_score REAL,
                    keywords TEXT
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_published_at ON headlines(published_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_sentiment ON headlines(sentiment_compound)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ticker ON headlines(ticker)")
            conn.commit()
    
    def insert_headline(self, headline: str, source: str, ticker: str, published_at: datetime,
                       sentiment: Dict[str, float], viral_score: float, keywords: List[str]) -> bool:
        """Insert headline if not exists. Returns True if inserted."""
        headline_hash = hashlib.sha256(headline.encode()).hexdigest()
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO headlines (hash, headline, source, ticker, published_at,
                        sentiment_compound, sentiment_pos, sentiment_neg, sentiment_neu, 
                        viral_score, keywords)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (headline_hash, headline, source, ticker, published_at,
                      sentiment['compound'], sentiment['pos'], sentiment['neg'], 
                      sentiment['neu'], viral_score, json.dumps(keywords)))
                return True
        except sqlite3.IntegrityError:
            return False
    
    def get_headlines(self, hours: int = 24, limit: int = 500, ticker: str = None) -> List[Dict]:
        """Fetch headlines from the last N hours."""
        cutoff = datetime.now() - timedelta(hours=hours)
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            if ticker:
                rows = conn.execute("""
                    SELECT * FROM headlines 
                    WHERE ingested_at >= ? AND ticker = ?
                    ORDER BY published_at DESC 
                    LIMIT ?
                """, (cutoff, ticker, limit)).fetchall()
            else:
                rows = conn.execute("""
                    SELECT * FROM headlines 
                    WHERE ingested_at >= ? 
                    ORDER BY published_at DESC 
                    LIMIT ?
                """, (cutoff, limit)).fetchall()
            return [dict(row) for row in rows]
    
    def search_headlines(self, query: str, limit: int = 100) -> List[Dict]:
        """Search headlines by keyword."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT * FROM headlines 
                WHERE headline LIKE ? OR keywords LIKE ?
                ORDER BY published_at DESC 
                LIMIT ?
            """, (f"%{query}%", f"%{query}%", limit)).fetchall()
            return [dict(row) for row in rows]
    
    def get_sentiment_stats(self, ticker: str = None) -> Dict[str, Any]:
        """Get aggregate sentiment statistics."""
        with sqlite3.connect(self.db_path) as conn:
            if ticker:
                row = conn.execute("""
                    SELECT 
                        COUNT(*) as total,
                        AVG(sentiment_compound) as avg_sentiment,
                        AVG(viral_score) as avg_viral,
                        SUM(CASE WHEN sentiment_compound > 0.05 THEN 1 ELSE 0 END) as positive,
                        SUM(CASE WHEN sentiment_compound < -0.05 THEN 1 ELSE 0 END) as negative,
                        SUM(CASE WHEN sentiment_compound BETWEEN -0.05 AND 0.05 THEN 1 ELSE 0 END) as neutral
                    FROM headlines
                    WHERE ingested_at >= datetime('now', '-24 hours') AND ticker = ?
                """, (ticker,)).fetchone()
            else:
                row = conn.execute("""
                    SELECT 
                        COUNT(*) as total,
                        AVG(sentiment_compound) as avg_sentiment,
                        AVG(viral_score) as avg_viral,
                        SUM(CASE WHEN sentiment_compound > 0.05 THEN 1 ELSE 0 END) as positive,
                        SUM(CASE WHEN sentiment_compound < -0.05 THEN 1 ELSE 0 END) as negative,
                        SUM(CASE WHEN sentiment_compound BETWEEN -0.05 AND 0.05 THEN 1 ELSE 0 END) as neutral
                    FROM headlines
                    WHERE ingested_at >= datetime('now', '-24 hours')
                """).fetchone()
            return {
                'total': row[0] or 0,
                'avg_sentiment': row[1] or 0,
                'avg_viral': row[2] or 0,
                'positive': row[3] or 0,
                'negative': row[4] or 0,
                'neutral': row[5] or 0
            }
    
    def get_daily_sentiment(self, ticker: str = None, days: int = 30) -> List[Dict]:
        """Get daily aggregated sentiment."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            if ticker:
                rows = conn.execute("""
                    SELECT 
                        DATE(published_at) as date,
                        AVG(sentiment_compound) as avg_sentiment,
                        SUM(viral_score) as total_viral,
                        COUNT(*) as count
                    FROM headlines
                    WHERE ticker = ? AND published_at >= datetime('now', ?)
                    GROUP BY DATE(published_at)
                    ORDER BY date
                """, (ticker, f'-{days} days')).fetchall()
            else:
                rows = conn.execute("""
                    SELECT 
                        DATE(published_at) as date,
                        AVG(sentiment_compound) as avg_sentiment,
                        SUM(viral_score) as total_viral,
                        COUNT(*) as count
                    FROM headlines
                    WHERE published_at >= datetime('now', ?)
                    GROUP BY DATE(published_at)
                    ORDER BY date
                """, (f'-{days} days',)).fetchall()
            return [dict(row) for row in rows]


# =============================================================================
# Analysis Engine
# =============================================================================

class NarrativeEngine:
    """Core analysis engine for financial narrative processing."""
    
    def __init__(self):
        self.db = DatabaseManager()
        self.analyzer = SentimentIntensityAnalyzer() if VADER_AVAILABLE else None
    
    def extract_keywords(self, text: str, top_n: int = 5) -> List[str]:
        """Extract top keywords from text."""
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        filtered = [w for w in words if w not in STOP_WORDS]
        return [word for word, _ in Counter(filtered).most_common(top_n)]
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment using VADER."""
        if not self.analyzer:
            return {'compound': 0, 'pos': 0, 'neg': 0, 'neu': 1}
        return self.analyzer.polarity_scores(text)
    
    def calculate_viral_score(self, headline: str, sentiment: float) -> float:
        """Calculate viral potential score."""
        length_factor = 100 if len(headline) < 80 else 80 if len(headline) < 120 else 60
        return abs(sentiment) * length_factor
    
    def fetch_ticker_news(self, ticker: str) -> Tuple[int, int]:
        """Fetch news for a specific ticker from Google News."""
        if not FEEDPARSER_AVAILABLE:
            return 0, 0
        
        url = GOOGLE_NEWS_URL.format(query=ticker)
        new_count = 0
        total = 0
        
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:30]:
                total += 1
                headline = entry.get('title', '').strip()
                if not headline:
                    continue
                
                # Parse published date
                published = None
                if hasattr(entry, 'published_parsed') and entry.published_parsed:
                    published = datetime(*entry.published_parsed[:6])
                else:
                    published = datetime.now()
                
                # Get source
                source = entry.get('source', {}).get('title', 'Google News') if hasattr(entry, 'source') else 'Google News'
                
                # Analyze
                sentiment = self.analyze_sentiment(headline)
                keywords = self.extract_keywords(headline)
                viral_score = self.calculate_viral_score(headline, sentiment['compound'])
                
                # Store
                if self.db.insert_headline(headline, source, ticker, published, sentiment, viral_score, keywords):
                    new_count += 1
        except Exception as e:
            print(f"Error fetching {ticker}: {e}")
        
        return new_count, total
    
    def fetch_general_feeds(self, selected_feeds: List[str] = None) -> Tuple[int, int]:
        """Fetch and process general RSS feeds."""
        if not FEEDPARSER_AVAILABLE:
            return 0, 0
        
        feeds_to_process = selected_feeds or list(RSS_FEEDS.keys())
        new_count = 0
        total = 0
        
        for feed_name in feeds_to_process:
            url = RSS_FEEDS.get(feed_name)
            if not url:
                continue
            
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries[:20]:
                    total += 1
                    headline = entry.get('title', '').strip()
                    if not headline:
                        continue
                    
                    published = None
                    if hasattr(entry, 'published_parsed') and entry.published_parsed:
                        published = datetime(*entry.published_parsed[:6])
                    else:
                        published = datetime.now()
                    
                    sentiment = self.analyze_sentiment(headline)
                    keywords = self.extract_keywords(headline)
                    viral_score = self.calculate_viral_score(headline, sentiment['compound'])
                    
                    if self.db.insert_headline(headline, feed_name, "GENERAL", published, sentiment, viral_score, keywords):
                        new_count += 1
            except Exception:
                continue
        
        return new_count, total
    
    def build_network(self, headlines: List[Dict]) -> Optional[Any]:
        """Build NetworkX graph from headlines."""
        if not NETWORKX_AVAILABLE or not headlines:
            return None
        
        G = nx.Graph()
        
        for h in headlines:
            G.add_node(h['id'], 
                      headline=h['headline'][:50],
                      sentiment=h['sentiment_compound'],
                      source=h['source'])
        
        for i, h1 in enumerate(headlines):
            kw1 = set(json.loads(h1['keywords']) if h1['keywords'] else [])
            for h2 in headlines[i+1:]:
                kw2 = set(json.loads(h2['keywords']) if h2['keywords'] else [])
                shared = kw1 & kw2
                if shared:
                    G.add_edge(h1['id'], h2['id'], weight=len(shared), keywords=list(shared))
        
        return G
    
    def get_price_data(self, ticker: str, period: str = "1mo") -> Optional[Any]:
        """Fetch price data using yfinance."""
        if not YFINANCE_AVAILABLE:
            return None
        try:
            df = yf.Ticker(ticker).history(period=period)
            if df.empty:
                return None
            df = df.reset_index()
            return df
        except Exception as e:
            print(f"Error fetching price data: {e}")
            return None


# =============================================================================
# Chart Builders
# =============================================================================

def create_candlestick_chart(price_df, sentiment_data: List[Dict], ticker: str) -> plt.Figure:
    """Create matplotlib candlestick chart with sentiment overlay."""
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), height_ratios=[3, 1], 
                                    gridspec_kw={'hspace': 0.1})
    fig.patch.set_facecolor('#0A0A0A')
    ax1.set_facecolor('#0E1117')
    ax2.set_facecolor('#0E1117')
    
    if price_df is not None and len(price_df) > 0:
        # Prepare OHLC data
        ohlc_data = []
        for i, row in price_df.iterrows():
            date_num = mdates.date2num(row['Date'])
            ohlc_data.append([date_num, row['Open'], row['High'], row['Low'], row['Close']])
        
        # Draw candlesticks
        candlestick_ohlc(ax1, ohlc_data, width=0.6, colorup='#00FF88', colordown='#FF3366',
                        alpha=0.9)
        
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        ax1.xaxis.set_major_locator(mdates.DayLocator(interval=5))
        ax1.set_ylabel('Price ($)', color='#888888', fontsize=10)
        ax1.tick_params(colors='#666666', labelsize=8)
        ax1.grid(True, alpha=0.1, color='#333333')
        ax1.set_title(f'{ticker} Price & Sentiment', color='#FFFFFF', fontsize=12, pad=10)
    else:
        ax1.text(0.5, 0.5, 'No price data available', ha='center', va='center', 
                color='#666666', fontsize=12, transform=ax1.transAxes)
    
    # Sentiment subplot
    if sentiment_data:
        dates = [datetime.strptime(d['date'], '%Y-%m-%d') for d in sentiment_data if d['date']]
        sentiments = [d['avg_sentiment'] for d in sentiment_data if d['date']]
        
        if dates and sentiments:
            colors = ['#00FF88' if s > 0.05 else '#FF3366' if s < -0.05 else '#888888' for s in sentiments]
            ax2.bar(dates, sentiments, color=colors, alpha=0.7, width=0.8)
            ax2.axhline(y=0, color='#444444', linestyle='--', linewidth=0.5)
            ax2.set_ylabel('Sentiment', color='#888888', fontsize=10)
            ax2.set_ylim(-1, 1)
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
            ax2.tick_params(colors='#666666', labelsize=8)
            ax2.grid(True, alpha=0.1, color='#333333')
    else:
        ax2.text(0.5, 0.5, 'No sentiment data', ha='center', va='center',
                color='#666666', fontsize=10, transform=ax2.transAxes)
    
    plt.tight_layout()
    return fig


def create_network_chart(G) -> plt.Figure:
    """Create network visualization chart."""
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#0A0A0A')
    ax.set_facecolor('#0E1117')
    
    if G is None or G.number_of_nodes() == 0:
        ax.text(0.5, 0.5, 'No network data\nFetch headlines first', ha='center', va='center',
               color='#666666', fontsize=14, transform=ax.transAxes)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    else:
        # Layout
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        
        # Node colors based on sentiment
        node_colors = []
        node_sizes = []
        for node in G.nodes():
            sentiment = G.nodes[node].get('sentiment', 0)
            if sentiment > 0.1:
                node_colors.append('#00FF88')
            elif sentiment < -0.1:
                node_colors.append('#FF3366')
            else:
                node_colors.append('#64748B')
            node_sizes.append(100 + abs(sentiment) * 300)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.2, edge_color='#3B82F6', width=0.5)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, 
                              node_size=node_sizes, alpha=0.8)
        
        ax.set_title('Narrative Constellation', color='#FFFFFF', fontsize=12, pad=10)
    
    ax.axis('off')
    plt.tight_layout()
    return fig


def create_sentiment_distribution_chart(headlines: List[Dict]) -> plt.Figure:
    """Create sentiment distribution histogram."""
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor('#0A0A0A')
    ax.set_facecolor('#0E1117')
    
    if headlines:
        sentiments = [h['sentiment_compound'] for h in headlines]
        
        # Create bins and histogram
        bins = np.linspace(-1, 1, 21)
        n, bins, patches = ax.hist(sentiments, bins=bins, edgecolor='none', alpha=0.8)
        
        # Color bars by sentiment
        for i, patch in enumerate(patches):
            bin_center = (bins[i] + bins[i+1]) / 2
            if bin_center > 0.05:
                patch.set_facecolor('#00FF88')
            elif bin_center < -0.05:
                patch.set_facecolor('#FF3366')
            else:
                patch.set_facecolor('#64748B')
        
        ax.axvline(x=0, color='#444444', linestyle='--', linewidth=1)
        ax.set_xlabel('Sentiment Score', color='#888888', fontsize=10)
        ax.set_ylabel('Count', color='#888888', fontsize=10)
        ax.set_title('Sentiment Distribution', color='#FFFFFF', fontsize=12, pad=10)
        ax.tick_params(colors='#666666', labelsize=8)
        ax.grid(True, alpha=0.1, color='#333333', axis='y')
    else:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center',
               color='#666666', fontsize=12, transform=ax.transAxes)
    
    plt.tight_layout()
    return fig


# =============================================================================
# Main Application
# =============================================================================

def main(page: ft.Page):
    """Main Flet application entry point."""
    
    # Page configuration
    page.title = "Narrative Constellation v3.0"
    page.theme_mode = ft.ThemeMode.DARK
    page.bgcolor = "#0A0A0A"
    page.padding = 0
    page.scroll = ft.ScrollMode.AUTO
    
    # Initialize engine
    engine = NarrativeEngine()
    
    # State
    headlines_data = []
    current_ticker = "AAPL"
    current_chart = None
    current_network = None
    current_distribution = None
    
    # =========================================================================
    # UI Components
    # =========================================================================
    
    status_text = ft.Text("Ready", color="#888888", size=12)
    progress_ring = ft.ProgressRing(width=16, height=16, stroke_width=2, visible=False)
    
    # Metrics
    total_metric = ft.Text("0", size=28, color="#00D4FF", weight=ft.FontWeight.BOLD)
    sentiment_metric = ft.Text("0.000", size=28, color="#00FF88", weight=ft.FontWeight.BOLD)
    viral_metric = ft.Text("0.0", size=28, color="#FFB800", weight=ft.FontWeight.BOLD)
    positive_metric = ft.Text("0", size=24, color="#00FF88", weight=ft.FontWeight.BOLD)
    negative_metric = ft.Text("0", size=24, color="#FF4444", weight=ft.FontWeight.BOLD)
    
    # Network stats
    network_nodes = ft.Text("0", size=20, color="#00D4FF", weight=ft.FontWeight.BOLD)
    network_edges = ft.Text("0", size=20, color="#FF6B35", weight=ft.FontWeight.BOLD)
    network_density = ft.Text("0.000", size=20, color="#00FF88", weight=ft.FontWeight.BOLD)
    
    # Charts containers
    price_chart_container = ft.Container(
        content=ft.Text("Select a ticker and fetch data", color="#666666", size=14),
        height=400,
        bgcolor="#1E1E1E",
        border_radius=12,
        border=ft.border.all(1, "#333333"),
        padding=20,
        alignment=ft.alignment.center,
    )
    
    network_chart_container = ft.Container(
        content=ft.Text("Network visualization will appear here", color="#666666", size=14),
        height=400,
        bgcolor="#1E1E1E",
        border_radius=12,
        border=ft.border.all(1, "#333333"),
        padding=20,
        alignment=ft.alignment.center,
    )
    
    distribution_chart_container = ft.Container(
        content=ft.Text("Sentiment distribution will appear here", color="#666666", size=14),
        height=300,
        bgcolor="#1E1E1E",
        border_radius=12,
        border=ft.border.all(1, "#333333"),
        padding=20,
        alignment=ft.alignment.center,
    )
    
    # Headlines list
    headlines_list = ft.ListView(spacing=2, height=400, auto_scroll=False)
    
    # Query results
    query_results_list = ft.ListView(spacing=2, height=300, auto_scroll=False)
    
    # Input fields
    ticker_input = ft.TextField(
        value="AAPL",
        label="Ticker Symbol",
        width=150,
        border_color="#333333",
        focused_border_color="#00D4FF",
        text_size=14,
        label_style=ft.TextStyle(color="#888888"),
    )
    
    query_input = ft.TextField(
        label="Search Headlines",
        hint_text="Enter keyword...",
        width=300,
        border_color="#333333",
        focused_border_color="#00D4FF",
        text_size=14,
        label_style=ft.TextStyle(color="#888888"),
    )
    
    # Ticker dropdown
    ticker_dropdown = ft.Dropdown(
        label="Quick Select",
        width=150,
        options=[ft.dropdown.Option(t) for t in DEFAULT_TICKERS],
        border_color="#333333",
        focused_border_color="#00D4FF",
        text_size=14,
        label_style=ft.TextStyle(color="#888888"),
    )
    
    # =========================================================================
    # Helper Functions
    # =========================================================================
    
    def create_headline_tile(headline: Dict) -> ft.Container:
        """Create a tile for a headline."""
        sentiment = headline.get('sentiment_compound', 0)
        viral = headline.get('viral_score', 0)
        
        if sentiment > 0.05:
            color = "#00FF88"
            icon_name = "trending_up"
        elif sentiment < -0.05:
            color = "#FF4444"
            icon_name = "trending_down"
        else:
            color = "#888888"
            icon_name = "trending_flat"
        
        return ft.Container(
            content=ft.Row([
                ft.Container(
                    content=ft.Column([
                        ft.Icon(name=icon_name, color=color, size=16),
                        ft.Text(f"{sentiment:.2f}", size=10, color=color, weight=ft.FontWeight.BOLD),
                    ], spacing=2, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
                    width=50,
                ),
                ft.Column([
                    ft.Text(
                        headline['headline'][:100] + ("..." if len(headline['headline']) > 100 else ""),
                        size=12, color="#FFFFFF", max_lines=2,
                    ),
                    ft.Row([
                        ft.Container(
                            content=ft.Text(headline.get('ticker', 'N/A'), size=9, color="#00D4FF"),
                            bgcolor="#00D4FF20",
                            padding=ft.padding.symmetric(horizontal=6, vertical=2),
                            border_radius=4,
                        ),
                        ft.Text(headline['source'], size=10, color="#666666"),
                        ft.Text(f"Viral: {viral:.0f}", size=10, color="#FFB800"),
                    ], spacing=8),
                ], spacing=4, expand=True),
            ], spacing=10),
            padding=10,
            bgcolor="#1A1A1A",
            border_radius=8,
            border=ft.border.all(1, "#2A2A2A"),
        )
    
    def update_metrics(ticker: str = None):
        """Update metric displays."""
        stats = engine.db.get_sentiment_stats(ticker)
        total_metric.value = str(stats['total'])
        sentiment_metric.value = f"{stats['avg_sentiment']:.3f}"
        viral_metric.value = f"{stats['avg_viral']:.1f}"
        
        if stats['avg_sentiment'] > 0.05:
            sentiment_metric.color = "#00FF88"
        elif stats['avg_sentiment'] < -0.05:
            sentiment_metric.color = "#FF4444"
        else:
            sentiment_metric.color = "#888888"
        
        positive_metric.value = str(stats['positive'])
        negative_metric.value = str(stats['negative'])
        page.update()
    
    def update_headlines_list(ticker: str = None):
        """Update headlines list."""
        nonlocal headlines_data
        headlines_data = engine.db.get_headlines(hours=168, limit=100, ticker=ticker if ticker != "GENERAL" else None)
        headlines_list.controls.clear()
        for h in headlines_data[:50]:
            headlines_list.controls.append(create_headline_tile(h))
        page.update()
    
    def update_charts(ticker: str):
        """Update all charts."""
        nonlocal current_chart, current_network, current_distribution
        
        # Price chart
        price_df = engine.get_price_data(ticker, "1mo")
        sentiment_data = engine.db.get_daily_sentiment(ticker, 30)
        
        fig = create_candlestick_chart(price_df, sentiment_data, ticker)
        price_chart_container.content = MatplotlibChart(fig, expand=True)
        plt.close(fig)
        
        # Network chart
        G = engine.build_network(headlines_data[:100])
        fig2 = create_network_chart(G)
        network_chart_container.content = MatplotlibChart(fig2, expand=True)
        plt.close(fig2)
        
        if G:
            network_nodes.value = str(G.number_of_nodes())
            network_edges.value = str(G.number_of_edges())
            density = nx.density(G) if G.number_of_nodes() > 1 else 0
            network_density.value = f"{density:.4f}"
        
        # Distribution chart
        fig3 = create_sentiment_distribution_chart(headlines_data)
        distribution_chart_container.content = MatplotlibChart(fig3, expand=True)
        plt.close(fig3)
        
        page.update()
    
    # =========================================================================
    # Event Handlers
    # =========================================================================
    
    def fetch_ticker_data(e):
        """Fetch data for selected ticker."""
        nonlocal current_ticker
        ticker = ticker_input.value.upper().strip()
        if not ticker:
            return
        
        current_ticker = ticker
        progress_ring.visible = True
        status_text.value = f"Fetching {ticker} news..."
        page.update()
        
        def do_fetch():
            new_count, total = engine.fetch_ticker_news(ticker)
            progress_ring.visible = False
            status_text.value = f"Done: {new_count} new headlines for {ticker}"
            update_metrics(ticker)
            update_headlines_list(ticker)
            update_charts(ticker)
        
        threading.Thread(target=do_fetch, daemon=True).start()
    
    def on_ticker_dropdown_change(e):
        """Handle ticker dropdown selection."""
        if ticker_dropdown.value:
            ticker_input.value = ticker_dropdown.value
            page.update()
    
    def search_headlines(e):
        """Search headlines by query."""
        query = query_input.value.strip()
        if not query:
            return
        
        results = engine.db.search_headlines(query, limit=50)
        query_results_list.controls.clear()
        
        if results:
            for h in results:
                query_results_list.controls.append(create_headline_tile(h))
        else:
            query_results_list.controls.append(
                ft.Text("No results found", color="#666666", size=14)
            )
        page.update()
    
    def fetch_general_feeds(e):
        """Fetch from general RSS feeds."""
        progress_ring.visible = True
        status_text.value = "Fetching general feeds..."
        page.update()
        
        def do_fetch():
            new_count, total = engine.fetch_general_feeds()
            progress_ring.visible = False
            status_text.value = f"Done: {new_count} new headlines from {total} processed"
            update_metrics()
            update_headlines_list()
        
        threading.Thread(target=do_fetch, daemon=True).start()
    
    def refresh_all(e):
        """Refresh all displays."""
        update_metrics(current_ticker if current_ticker != "GENERAL" else None)
        update_headlines_list(current_ticker if current_ticker != "GENERAL" else None)
        update_charts(current_ticker)
    
    # =========================================================================
    # Build UI Layout
    # =========================================================================
    
    # Header
    header = ft.Container(
        content=ft.Row([
            ft.Row([
                ft.Icon(name="auto_graph", color="#00D4FF", size=32),
                ft.Column([
                    ft.Row([
                        ft.Text("Narrative Constellation", size=22, weight=ft.FontWeight.BOLD, color="#FFFFFF"),
                        ft.Container(
                            content=ft.Text("v3.0", size=10, color="#00D4FF"),
                            bgcolor="#00D4FF20",
                            padding=ft.padding.symmetric(horizontal=8, vertical=2),
                            border_radius=4,
                        ),
                    ]),
                    ft.Text("Financial Narrative Analysis with NetworkX Graph Theory", size=11, color="#888888"),
                ], spacing=2),
            ]),
            ft.Row([progress_ring, status_text], spacing=10),
        ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
        padding=20,
        bgcolor="#0E1117",
    )
    
    # Control panel
    control_panel = ft.Container(
        content=ft.Row([
            ticker_input,
            ticker_dropdown,
            ft.ElevatedButton("Fetch Ticker", icon="download", on_click=fetch_ticker_data,
                            bgcolor="#00D4FF", color="#000000"),
            ft.Container(width=20),
            ft.ElevatedButton("General Feeds", icon="rss_feed", on_click=fetch_general_feeds,
                            bgcolor="#3B82F6", color="#FFFFFF"),
            ft.OutlinedButton("Refresh", icon="refresh", on_click=refresh_all),
        ], spacing=10),
        padding=15,
        bgcolor="#1E1E1E",
        border_radius=12,
    )
    
    # Metrics row
    metrics_row = ft.Row([
        ft.Container(
            content=ft.Column([
                ft.Text("Total Headlines", size=11, color="#888888"),
                total_metric,
                ft.Text("Last 7 days", size=10, color="#666666"),
            ], spacing=4, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
            padding=15, border_radius=12, bgcolor="#1E1E1E",
            border=ft.border.all(1, "#333333"), expand=True,
        ),
        ft.Container(
            content=ft.Column([
                ft.Text("Avg Sentiment", size=11, color="#888888"),
                sentiment_metric,
                ft.Text("VADER Compound", size=10, color="#666666"),
            ], spacing=4, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
            padding=15, border_radius=12, bgcolor="#1E1E1E",
            border=ft.border.all(1, "#333333"), expand=True,
        ),
        ft.Container(
            content=ft.Column([
                ft.Text("Avg Viral Score", size=11, color="#888888"),
                viral_metric,
                ft.Text("Contagion Index", size=10, color="#666666"),
            ], spacing=4, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
            padding=15, border_radius=12, bgcolor="#1E1E1E",
            border=ft.border.all(1, "#333333"), expand=True,
        ),
        ft.Container(
            content=ft.Column([
                ft.Row([
                    ft.Column([
                        ft.Text("Bullish", size=10, color="#888888"),
                        positive_metric,
                    ], horizontal_alignment=ft.CrossAxisAlignment.CENTER),
                    ft.Container(width=1, height=40, bgcolor="#333333"),
                    ft.Column([
                        ft.Text("Bearish", size=10, color="#888888"),
                        negative_metric,
                    ], horizontal_alignment=ft.CrossAxisAlignment.CENTER),
                ], alignment=ft.MainAxisAlignment.SPACE_AROUND),
            ], spacing=4, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
            padding=15, border_radius=12, bgcolor="#1E1E1E",
            border=ft.border.all(1, "#333333"), expand=True,
        ),
    ], spacing=15)
    
    # Query section
    query_section = ft.Container(
        content=ft.Column([
            ft.Row([
                ft.Icon(name="search", color="#00D4FF", size=20),
                ft.Text("Query Headlines", size=14, weight=ft.FontWeight.W_600, color="#FFFFFF"),
            ]),
            ft.Divider(height=1, color="#333333"),
            ft.Row([
                query_input,
                ft.ElevatedButton("Search", icon="search", on_click=search_headlines,
                                bgcolor="#8B5CF6", color="#FFFFFF"),
            ], spacing=10),
            query_results_list,
        ], spacing=10),
        padding=20,
        border_radius=12,
        bgcolor="#1E1E1E",
        border=ft.border.all(1, "#333333"),
    )
    
    # Network stats panel
    network_panel = ft.Container(
        content=ft.Column([
            ft.Row([
                ft.Icon(name="hub", color="#FF6B35", size=20),
                ft.Text("Network Stats", size=14, weight=ft.FontWeight.W_600, color="#FFFFFF"),
            ]),
            ft.Divider(height=1, color="#333333"),
            ft.Row([
                ft.Column([ft.Text("Nodes", size=10, color="#888888"), network_nodes],
                         horizontal_alignment=ft.CrossAxisAlignment.CENTER, expand=True),
                ft.Column([ft.Text("Edges", size=10, color="#888888"), network_edges],
                         horizontal_alignment=ft.CrossAxisAlignment.CENTER, expand=True),
                ft.Column([ft.Text("Density", size=10, color="#888888"), network_density],
                         horizontal_alignment=ft.CrossAxisAlignment.CENTER, expand=True),
            ]),
        ], spacing=10),
        padding=15,
        border_radius=12,
        bgcolor="#1E1E1E",
        border=ft.border.all(1, "#333333"),
    )
    
    # Charts section with tabs
    charts_tabs = ft.Tabs(
        selected_index=0,
        animation_duration=300,
        tabs=[
            ft.Tab(
                text="Price & Sentiment",
                icon="candlestick_chart",
                content=ft.Container(content=price_chart_container, padding=10),
            ),
            ft.Tab(
                text="Network Graph",
                icon="hub",
                content=ft.Container(content=network_chart_container, padding=10),
            ),
            ft.Tab(
                text="Distribution",
                icon="bar_chart",
                content=ft.Container(content=distribution_chart_container, padding=10),
            ),
        ],
        expand=True,
    )
    
    charts_section = ft.Container(
        content=ft.Column([
            ft.Row([
                ft.Icon(name="insights", color="#00D4FF", size=20),
                ft.Text("Visualizations", size=14, weight=ft.FontWeight.W_600, color="#FFFFFF"),
            ]),
            ft.Divider(height=1, color="#333333"),
            charts_tabs,
        ], spacing=10),
        padding=20,
        border_radius=12,
        bgcolor="#1E1E1E",
        border=ft.border.all(1, "#333333"),
        height=520,
    )
    
    # Headlines section
    headlines_section = ft.Container(
        content=ft.Column([
            ft.Row([
                ft.Icon(name="article", color="#00D4FF", size=20),
                ft.Text("Recent Headlines", size=14, weight=ft.FontWeight.W_600, color="#FFFFFF"),
            ]),
            ft.Divider(height=1, color="#333333"),
            headlines_list,
        ], spacing=10),
        padding=20,
        border_radius=12,
        bgcolor="#1E1E1E",
        border=ft.border.all(1, "#333333"),
    )
    
    # Dependencies footer
    deps = []
    for name, available in [("feedparser", FEEDPARSER_AVAILABLE), ("VADER", VADER_AVAILABLE),
                            ("yfinance", YFINANCE_AVAILABLE), ("NetworkX", NETWORKX_AVAILABLE)]:
        color = "#00FF88" if available else "#FF4444"
        deps.append(ft.Container(ft.Text(name, size=10, color=color),
                                bgcolor=f"{color}20", padding=5, border_radius=4))
    
    footer = ft.Container(
        content=ft.Row([ft.Text("Dependencies:", size=10, color="#666666"), *deps], spacing=8),
        padding=ft.padding.only(top=10, bottom=20),
    )
    
    # =========================================================================
    # Assemble Page
    # =========================================================================
    
    page.add(
        ft.Column([
            header,
            ft.Container(
                content=ft.Column([
                    control_panel,
                    metrics_row,
                    ft.Row([
                        ft.Container(content=charts_section, expand=2),
                        ft.Column([network_panel, query_section], expand=1, spacing=15),
                    ], spacing=15),
                    headlines_section,
                    footer,
                ], spacing=15),
                padding=20,
            ),
        ], spacing=0)
    )
    
    # Initial load
    update_metrics()
    update_headlines_list()


if __name__ == "__main__":
    ft.app(target=main)
