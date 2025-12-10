"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    NARRATIVE MONITOR - Enterprise MVP                        ║
║                   Real-Time Sentiment & Price Correlation                    ║
╚══════════════════════════════════════════════════════════════════════════════╝

A modular, enterprise-grade sentiment monitoring system that correlates
news narrative with price action for institutional-grade insights.

Architecture:
    ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
    │  Data Ingestion │ --> │  SQLite Layer   │ --> │  Analytics Svc  │
    │  (RSS + VADER)  │     │  (Persistence)  │     │  (Correlation)  │
    └─────────────────┘     └─────────────────┘     └─────────────────┘
                                    │
                                    v
                          ┌─────────────────┐
                          │  Presentation   │
                          │  (Streamlit)    │
                          └─────────────────┘

Author: Financial Software Architecture Team
Version: 1.0.0 MVP
"""

import streamlit as st
import yfinance as yf
import feedparser
import sqlite3
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple
import hashlib
import time
import os
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
import logging

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION & CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

class Config:
    """Centralized configuration for easy scaling to env vars / config files."""
    DB_PATH = "narrative_monitor.db"
    RSS_BASE_URL = "https://news.google.com/rss/search?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en"
    DEFAULT_TICKERS = ["AAPL", "NVDA", "TSLA", "MSFT", "GOOGL", "AMZN", "META", "AMD", "TSM", "INTC"]
    SENTIMENT_WINDOW_HOURS = 72
    PRICE_LOOKBACK_DAYS = 30
    REFRESH_INTERVAL_SECONDS = 300  # 5 minutes
    
class SentimentCategory(Enum):
    """Sentiment classification thresholds."""
    VERY_BULLISH = ("Very Bullish", 0.5, "#00FF88")
    BULLISH = ("Bullish", 0.15, "#4ADE80")
    NEUTRAL = ("Neutral", -0.15, "#94A3B8")
    BEARISH = ("Bearish", -0.5, "#F87171")
    VERY_BEARISH = ("Very Bearish", float('-inf'), "#FF3366")
    
    @classmethod
    def classify(cls, score: float) -> Tuple[str, str]:
        """Return (label, color) for a sentiment score."""
        for category in cls:
            if score >= category.value[1]:
                return category.value[0], category.value[2]
        return cls.VERY_BEARISH.value[0], cls.VERY_BEARISH.value[2]

@dataclass
class NewsItem:
    """Data class for news items."""
    ticker: str
    headline: str
    source: str
    published: datetime
    sentiment_score: float
    headline_hash: str

# ══════════════════════════════════════════════════════════════════════════════
# LOGGING SETUP
# ══════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# DATA LAYER - SQLite Database Service
# ══════════════════════════════════════════════════════════════════════════════

class DatabaseService:
    """
    Enterprise-grade SQLite backend for sentiment persistence.
    Implements connection pooling pattern for thread safety.
    """
    
    def __init__(self, db_path: str = Config.DB_PATH):
        self.db_path = db_path
        self._init_schema()
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections with auto-commit/rollback."""
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            conn.close()
    
    def _init_schema(self):
        """Initialize database schema with proper indexes."""
        with self.get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sentiment_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    ticker TEXT NOT NULL,
                    headline TEXT NOT NULL,
                    source TEXT,
                    sentiment_score REAL NOT NULL,
                    headline_hash TEXT UNIQUE NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # Performance indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ticker ON sentiment_history(ticker)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON sentiment_history(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_hash ON sentiment_history(headline_hash)")
            logger.info("Database schema initialized")
    
    def headline_exists(self, headline_hash: str) -> bool:
        """Check for duplicate headlines using hash."""
        with self.get_connection() as conn:
            result = conn.execute(
                "SELECT 1 FROM sentiment_history WHERE headline_hash = ? LIMIT 1",
                (headline_hash,)
            ).fetchone()
            return result is not None
    
    def insert_news_item(self, item: NewsItem) -> bool:
        """Insert a news item, returns False if duplicate."""
        if self.headline_exists(item.headline_hash):
            return False
        
        with self.get_connection() as conn:
            conn.execute("""
                INSERT INTO sentiment_history 
                (timestamp, ticker, headline, source, sentiment_score, headline_hash)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                item.published.isoformat(),
                item.ticker,
                item.headline,
                item.source,
                item.sentiment_score,
                item.headline_hash
            ))
        return True
    
    def get_sentiment_history(
        self, 
        ticker: str, 
        hours: int = Config.SENTIMENT_WINDOW_HOURS
    ) -> pd.DataFrame:
        """Fetch sentiment history for a ticker within time window."""
        cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
        
        with self.get_connection() as conn:
            df = pd.read_sql_query("""
                SELECT timestamp, ticker, headline, source, sentiment_score
                FROM sentiment_history
                WHERE ticker = ? AND timestamp >= ?
                ORDER BY timestamp DESC
            """, conn, params=(ticker, cutoff))
        
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    
    def get_hourly_sentiment(self, ticker: str, hours: int = 72) -> pd.DataFrame:
        """Calculate average hourly sentiment."""
        cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
        
        with self.get_connection() as conn:
            df = pd.read_sql_query("""
                SELECT 
                    strftime('%Y-%m-%d %H:00:00', timestamp) as hour,
                    AVG(sentiment_score) as avg_sentiment,
                    COUNT(*) as news_count
                FROM sentiment_history
                WHERE ticker = ? AND timestamp >= ?
                GROUP BY hour
                ORDER BY hour
            """, conn, params=(ticker, cutoff))
        
        if not df.empty:
            df['hour'] = pd.to_datetime(df['hour'])
        return df
    
    def get_daily_momentum(self, ticker: str, days: int = 7) -> pd.DataFrame:
        """Calculate daily sentiment momentum (day-over-day change)."""
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        
        with self.get_connection() as conn:
            df = pd.read_sql_query("""
                SELECT 
                    DATE(timestamp) as date,
                    AVG(sentiment_score) as avg_sentiment,
                    COUNT(*) as news_count
                FROM sentiment_history
                WHERE ticker = ? AND timestamp >= ?
                GROUP BY DATE(timestamp)
                ORDER BY date
            """, conn, params=(ticker, cutoff))
        
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            df['momentum'] = df['avg_sentiment'].diff()
        return df
    
    def get_total_records(self) -> int:
        """Get total record count for system health."""
        with self.get_connection() as conn:
            result = conn.execute("SELECT COUNT(*) FROM sentiment_history").fetchone()
            return result[0] if result else 0
    
    def get_last_fetch_time(self, ticker: str = None) -> Optional[datetime]:
        """Get the timestamp of the most recent data fetch."""
        query = "SELECT MAX(created_at) FROM sentiment_history"
        params = ()
        if ticker:
            query += " WHERE ticker = ?"
            params = (ticker,)
        
        with self.get_connection() as conn:
            result = conn.execute(query, params).fetchone()
            if result and result[0]:
                return datetime.fromisoformat(result[0])
        return None
    
    def get_all_news(self, limit: int = 100) -> pd.DataFrame:
        """Get recent news across all tickers."""
        with self.get_connection() as conn:
            df = pd.read_sql_query("""
                SELECT timestamp, ticker, headline, source, sentiment_score
                FROM sentiment_history
                ORDER BY timestamp DESC
                LIMIT ?
            """, conn, params=(limit,))
        
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df

# ══════════════════════════════════════════════════════════════════════════════
# NLP ENGINE - Sentiment Analysis Service
# ══════════════════════════════════════════════════════════════════════════════

class SentimentService:
    """
    VADER-based sentiment analysis with financial context enhancement.
    Architecture: Stateless service, can be easily swapped for FinBERT/GPT.
    """
    
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        # Financial lexicon enhancements
        self._enhance_lexicon()
    
    def _enhance_lexicon(self):
        """Add financial-specific terms to VADER lexicon."""
        financial_terms = {
            'bullish': 2.5, 'bearish': -2.5, 'upgrade': 2.0, 'downgrade': -2.0,
            'beat': 1.5, 'miss': -1.5, 'exceed': 1.5, 'disappoint': -1.5,
            'rally': 2.0, 'crash': -2.5, 'surge': 2.0, 'plunge': -2.5,
            'breakout': 1.8, 'breakdown': -1.8, 'momentum': 1.0, 'volatile': -0.5,
            'growth': 1.5, 'decline': -1.5, 'profit': 1.5, 'loss': -1.5,
            'outperform': 2.0, 'underperform': -2.0, 'buy': 1.5, 'sell': -1.5,
            'accumulate': 1.5, 'distribute': -1.0, 'overbought': -0.5, 'oversold': 0.5,
            'record': 1.5, 'bankrupt': -3.0, 'lawsuit': -1.5, 'investigation': -1.5,
            'partnership': 1.0, 'acquisition': 1.0, 'layoff': -1.5, 'restructure': -0.5
        }
        self.analyzer.lexicon.update(financial_terms)
    
    def analyze(self, text: str) -> float:
        """
        Analyze sentiment of text, returns compound score [-1, 1].
        Compound score considers all lexicon ratings and rules.
        """
        if not text or not text.strip():
            return 0.0
        
        scores = self.analyzer.polarity_scores(text)
        return scores['compound']
    
    def analyze_batch(self, texts: List[str]) -> List[float]:
        """Batch sentiment analysis for efficiency."""
        return [self.analyze(text) for text in texts]

# ══════════════════════════════════════════════════════════════════════════════
# INGESTION SERVICE - RSS Feed Parser
# ══════════════════════════════════════════════════════════════════════════════

class IngestionService:
    """
    Google News RSS ingestion service.
    Handles rate limiting, error recovery, and deduplication.
    """
    
    def __init__(self, db: DatabaseService, sentiment: SentimentService):
        self.db = db
        self.sentiment = sentiment
    
    @staticmethod
    def _generate_hash(headline: str, ticker: str) -> str:
        """Generate unique hash for headline deduplication."""
        content = f"{ticker}:{headline.lower().strip()}"
        return hashlib.sha256(content.encode()).hexdigest()[:32]
    
    @staticmethod
    def _parse_publish_date(entry: dict) -> datetime:
        """Parse RSS entry publish date with fallback."""
        if hasattr(entry, 'published_parsed') and entry.published_parsed:
            try:
                return datetime(*entry.published_parsed[:6])
            except:
                pass
        return datetime.now()
    
    @staticmethod
    def _extract_source(entry: dict) -> str:
        """Extract news source from RSS entry."""
        if hasattr(entry, 'source') and hasattr(entry.source, 'title'):
            return entry.source.title
        # Try to extract from title (Google News format: "Headline - Source")
        title = entry.get('title', '')
        if ' - ' in title:
            return title.split(' - ')[-1].strip()
        return "Unknown"
    
    def fetch_news(self, ticker: str) -> Tuple[int, int, str]:
        """
        Fetch and process news for a ticker.
        Returns: (new_count, duplicate_count, status_message)
        """
        url = Config.RSS_BASE_URL.format(ticker=ticker)
        new_count = 0
        dup_count = 0
        
        try:
            feed = feedparser.parse(url)
            
            if feed.bozo and not feed.entries:
                return 0, 0, f"Feed error: {feed.bozo_exception}"
            
            if not feed.entries:
                return 0, 0, "No news found"
            
            for entry in feed.entries:
                headline = entry.get('title', '').strip()
                if not headline:
                    continue
                
                # Remove source suffix from headline if present
                if ' - ' in headline:
                    headline = ' - '.join(headline.split(' - ')[:-1])
                
                headline_hash = self._generate_hash(headline, ticker)
                
                # Skip if exists
                if self.db.headline_exists(headline_hash):
                    dup_count += 1
                    continue
                
                # Analyze and store
                sentiment_score = self.sentiment.analyze(headline)
                source = self._extract_source(entry)
                published = self._parse_publish_date(entry)
                
                news_item = NewsItem(
                    ticker=ticker,
                    headline=headline,
                    source=source,
                    published=published,
                    sentiment_score=sentiment_score,
                    headline_hash=headline_hash
                )
                
                if self.db.insert_news_item(news_item):
                    new_count += 1
            
            return new_count, dup_count, "Success"
            
        except Exception as e:
            logger.error(f"Ingestion error for {ticker}: {e}")
            return 0, 0, f"Error: {str(e)}"
    
    def fetch_multiple(self, tickers: List[str], progress_callback=None) -> Dict[str, Tuple[int, int, str]]:
        """Fetch news for multiple tickers with optional progress callback."""
        results = {}
        for i, ticker in enumerate(tickers):
            results[ticker] = self.fetch_news(ticker)
            if progress_callback:
                progress_callback((i + 1) / len(tickers))
            time.sleep(0.5)  # Rate limiting
        return results

# ══════════════════════════════════════════════════════════════════════════════
# ANALYTICS SERVICE - Correlation & Metrics
# ══════════════════════════════════════════════════════════════════════════════

class AnalyticsService:
    """
    Analytics engine for sentiment-price correlation.
    Provides statistical measures for alpha generation signals.
    """
    
    def __init__(self, db: DatabaseService):
        self.db = db
    
    @staticmethod
    @st.cache_data(ttl=300)
    def fetch_price_data(ticker: str, days: int = Config.PRICE_LOOKBACK_DAYS) -> pd.DataFrame:
        """Fetch OHLCV data from yfinance with caching."""
        try:
            end = datetime.now()
            start = end - timedelta(days=days)
            df = yf.download(ticker, start=start, end=end, progress=False)
            if df.empty:
                return pd.DataFrame()
            df = df.reset_index()
            df.columns = df.columns.get_level_values(0)
            return df
        except Exception as e:
            logger.error(f"Price fetch error for {ticker}: {e}")
            return pd.DataFrame()
    
    def calculate_correlation(self, ticker: str) -> Tuple[Optional[float], pd.DataFrame]:
        """
        Calculate Pearson correlation between daily sentiment and price returns.
        Returns: (correlation_coefficient, merged_dataframe)
        """
        # Get sentiment data
        sentiment_df = self.db.get_daily_momentum(ticker, days=30)
        if sentiment_df.empty or len(sentiment_df) < 3:
            return None, pd.DataFrame()
        
        # Get price data
        price_df = self.fetch_price_data(ticker, days=30)
        if price_df.empty:
            return None, pd.DataFrame()
        
        # Prepare for merge
        price_df['date'] = pd.to_datetime(price_df['Date']).dt.date
        sentiment_df['date'] = sentiment_df['date'].dt.date
        
        # Calculate daily returns
        price_df['returns'] = price_df['Close'].pct_change()
        
        # Merge datasets
        merged = pd.merge(
            price_df[['date', 'Open', 'High', 'Low', 'Close', 'Volume', 'returns']],
            sentiment_df[['date', 'avg_sentiment', 'news_count']],
            on='date',
            how='inner'
        )
        
        if len(merged) < 3:
            return None, merged
        
        # Calculate correlation
        correlation = merged['returns'].corr(merged['avg_sentiment'])
        
        return correlation, merged
    
    def get_sentiment_summary(self, ticker: str) -> Dict:
        """Get comprehensive sentiment summary for a ticker."""
        hourly = self.db.get_hourly_sentiment(ticker, hours=24)
        daily = self.db.get_daily_momentum(ticker, days=7)
        
        summary = {
            'current_sentiment': None,
            'sentiment_24h_avg': None,
            'sentiment_7d_avg': None,
            'momentum': None,
            'news_count_24h': 0,
            'trend': 'Neutral'
        }
        
        if not hourly.empty:
            summary['sentiment_24h_avg'] = hourly['avg_sentiment'].mean()
            summary['news_count_24h'] = hourly['news_count'].sum()
            if len(hourly) >= 2:
                recent = hourly.tail(6)['avg_sentiment'].mean()
                older = hourly.head(6)['avg_sentiment'].mean()
                summary['momentum'] = recent - older
                summary['trend'] = 'Improving' if summary['momentum'] > 0.05 else \
                                   'Declining' if summary['momentum'] < -0.05 else 'Stable'
        
        if not daily.empty:
            summary['sentiment_7d_avg'] = daily['avg_sentiment'].mean()
            summary['current_sentiment'] = daily['avg_sentiment'].iloc[-1] if len(daily) > 0 else None
        
        return summary

# ══════════════════════════════════════════════════════════════════════════════
# PRESENTATION LAYER - Streamlit Dashboard
# ══════════════════════════════════════════════════════════════════════════════

def apply_dark_theme():
    """Apply professional dark theme styling."""
    st.markdown("""
    <style>
        /* Base dark theme */
        .stApp {
            background-color: #0E1117;
        }
        
        /* Remove default Streamlit padding */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        
        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* Custom card styling */
        .metric-card {
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.8), rgba(15, 23, 42, 0.9));
            border: 1px solid rgba(100, 116, 139, 0.2);
            border-radius: 12px;
            padding: 1.5rem;
            margin: 0.5rem 0;
            backdrop-filter: blur(10px);
        }
        
        .metric-label {
            color: #94A3B8;
            font-size: 0.85rem;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.5rem;
        }
        
        .metric-value {
            color: #F1F5F9;
            font-size: 2rem;
            font-weight: 700;
            font-family: 'JetBrains Mono', monospace;
        }
        
        .metric-delta {
            font-size: 0.9rem;
            margin-top: 0.25rem;
        }
        
        .delta-positive { color: #00FF88; }
        .delta-negative { color: #FF3366; }
        .delta-neutral { color: #94A3B8; }
        
        /* Status indicators */
        .status-online {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            color: #00FF88;
            font-weight: 500;
        }
        
        .status-dot {
            width: 8px;
            height: 8px;
            background: #00FF88;
            border-radius: 50%;
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        /* Sentiment badges */
        .sentiment-badge {
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
        }
        
        /* Section headers */
        .section-header {
            color: #F1F5F9;
            font-size: 1.25rem;
            font-weight: 600;
            margin: 1.5rem 0 1rem 0;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid rgba(100, 116, 139, 0.2);
        }
        
        /* Sidebar styling */
        .css-1d391kg {
            background-color: #0F172A;
        }
        
        /* Table styling */
        .dataframe {
            background-color: #1E293B !important;
        }
        
        /* Button styling */
        .stButton > button {
            background: linear-gradient(135deg, #3B82F6, #1D4ED8);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.5rem 1rem;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        
        .stButton > button:hover {
            background: linear-gradient(135deg, #60A5FA, #3B82F6);
            transform: translateY(-1px);
        }
        
        /* Scrollable container */
        .news-feed {
            max-height: 500px;
            overflow-y: auto;
            padding-right: 1rem;
        }
        
        /* News item styling */
        .news-item {
            background: rgba(30, 41, 59, 0.5);
            border-left: 3px solid;
            padding: 1rem;
            margin-bottom: 0.75rem;
            border-radius: 0 8px 8px 0;
        }
        
        .news-headline {
            color: #F1F5F9;
            font-size: 0.95rem;
            margin-bottom: 0.5rem;
        }
        
        .news-meta {
            color: #64748B;
            font-size: 0.8rem;
        }
    </style>
    """, unsafe_allow_html=True)

def render_metric_card(label: str, value: str, delta: str = None, delta_type: str = "neutral"):
    """Render a professional metric card."""
    delta_class = f"delta-{delta_type}"
    delta_html = f'<div class="metric-delta {delta_class}">{delta}</div>' if delta else ''
    
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)

def render_sentiment_badge(score: float) -> str:
    """Render a color-coded sentiment badge."""
    label, color = SentimentCategory.classify(score)
    return f'<span class="sentiment-badge" style="background-color: {color}20; color: {color}; border: 1px solid {color}40;">{label}</span>'

def create_correlation_chart(merged_df: pd.DataFrame, ticker: str) -> go.Figure:
    """Create dual-axis candlestick + sentiment chart."""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.7, 0.3],
        subplot_titles=(f'{ticker} Price Action', 'Sentiment Moving Average')
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=merged_df['date'],
            open=merged_df['Open'],
            high=merged_df['High'],
            low=merged_df['Low'],
            close=merged_df['Close'],
            name='Price',
            increasing_line_color='#00FF88',
            decreasing_line_color='#FF3366'
        ),
        row=1, col=1
    )
    
    # Volume bars
    colors = ['#00FF88' if close >= open else '#FF3366' 
              for close, open in zip(merged_df['Close'], merged_df['Open'])]
    
    # Sentiment line
    fig.add_trace(
        go.Scatter(
            x=merged_df['date'],
            y=merged_df['avg_sentiment'],
            mode='lines+markers',
            name='Sentiment',
            line=dict(color='#3B82F6', width=2),
            marker=dict(size=6),
            fill='tozeroy',
            fillcolor='rgba(59, 130, 246, 0.1)'
        ),
        row=2, col=1
    )
    
    # Add zero line for sentiment
    fig.add_hline(y=0, line_dash="dash", line_color="rgba(148, 163, 184, 0.5)", row=2, col=1)
    
    # Layout
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#0E1117',
        plot_bgcolor='#0E1117',
        height=600,
        margin=dict(l=50, r=50, t=50, b=50),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='rgba(0,0,0,0)'
        ),
        xaxis_rangeslider_visible=False
    )
    
    # Update axes
    fig.update_xaxes(gridcolor='rgba(100, 116, 139, 0.1)')
    fig.update_yaxes(gridcolor='rgba(100, 116, 139, 0.1)')
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Sentiment", row=2, col=1, range=[-1, 1])
    
    return fig

def create_sentiment_gauge(score: float) -> go.Figure:
    """Create a sentiment gauge chart."""
    label, color = SentimentCategory.classify(score)
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Current Sentiment", 'font': {'size': 16, 'color': '#94A3B8'}},
        number={'font': {'size': 36, 'color': color}},
        gauge={
            'axis': {'range': [-1, 1], 'tickcolor': '#64748B'},
            'bar': {'color': color},
            'bgcolor': '#1E293B',
            'borderwidth': 0,
            'steps': [
                {'range': [-1, -0.5], 'color': 'rgba(255, 51, 102, 0.2)'},
                {'range': [-0.5, -0.15], 'color': 'rgba(248, 113, 113, 0.2)'},
                {'range': [-0.15, 0.15], 'color': 'rgba(148, 163, 184, 0.2)'},
                {'range': [0.15, 0.5], 'color': 'rgba(74, 222, 128, 0.2)'},
                {'range': [0.5, 1], 'color': 'rgba(0, 255, 136, 0.2)'}
            ],
            'threshold': {
                'line': {'color': 'white', 'width': 2},
                'thickness': 0.75,
                'value': score
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor='#0E1117',
        plot_bgcolor='#0E1117',
        height=250,
        margin=dict(l=30, r=30, t=50, b=30)
    )
    
    return fig

def main():
    """Main application entry point."""
    st.set_page_config(
        page_title="Narrative Monitor | Enterprise MVP",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    apply_dark_theme()
    
    # Initialize services
    db = DatabaseService()
    sentiment_svc = SentimentService()
    ingestion_svc = IngestionService(db, sentiment_svc)
    analytics_svc = AnalyticsService(db)
    
    # ══════════════════════════════════════════════════════════════════════════
    # SIDEBAR
    # ══════════════════════════════════════════════════════════════════════════
    
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <h1 style="color: #F1F5F9; font-size: 1.5rem; margin: 0;">
                📊 Narrative Monitor
            </h1>
            <p style="color: #64748B; font-size: 0.85rem; margin-top: 0.5rem;">
                Enterprise Sentiment Analytics
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Ticker selection
        st.markdown("### 🎯 Select Ticker")
        selected_ticker = st.selectbox(
            "Choose a stock to analyze",
            Config.DEFAULT_TICKERS,
            label_visibility="collapsed"
        )
        
        st.markdown("### ⚙️ Data Controls")
        
        # Fetch single ticker
        if st.button("🔄 Refresh News", use_container_width=True):
            with st.spinner(f"Fetching news for {selected_ticker}..."):
                new, dup, status = ingestion_svc.fetch_news(selected_ticker)
                if status == "Success":
                    st.success(f"✓ {new} new headlines ({dup} duplicates)")
                else:
                    st.warning(status)
        
        # Fetch all tickers
        if st.button("📥 Fetch All Tickers", use_container_width=True):
            progress = st.progress(0)
            status_text = st.empty()
            
            results = {}
            for i, ticker in enumerate(Config.DEFAULT_TICKERS):
                status_text.text(f"Fetching {ticker}...")
                results[ticker] = ingestion_svc.fetch_news(ticker)
                progress.progress((i + 1) / len(Config.DEFAULT_TICKERS))
                time.sleep(0.3)
            
            total_new = sum(r[0] for r in results.values())
            st.success(f"✓ Fetched {total_new} new headlines")
        
        st.markdown("---")
        
        # System status
        st.markdown("### 💓 System Heartbeat")
        total_records = db.get_total_records()
        last_fetch = db.get_last_fetch_time()
        
        st.markdown(f"""
        <div class="metric-card" style="padding: 1rem;">
            <div class="status-online">
                <span class="status-dot"></span>
                System Online
            </div>
            <div style="margin-top: 1rem; color: #94A3B8; font-size: 0.85rem;">
                <strong>Total Records:</strong> {total_records:,}<br>
                <strong>Last Fetch:</strong> {last_fetch.strftime('%H:%M:%S') if last_fetch else 'Never'}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # ══════════════════════════════════════════════════════════════════════════
    # MAIN CONTENT
    # ══════════════════════════════════════════════════════════════════════════
    
    # Header
    st.markdown(f"""
    <div style="margin-bottom: 2rem;">
        <h1 style="color: #F1F5F9; font-size: 2rem; margin: 0;">
            {selected_ticker} Sentiment Analysis
        </h1>
        <p style="color: #64748B; font-size: 0.95rem;">
            Real-time narrative monitoring with price correlation
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Get analytics data
    summary = analytics_svc.get_sentiment_summary(selected_ticker)
    correlation, merged_df = analytics_svc.calculate_correlation(selected_ticker)
    
    # ══════════════════════════════════════════════════════════════════════════
    # METRICS ROW
    # ══════════════════════════════════════════════════════════════════════════
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        sentiment_val = summary['sentiment_24h_avg']
        if sentiment_val is not None:
            label, color = SentimentCategory.classify(sentiment_val)
            render_metric_card(
                "24H Sentiment",
                f"{sentiment_val:+.3f}",
                label,
                "positive" if sentiment_val > 0.15 else "negative" if sentiment_val < -0.15 else "neutral"
            )
        else:
            render_metric_card("24H Sentiment", "N/A", "No data")
    
    with col2:
        if correlation is not None:
            corr_label = "Strong" if abs(correlation) > 0.5 else "Moderate" if abs(correlation) > 0.3 else "Weak"
            render_metric_card(
                "Price-Sentiment Corr",
                f"{correlation:+.3f}",
                f"{corr_label} correlation",
                "positive" if correlation > 0.3 else "negative" if correlation < -0.3 else "neutral"
            )
        else:
            render_metric_card("Price-Sentiment Corr", "N/A", "Insufficient data")
    
    with col3:
        momentum = summary['momentum']
        if momentum is not None:
            render_metric_card(
                "Sentiment Momentum",
                f"{momentum:+.3f}",
                summary['trend'],
                "positive" if momentum > 0.05 else "negative" if momentum < -0.05 else "neutral"
            )
        else:
            render_metric_card("Sentiment Momentum", "N/A", "No data")
    
    with col4:
        render_metric_card(
            "News Volume (24H)",
            str(summary['news_count_24h']),
            "Headlines analyzed",
            "neutral"
        )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ══════════════════════════════════════════════════════════════════════════
    # CHARTS
    # ══════════════════════════════════════════════════════════════════════════
    
    col_chart, col_gauge = st.columns([3, 1])
    
    with col_chart:
        st.markdown('<div class="section-header">📈 Price vs Sentiment Correlation</div>', unsafe_allow_html=True)
        
        if not merged_df.empty:
            fig = create_correlation_chart(merged_df, selected_ticker)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("📊 Insufficient data for correlation chart. Fetch more news to see the visualization.")
    
    with col_gauge:
        st.markdown('<div class="section-header">🎯 Sentiment Gauge</div>', unsafe_allow_html=True)
        
        if summary['current_sentiment'] is not None:
            gauge_fig = create_sentiment_gauge(summary['current_sentiment'])
            st.plotly_chart(gauge_fig, use_container_width=True)
        else:
            st.info("No sentiment data available")
    
    # ══════════════════════════════════════════════════════════════════════════
    # NEWS FEED
    # ══════════════════════════════════════════════════════════════════════════
    
    st.markdown('<div class="section-header">📰 Latest Headlines</div>', unsafe_allow_html=True)
    
    news_df = db.get_sentiment_history(selected_ticker, hours=72)
    
    if not news_df.empty:
        # Create display dataframe
        display_df = news_df.copy()
        display_df['Time'] = display_df['timestamp'].dt.strftime('%m/%d %H:%M')
        display_df['Sentiment'] = display_df['sentiment_score'].apply(
            lambda x: f"{x:+.3f}"
        )
        display_df['Category'] = display_df['sentiment_score'].apply(
            lambda x: SentimentCategory.classify(x)[0]
        )
        
        # Color code the dataframe
        def color_sentiment(val):
            try:
                score = float(val)
                _, color = SentimentCategory.classify(score)
                return f'color: {color}'
            except:
                return ''
        
        styled_df = display_df[['Time', 'headline', 'source', 'Sentiment', 'Category']].rename(
            columns={'headline': 'Headline', 'source': 'Source'}
        ).head(50)
        
        st.dataframe(
            styled_df.style.applymap(color_sentiment, subset=['Sentiment']),
            use_container_width=True,
            height=400
        )
    else:
        st.info(f"📭 No headlines found for {selected_ticker}. Click 'Refresh News' to fetch the latest.")
    
    # ══════════════════════════════════════════════════════════════════════════
    # FOOTER
    # ══════════════════════════════════════════════════════════════════════════
    
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0; color: #64748B; font-size: 0.8rem;">
        <hr style="border-color: rgba(100, 116, 139, 0.2); margin-bottom: 1rem;">
        Narrative Monitor MVP v1.0.0 | Enterprise Sentiment Analytics<br>
        Data: Google News RSS + Yahoo Finance | NLP: VADER Sentiment
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
