"""
╔══════════════════════════════════════════════════════════════════════════════╗
║               HYBRID NARRATIVE ENGINE - Contagion Analytics                  ║
║                  Real-Time Viral Potential & Price Correlation               ║
╚══════════════════════════════════════════════════════════════════════════════╝

A sophisticated sentiment-contagion analysis system that combines real news
data with proprietary viral scoring algorithms to detect market contagion.

Viral Potential Algorithm:
    VP = (Sentiment Intensity × 100) × Length Factor
    
    Where:
    - Sentiment Intensity = |VADER compound score|
    - Length Factor = 1.5 - (headline_length / 200), clamped [0.8, 1.5]
    
    Rationale: High-conviction, concise narratives propagate faster.

Author: Financial Engineering & UI Architecture Team
Version: 2.0.0 Hybrid Engine
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
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple
import hashlib
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
import time

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

class Config:
    """Centralized configuration."""
    DB_PATH = "narrative_alpha.db"
    RSS_BASE_URL = "https://news.google.com/rss/search?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en"
    DEFAULT_TICKERS = ["AAPL", "NVDA", "TSLA", "MSFT", "GOOGL", "AMZN", "META", "AMD", "TSM", "INTC"]
    LOOKBACK_DAYS = 30
    
class SentimentZone(Enum):
    """Sentiment classification with professional styling."""
    STRONG_BULLISH = ("Strong Bullish", 0.5, "#00FF88", "up")
    BULLISH = ("Bullish", 0.15, "#4ADE80", "up")
    NEUTRAL = ("Neutral", -0.15, "#64748B", "neutral")
    BEARISH = ("Bearish", -0.5, "#F87171", "down")
    STRONG_BEARISH = ("Strong Bearish", float('-inf'), "#FF3366", "down")
    
    @classmethod
    def classify(cls, score: float) -> Tuple[str, str, str]:
        """Return (label, color, direction) for a sentiment score."""
        for zone in cls:
            if score >= zone.value[1]:
                return zone.value[0], zone.value[2], zone.value[3]
        return cls.STRONG_BEARISH.value[0], cls.STRONG_BEARISH.value[2], cls.STRONG_BEARISH.value[3]

@dataclass
class NewsRecord:
    """Data class for news items with viral metrics."""
    ticker: str
    headline: str
    source: str
    published: datetime
    sentiment_score: float
    sentiment_intensity: float
    viral_potential: float
    headline_hash: str

# ══════════════════════════════════════════════════════════════════════════════
# SVG ICON SYSTEM (No Emojis - Professional Terminal Aesthetic)
# ══════════════════════════════════════════════════════════════════════════════

class Icons:
    """Professional SVG icon system."""
    
    @staticmethod
    def arrow_up_right(color: str = "#00FF88", size: int = 16) -> str:
        return f'''<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
            <line x1="7" y1="17" x2="17" y2="7"></line>
            <polyline points="7 7 17 7 17 17"></polyline>
        </svg>'''
    
    @staticmethod
    def arrow_down_right(color: str = "#FF3366", size: int = 16) -> str:
        return f'''<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
            <line x1="7" y1="7" x2="17" y2="17"></line>
            <polyline points="17 7 17 17 7 17"></polyline>
        </svg>'''
    
    @staticmethod
    def minus_circle(color: str = "#64748B", size: int = 16) -> str:
        return f'''<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <circle cx="12" cy="12" r="10"></circle>
            <line x1="8" y1="12" x2="16" y2="12"></line>
        </svg>'''
    
    @staticmethod
    def activity(color: str = "#3B82F6", size: int = 16) -> str:
        return f'''<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline>
        </svg>'''
    
    @staticmethod
    def trending_up(color: str = "#00FF88", size: int = 16) -> str:
        return f'''<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <polyline points="23 6 13.5 15.5 8.5 10.5 1 18"></polyline>
            <polyline points="17 6 23 6 23 12"></polyline>
        </svg>'''
    
    @staticmethod
    def database(color: str = "#8B5CF6", size: int = 16) -> str:
        return f'''<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <ellipse cx="12" cy="5" rx="9" ry="3"></ellipse>
            <path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"></path>
            <path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"></path>
        </svg>'''
    
    @staticmethod
    def zap(color: str = "#FFB800", size: int = 16) -> str:
        return f'''<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon>
        </svg>'''
    
    @staticmethod
    def refresh(color: str = "#00D9FF", size: int = 16) -> str:
        return f'''<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <polyline points="23 4 23 10 17 10"></polyline>
            <polyline points="1 20 1 14 7 14"></polyline>
            <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"></path>
        </svg>'''
    
    @staticmethod
    def get_direction_icon(direction: str, color: str, size: int = 18) -> str:
        """Get appropriate icon based on sentiment direction."""
        if direction == "up":
            return Icons.arrow_up_right(color, size)
        elif direction == "down":
            return Icons.arrow_down_right(color, size)
        else:
            return Icons.minus_circle(color, size)

# ══════════════════════════════════════════════════════════════════════════════
# DATA LAYER - SQLite with Viral Metrics
# ══════════════════════════════════════════════════════════════════════════════

class DatabaseService:
    """SQLite backend with viral potential storage."""
    
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
        """Initialize database schema."""
        with self.get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS narrative_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    ticker TEXT NOT NULL,
                    headline TEXT NOT NULL,
                    source TEXT,
                    sentiment_score REAL NOT NULL,
                    sentiment_intensity REAL NOT NULL,
                    viral_potential REAL NOT NULL,
                    headline_hash TEXT UNIQUE NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ticker ON narrative_data(ticker)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON narrative_data(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_viral ON narrative_data(viral_potential)")
    
    def headline_exists(self, headline_hash: str) -> bool:
        """Check for duplicate headlines."""
        with self.get_connection() as conn:
            result = conn.execute(
                "SELECT 1 FROM narrative_data WHERE headline_hash = ? LIMIT 1",
                (headline_hash,)
            ).fetchone()
            return result is not None
    
    def insert_record(self, record: NewsRecord) -> bool:
        """Insert a news record, returns False if duplicate."""
        if self.headline_exists(record.headline_hash):
            return False
        
        with self.get_connection() as conn:
            conn.execute("""
                INSERT INTO narrative_data 
                (timestamp, ticker, headline, source, sentiment_score, 
                 sentiment_intensity, viral_potential, headline_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.published.isoformat(),
                record.ticker,
                record.headline,
                record.source,
                record.sentiment_score,
                record.sentiment_intensity,
                record.viral_potential,
                record.headline_hash
            ))
        return True
    
    def get_ticker_data(self, ticker: str, days: int = 30) -> pd.DataFrame:
        """Fetch all data for a ticker."""
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        
        with self.get_connection() as conn:
            df = pd.read_sql_query("""
                SELECT timestamp, ticker, headline, source, sentiment_score,
                       sentiment_intensity, viral_potential
                FROM narrative_data
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
                    AVG(sentiment_score) as avg_sentiment,
                    AVG(viral_potential) as avg_viral,
                    MAX(viral_potential) as max_viral,
                    SUM(viral_potential) as total_viral,
                    COUNT(*) as news_count
                FROM narrative_data
                WHERE ticker = ? AND timestamp >= ?
                GROUP BY DATE(timestamp)
                ORDER BY date
            """, conn, params=(ticker, cutoff))
        
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
        return df
    
    def get_stats(self) -> Dict:
        """Get database statistics."""
        with self.get_connection() as conn:
            total = conn.execute("SELECT COUNT(*) FROM narrative_data").fetchone()[0]
            tickers = conn.execute("SELECT COUNT(DISTINCT ticker) FROM narrative_data").fetchone()[0]
            latest = conn.execute("SELECT MAX(created_at) FROM narrative_data").fetchone()[0]
            avg_viral = conn.execute("SELECT AVG(viral_potential) FROM narrative_data").fetchone()[0]
        
        return {
            'total_records': total,
            'unique_tickers': tickers,
            'last_update': datetime.fromisoformat(latest) if latest else None,
            'avg_viral_potential': avg_viral or 0
        }

# ══════════════════════════════════════════════════════════════════════════════
# VIRAL POTENTIAL ALGORITHM
# ══════════════════════════════════════════════════════════════════════════════

class ViralScorer:
    """
    Proprietary Viral Potential scoring algorithm.
    
    Formula: VP = (Sentiment Intensity × 100) × Length Factor
    
    Rationale:
    - High sentiment intensity = strong market conviction
    - Shorter headlines = more shareable, faster propagation
    - Combined = theoretical viral contagion potential
    """
    
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        self._enhance_lexicon()
    
    def _enhance_lexicon(self):
        """Add financial-specific terms."""
        financial_terms = {
            'bullish': 2.5, 'bearish': -2.5, 'upgrade': 2.0, 'downgrade': -2.0,
            'beat': 1.5, 'miss': -1.5, 'surge': 2.0, 'plunge': -2.5,
            'breakout': 1.8, 'crash': -3.0, 'rally': 2.0, 'selloff': -2.0,
            'soar': 2.5, 'tank': -2.5, 'moon': 2.0, 'dump': -2.0,
            'squeeze': 1.5, 'manipulation': -1.5, 'fraud': -3.0, 'sec': -1.0,
            'lawsuit': -1.5, 'bankruptcy': -3.0, 'default': -2.5, 'recall': -1.5,
            'acquisition': 1.5, 'merger': 1.0, 'spinoff': 0.5, 'ipo': 1.0
        }
        self.analyzer.lexicon.update(financial_terms)
    
    def calculate_length_factor(self, headline: str) -> float:
        """
        Calculate brevity bonus.
        Shorter headlines spread faster in social media.
        Factor range: [0.8, 1.5]
        """
        length = len(headline)
        factor = 1.5 - (length / 200)
        return max(0.8, min(1.5, factor))
    
    def score(self, headline: str) -> Tuple[float, float, float]:
        """
        Calculate viral potential score.
        Returns: (sentiment_score, intensity, viral_potential)
        """
        if not headline or not headline.strip():
            return 0.0, 0.0, 0.0
        
        # Get VADER scores
        scores = self.analyzer.polarity_scores(headline)
        sentiment_score = scores['compound']
        
        # Intensity is the absolute conviction level
        intensity = abs(sentiment_score)
        
        # Length factor (brevity bonus)
        length_factor = self.calculate_length_factor(headline)
        
        # Viral Potential formula
        viral_potential = (intensity * 100) * length_factor
        
        return sentiment_score, intensity, viral_potential

# ══════════════════════════════════════════════════════════════════════════════
# INGESTION SERVICE
# ══════════════════════════════════════════════════════════════════════════════

class IngestionService:
    """RSS feed ingestion with viral scoring."""
    
    def __init__(self, db: DatabaseService, scorer: ViralScorer):
        self.db = db
        self.scorer = scorer
    
    @staticmethod
    def _generate_hash(headline: str, ticker: str) -> str:
        """Generate unique hash for deduplication."""
        content = f"{ticker}:{headline.lower().strip()}"
        return hashlib.sha256(content.encode()).hexdigest()[:32]
    
    @staticmethod
    def _parse_date(entry: dict) -> datetime:
        """Parse RSS entry publish date."""
        if hasattr(entry, 'published_parsed') and entry.published_parsed:
            try:
                return datetime(*entry.published_parsed[:6])
            except:
                pass
        return datetime.now()
    
    @staticmethod
    def _extract_source(entry: dict) -> str:
        """Extract news source."""
        if hasattr(entry, 'source') and hasattr(entry.source, 'title'):
            return entry.source.title
        title = entry.get('title', '')
        if ' - ' in title:
            return title.split(' - ')[-1].strip()
        return "Unknown"
    
    def ingest(self, ticker: str) -> Tuple[int, int, str]:
        """
        Ingest news for a ticker.
        Returns: (new_count, duplicate_count, status)
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
                
                # Clean headline (remove source suffix)
                if ' - ' in headline:
                    headline = ' - '.join(headline.split(' - ')[:-1])
                
                headline_hash = self._generate_hash(headline, ticker)
                
                if self.db.headline_exists(headline_hash):
                    dup_count += 1
                    continue
                
                # Score the headline
                sentiment, intensity, viral = self.scorer.score(headline)
                source = self._extract_source(entry)
                published = self._parse_date(entry)
                
                record = NewsRecord(
                    ticker=ticker,
                    headline=headline,
                    source=source,
                    published=published,
                    sentiment_score=sentiment,
                    sentiment_intensity=intensity,
                    viral_potential=viral,
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

class ChartEngine:
    """Plotly visualization engine for contagion analytics."""
    
    @staticmethod
    @st.cache_data(ttl=300)
    def fetch_price_data(ticker: str, days: int = 30) -> pd.DataFrame:
        """Fetch OHLCV data from yfinance."""
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
            return pd.DataFrame()
    
    @staticmethod
    def create_contagion_chart(price_df: pd.DataFrame, viral_df: pd.DataFrame, ticker: str) -> go.Figure:
        """
        Create the Contagion Graph - Dual axis with price candlestick and viral potential bars.
        """
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            row_heights=[0.65, 0.35],
            specs=[[{"secondary_y": True}], [{"secondary_y": False}]]
        )
        
        # Candlestick chart
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
        
        # Viral Potential bars on secondary axis
        if not viral_df.empty:
            # Color bars based on sentiment
            colors = ['#00FF88' if s > 0.15 else '#FF3366' if s < -0.15 else '#64748B' 
                     for s in viral_df['avg_sentiment']]
            
            fig.add_trace(
                go.Bar(
                    x=viral_df['date'],
                    y=viral_df['total_viral'],
                    name='Viral Potential',
                    marker_color=colors,
                    opacity=0.6
                ),
                row=1, col=1, secondary_y=True
            )
        
        # Sentiment line in bottom panel
        if not viral_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=viral_df['date'],
                    y=viral_df['avg_sentiment'],
                    mode='lines+markers',
                    name='Avg Sentiment',
                    line=dict(color='#3B82F6', width=2),
                    marker=dict(size=6),
                    fill='tozeroy',
                    fillcolor='rgba(59, 130, 246, 0.15)'
                ),
                row=2, col=1
            )
            
            # Zero line
            fig.add_hline(y=0, line_dash="dash", line_color="rgba(148, 163, 184, 0.4)", row=2, col=1)
        
        # Layout
        fig.update_layout(
            title=dict(
                text=f'{ticker} Price Action vs Narrative Contagion',
                font=dict(size=18, color='#F1F5F9'),
                x=0.5
            ),
            template='plotly_dark',
            paper_bgcolor='#0E1117',
            plot_bgcolor='rgba(15, 23, 42, 0.8)',
            height=550,
            margin=dict(l=60, r=60, t=60, b=40),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                bgcolor='rgba(0,0,0,0)',
                font=dict(color='#94A3B8')
            ),
            xaxis_rangeslider_visible=False
        )
        
        # Update axes
        fig.update_xaxes(gridcolor='rgba(100, 116, 139, 0.15)', showgrid=True)
        fig.update_yaxes(gridcolor='rgba(100, 116, 139, 0.15)', showgrid=True)
        fig.update_yaxes(title_text="Price ($)", row=1, col=1, secondary_y=False, title_font=dict(color='#94A3B8'))
        fig.update_yaxes(title_text="Viral Potential", row=1, col=1, secondary_y=True, title_font=dict(color='#94A3B8'))
        fig.update_yaxes(title_text="Sentiment", row=2, col=1, range=[-1, 1], title_font=dict(color='#94A3B8'))
        
        return fig
    
    @staticmethod
    def create_cluster_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
        """
        Create Fear vs Greed cluster scatter plot.
        X-axis: Sentiment Score
        Y-axis: Viral Potential
        """
        if df.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="No Data Available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color='#64748B')
            )
            fig.update_layout(
                template='plotly_dark',
                paper_bgcolor='#0E1117',
                plot_bgcolor='rgba(15, 23, 42, 0.8)',
                height=400
            )
            return fig
        
        # Classify into zones
        df = df.copy()
        df['zone'] = df['sentiment_score'].apply(
            lambda x: 'Greed' if x > 0.15 else 'Fear' if x < -0.15 else 'Neutral'
        )
        df['color'] = df['sentiment_score'].apply(
            lambda x: '#00FF88' if x > 0.15 else '#FF3366' if x < -0.15 else '#64748B'
        )
        
        fig = go.Figure()
        
        # Add scatter points
        for zone, color, symbol in [('Greed', '#00FF88', 'triangle-up'), 
                                     ('Fear', '#FF3366', 'triangle-down'), 
                                     ('Neutral', '#64748B', 'circle')]:
            zone_df = df[df['zone'] == zone]
            if not zone_df.empty:
                fig.add_trace(go.Scatter(
                    x=zone_df['sentiment_score'],
                    y=zone_df['viral_potential'],
                    mode='markers',
                    name=zone,
                    marker=dict(
                        size=10,
                        color=color,
                        symbol=symbol,
                        opacity=0.7,
                        line=dict(width=1, color='white')
                    ),
                    text=zone_df['headline'].str[:50] + '...',
                    hovertemplate='<b>%{text}</b><br>Sentiment: %{x:.3f}<br>Viral: %{y:.1f}<extra></extra>'
                ))
        
        # Add quadrant lines
        fig.add_vline(x=0, line_dash="dash", line_color="rgba(148, 163, 184, 0.4)")
        fig.add_hline(y=df['viral_potential'].median(), line_dash="dash", line_color="rgba(148, 163, 184, 0.4)")
        
        # Add zone annotations
        fig.add_annotation(x=0.7, y=df['viral_potential'].max() * 0.9, text="HIGH GREED<br>HIGH VIRAL",
                          showarrow=False, font=dict(color='#00FF88', size=10), opacity=0.7)
        fig.add_annotation(x=-0.7, y=df['viral_potential'].max() * 0.9, text="HIGH FEAR<br>HIGH VIRAL",
                          showarrow=False, font=dict(color='#FF3366', size=10), opacity=0.7)
        
        fig.update_layout(
            title=dict(
                text='Sentiment vs Viral Potential Clusters',
                font=dict(size=16, color='#F1F5F9'),
                x=0.5
            ),
            template='plotly_dark',
            paper_bgcolor='#0E1117',
            plot_bgcolor='rgba(15, 23, 42, 0.8)',
            height=400,
            margin=dict(l=60, r=40, t=60, b=40),
            xaxis=dict(
                title='Sentiment Score',
                range=[-1.1, 1.1],
                gridcolor='rgba(100, 116, 139, 0.15)',
                title_font=dict(color='#94A3B8')
            ),
            yaxis=dict(
                title='Viral Potential',
                gridcolor='rgba(100, 116, 139, 0.15)',
                title_font=dict(color='#94A3B8')
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                bgcolor='rgba(0,0,0,0)',
                font=dict(color='#94A3B8')
            )
        )
        
        return fig

# ══════════════════════════════════════════════════════════════════════════════
# PRESENTATION LAYER - Terminal UI
# ══════════════════════════════════════════════════════════════════════════════

def apply_terminal_theme():
    """Apply professional terminal theme with glassmorphism."""
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');
        
        /* Base styling */
        .stApp {
            background: linear-gradient(135deg, #0E1117 0%, #1A1F2E 50%, #0E1117 100%);
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }
        
        /* Hide Streamlit elements */
        #MainMenu, footer, header, .stDeployButton {visibility: hidden;}
        
        /* Glassmorphism card */
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
        
        /* Metric styling */
        .metric-container {
            display: flex;
            flex-direction: column;
            gap: 0.25rem;
        }
        
        .metric-label {
            color: #64748B;
            font-size: 0.75rem;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.1em;
        }
        
        .metric-value {
            color: #F1F5F9;
            font-size: 1.75rem;
            font-weight: 700;
            font-family: 'JetBrains Mono', monospace;
            line-height: 1.2;
        }
        
        .metric-delta {
            font-size: 0.85rem;
            font-weight: 500;
        }
        
        /* News card styling */
        .news-card {
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.5), rgba(15, 23, 42, 0.6));
            backdrop-filter: blur(10px);
            border-radius: 12px;
            padding: 1rem 1.25rem;
            margin-bottom: 0.75rem;
            border-left: 3px solid;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        
        .news-card:hover {
            transform: translateX(4px);
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        }
        
        .news-header {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            margin-bottom: 0.5rem;
        }
        
        .news-headline {
            color: #E2E8F0;
            font-size: 0.95rem;
            font-weight: 500;
            line-height: 1.4;
        }
        
        .news-meta {
            display: flex;
            align-items: center;
            gap: 1rem;
            color: #64748B;
            font-size: 0.8rem;
            margin-top: 0.5rem;
        }
        
        /* Intensity meter */
        .intensity-meter {
            display: flex;
            gap: 3px;
            margin-top: 0.75rem;
        }
        
        .intensity-block {
            width: 20px;
            height: 6px;
            border-radius: 2px;
            background: rgba(100, 116, 139, 0.3);
        }
        
        .intensity-block.active {
            background: currentColor;
            box-shadow: 0 0 8px currentColor;
        }
        
        /* Section header */
        .section-header {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            color: #F1F5F9;
            font-size: 1.1rem;
            font-weight: 600;
            margin: 1.5rem 0 1rem 0;
            padding-bottom: 0.75rem;
            border-bottom: 1px solid rgba(100, 116, 139, 0.2);
        }
        
        /* Status indicator */
        .status-indicator {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.5rem 1rem;
            background: rgba(0, 255, 136, 0.1);
            border: 1px solid rgba(0, 255, 136, 0.3);
            border-radius: 20px;
            color: #00FF88;
            font-size: 0.8rem;
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
            0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(0, 255, 136, 0.7); }
            50% { opacity: 0.8; box-shadow: 0 0 0 4px rgba(0, 255, 136, 0); }
        }
        
        /* Sidebar styling */
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
            padding: 0.75rem 1rem;
            font-weight: 600;
            font-size: 0.9rem;
            transition: all 0.3s ease;
            margin-top: 0.5rem;
        }
        
        section[data-testid="stSidebar"] .stButton > button:hover {
            background: linear-gradient(135deg, #60A5FA, #3B82F6);
            transform: translateY(-2px);
            box-shadow: 0 4px 15px rgba(59, 130, 246, 0.4);
        }
        
        /* Scrollbar */
        ::-webkit-scrollbar {
            width: 6px;
            height: 6px;
        }
        
        ::-webkit-scrollbar-track {
            background: rgba(15, 23, 42, 0.5);
        }
        
        ::-webkit-scrollbar-thumb {
            background: rgba(100, 116, 139, 0.5);
            border-radius: 3px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: rgba(100, 116, 139, 0.7);
        }
        
        /* DataFrame styling */
        .stDataFrame {
            border-radius: 12px;
            overflow: hidden;
        }
    </style>
    """, unsafe_allow_html=True)

def render_metric_card(label: str, value: str, icon_html: str, color: str = "#F1F5F9"):
    """Render a professional metric card."""
    st.markdown(f"""
    <div class="glass-card">
        <div class="metric-container">
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                {icon_html}
                <span class="metric-label">{label}</span>
            </div>
            <div class="metric-value" style="color: {color};">{value}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_intensity_meter(viral_score: float, max_score: float = 100) -> str:
    """Generate intensity meter HTML."""
    blocks = 10
    filled = min(blocks, int((viral_score / max_score) * blocks))
    
    # Determine color based on score
    if viral_score > 70:
        color = "#FF3366"
    elif viral_score > 40:
        color = "#FFB800"
    else:
        color = "#00FF88"
    
    meter_html = '<div class="intensity-meter">'
    for i in range(blocks):
        active = "active" if i < filled else ""
        meter_html += f'<div class="intensity-block {active}" style="color: {color};"></div>'
    meter_html += '</div>'
    
    return meter_html

def render_news_card(headline: str, source: str, timestamp: datetime, 
                     sentiment: float, viral: float) -> str:
    """Render a glassmorphism news card with intensity meter."""
    label, color, direction = SentimentZone.classify(sentiment)
    icon = Icons.get_direction_icon(direction, color, 18)
    intensity_meter = render_intensity_meter(viral)
    
    time_str = timestamp.strftime('%m/%d %H:%M') if isinstance(timestamp, datetime) else str(timestamp)[:16]
    
    return f"""
    <div class="news-card" style="border-left-color: {color};">
        <div class="news-header">
            {icon}
            <span style="color: {color}; font-size: 0.75rem; font-weight: 600; text-transform: uppercase;">{label}</span>
            <span style="color: #64748B; font-size: 0.75rem; margin-left: auto;">{sentiment:+.3f}</span>
        </div>
        <div class="news-headline">{headline}</div>
        <div class="news-meta">
            <span>{source}</span>
            <span>{time_str}</span>
            <span style="margin-left: auto; color: #FFB800; font-family: 'JetBrains Mono', monospace;">
                VP: {viral:.1f}
            </span>
        </div>
        {intensity_meter}
    </div>
    """

def main():
    """Main application entry point."""
    st.set_page_config(
        page_title="Hybrid Narrative Engine",
        page_icon="chart_with_upwards_trend",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    apply_terminal_theme()
    
    # Initialize services
    db = DatabaseService()
    scorer = ViralScorer()
    ingestion = IngestionService(db, scorer)
    charts = ChartEngine()
    
    # ══════════════════════════════════════════════════════════════════════════
    # SIDEBAR
    # ══════════════════════════════════════════════════════════════════════════
    
    with st.sidebar:
        st.markdown(f"""
        <div style="text-align: center; padding: 1.5rem 0;">
            <div style="margin-bottom: 0.75rem;">{Icons.activity('#00D9FF', 32)}</div>
            <h1 style="color: #F1F5F9; font-size: 1.25rem; margin: 0; font-weight: 700;">
                Hybrid Narrative Engine
            </h1>
            <p style="color: #64748B; font-size: 0.8rem; margin-top: 0.5rem;">
                Contagion Analytics Platform
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Ticker input
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
            {Icons.trending_up('#00FF88', 16)}
            <span style="color: #94A3B8; font-size: 0.85rem; font-weight: 500;">Select Ticker</span>
        </div>
        """, unsafe_allow_html=True)
        
        selected_ticker = st.selectbox(
            "Ticker",
            Config.DEFAULT_TICKERS,
            label_visibility="collapsed"
        )
        
        # Custom ticker input
        custom_ticker = st.text_input(
            "Or enter custom ticker",
            placeholder="e.g., COIN",
            label_visibility="collapsed"
        )
        
        if custom_ticker:
            selected_ticker = custom_ticker.upper().strip()
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Ingest button
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
            {Icons.refresh('#00D9FF', 16)}
            <span style="color: #94A3B8; font-size: 0.85rem; font-weight: 500;">Data Controls</span>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Ingest Live Data", key="ingest_single"):
            with st.spinner(f"Fetching {selected_ticker}..."):
                new, dup, status = ingestion.ingest(selected_ticker)
                if status == "Success":
                    st.success(f"{new} new headlines ingested")
                else:
                    st.warning(status)
        
        if st.button("Ingest All Tickers", key="ingest_all"):
            progress = st.progress(0)
            total_new = 0
            for i, ticker in enumerate(Config.DEFAULT_TICKERS):
                new, _, _ = ingestion.ingest(ticker)
                total_new += new
                progress.progress((i + 1) / len(Config.DEFAULT_TICKERS))
                time.sleep(0.2)
            st.success(f"{total_new} total headlines ingested")
        
        st.markdown("---")
        
        # System status
        stats = db.get_stats()
        
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 1rem;">
            {Icons.database('#8B5CF6', 16)}
            <span style="color: #94A3B8; font-size: 0.85rem; font-weight: 500;">System Status</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="glass-card-sm">
            <div class="status-indicator">
                <span class="status-dot"></span>
                Online
            </div>
            <div style="margin-top: 1rem; color: #94A3B8; font-size: 0.8rem; line-height: 1.8;">
                <div style="display: flex; justify-content: space-between;">
                    <span>Total Records</span>
                    <span style="color: #F1F5F9; font-family: 'JetBrains Mono', monospace;">{stats['total_records']:,}</span>
                </div>
                <div style="display: flex; justify-content: space-between;">
                    <span>Unique Tickers</span>
                    <span style="color: #F1F5F9; font-family: 'JetBrains Mono', monospace;">{stats['unique_tickers']}</span>
                </div>
                <div style="display: flex; justify-content: space-between;">
                    <span>Avg Viral Score</span>
                    <span style="color: #FFB800; font-family: 'JetBrains Mono', monospace;">{stats['avg_viral_potential']:.1f}</span>
                </div>
                <div style="display: flex; justify-content: space-between;">
                    <span>Last Update</span>
                    <span style="color: #F1F5F9; font-size: 0.75rem;">
                        {stats['last_update'].strftime('%H:%M:%S') if stats['last_update'] else 'Never'}
                    </span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # ══════════════════════════════════════════════════════════════════════════
    # MAIN CONTENT
    # ══════════════════════════════════════════════════════════════════════════
    
    # Header
    st.markdown(f"""
    <div style="margin-bottom: 2rem;">
        <h1 style="color: #F1F5F9; font-size: 1.75rem; margin: 0; font-weight: 700;">
            {selected_ticker} Narrative Contagion Analysis
        </h1>
        <p style="color: #64748B; font-size: 0.9rem; margin-top: 0.5rem;">
            Real-time sentiment propagation with theoretical viral potential scoring
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Fetch data
    ticker_data = db.get_ticker_data(selected_ticker)
    daily_data = db.get_daily_aggregates(selected_ticker)
    price_data = charts.fetch_price_data(selected_ticker)
    
    # ══════════════════════════════════════════════════════════════════════════
    # METRICS ROW
    # ══════════════════════════════════════════════════════════════════════════
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if not ticker_data.empty:
            avg_sentiment = ticker_data['sentiment_score'].mean()
            _, color, _ = SentimentZone.classify(avg_sentiment)
            render_metric_card(
                "Avg Sentiment",
                f"{avg_sentiment:+.3f}",
                Icons.activity(color, 18),
                color
            )
        else:
            render_metric_card("Avg Sentiment", "N/A", Icons.activity('#64748B', 18), '#64748B')
    
    with col2:
        if not ticker_data.empty:
            avg_viral = ticker_data['viral_potential'].mean()
            render_metric_card(
                "Avg Viral Potential",
                f"{avg_viral:.1f}",
                Icons.zap('#FFB800', 18),
                '#FFB800'
            )
        else:
            render_metric_card("Avg Viral Potential", "N/A", Icons.zap('#64748B', 18), '#64748B')
    
    with col3:
        if not ticker_data.empty:
            max_viral = ticker_data['viral_potential'].max()
            render_metric_card(
                "Peak Viral Score",
                f"{max_viral:.1f}",
                Icons.trending_up('#FF3366', 18),
                '#FF3366'
            )
        else:
            render_metric_card("Peak Viral Score", "N/A", Icons.trending_up('#64748B', 18), '#64748B')
    
    with col4:
        news_count = len(ticker_data)
        render_metric_card(
            "Headlines Tracked",
            f"{news_count:,}",
            Icons.database('#8B5CF6', 18),
            '#8B5CF6'
        )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ══════════════════════════════════════════════════════════════════════════
    # CHARTS
    # ══════════════════════════════════════════════════════════════════════════
    
    # Contagion Chart
    st.markdown(f"""
    <div class="section-header">
        {Icons.activity('#3B82F6', 20)}
        <span>Contagion Graph - Price vs Viral Potential</span>
    </div>
    """, unsafe_allow_html=True)
    
    if not price_data.empty:
        contagion_fig = charts.create_contagion_chart(price_data, daily_data, selected_ticker)
        st.plotly_chart(contagion_fig, use_container_width=True)
    else:
        st.markdown("""
        <div class="glass-card" style="text-align: center; padding: 3rem;">
            <p style="color: #64748B; font-size: 1rem;">No price data available for this ticker</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Two columns for cluster chart and news feed
    col_cluster, col_news = st.columns([1, 1])
    
    with col_cluster:
        st.markdown(f"""
        <div class="section-header">
            {Icons.zap('#FFB800', 20)}
            <span>Fear vs Greed Clusters</span>
        </div>
        """, unsafe_allow_html=True)
        
        cluster_fig = charts.create_cluster_chart(ticker_data, selected_ticker)
        st.plotly_chart(cluster_fig, use_container_width=True)
    
    with col_news:
        st.markdown(f"""
        <div class="section-header">
            {Icons.trending_up('#00FF88', 20)}
            <span>Latest Headlines</span>
        </div>
        """, unsafe_allow_html=True)
        
        if not ticker_data.empty:
            news_html = '<div style="max-height: 400px; overflow-y: auto; padding-right: 0.5rem;">'
            for _, row in ticker_data.head(15).iterrows():
                news_html += render_news_card(
                    row['headline'],
                    row['source'],
                    row['timestamp'],
                    row['sentiment_score'],
                    row['viral_potential']
                )
            news_html += '</div>'
            st.markdown(news_html, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="glass-card" style="text-align: center; padding: 3rem;">
                <p style="color: #64748B; font-size: 1rem;">No headlines found</p>
                <p style="color: #475569; font-size: 0.85rem; margin-top: 0.5rem;">
                    Click "Ingest Live Data" in the sidebar to fetch news
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    # ══════════════════════════════════════════════════════════════════════════
    # FOOTER
    # ══════════════════════════════════════════════════════════════════════════
    
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0; margin-top: 2rem; border-top: 1px solid rgba(100, 116, 139, 0.2);">
        <p style="color: #475569; font-size: 0.75rem;">
            Hybrid Narrative Engine v2.0.0 | Contagion Analytics Platform<br>
            Data: Google News RSS + Yahoo Finance | NLP: VADER Sentiment | Viral Algorithm: Proprietary
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
