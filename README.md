# Narrative Constellation Terminal

A comprehensive financial narrative analysis platform combining Graph Retrieval-Augmented Generation (GraphRAG), VADER sentiment analysis, NetworkX graph theory, and real-time market data visualization.

---

## Table of Contents

1. [Overview](#overview)
2. [Applications](#applications)
   - [Narrative Constellation Terminal](#narrative-constellation-terminal-v30)
   - [Narrative Alpha Engine](#narrative-alpha-engine)
   - [Narrative Monitor](#narrative-monitor)
   - [Shadow Supply Chain Hunter](#shadow-supply-chain-hunter-graphrag)
3. [Architecture](#architecture)
4. [Installation](#installation)
5. [Configuration](#configuration)
6. [Usage](#usage)
7. [Technical Specifications](#technical-specifications)
8. [API Reference](#api-reference)
9. [Database Schema](#database-schema)
10. [Troubleshooting](#troubleshooting)
11. [License](#license)

---

## Overview

This platform provides enterprise-grade financial narrative analysis through multiple specialized applications. Each application serves a distinct purpose in the quantitative analysis workflow:

- Real-time RSS feed ingestion from major financial news sources
- VADER sentiment analysis with financial lexicon enhancements
- NetworkX-based headline relationship mapping
- Interactive Plotly visualizations with candlestick charts
- SQLite persistence with SHA-256 deduplication
- Graph-theoretic cluster detection and contagion analysis

The system processes financial headlines to extract sentiment signals, identify narrative clusters, and correlate news patterns with price action.

---

## Applications

### Narrative Constellation Terminal v3.0

**File:** `narrative_constellation.py`

The flagship application providing graph-theoretic financial narrative analysis with an interactive constellation visualization.

#### Core Features

**Network Graph Visualization**
- Headlines represented as nodes in a semantic network
- Edges created based on shared keyword overlap between headlines
- Node color indicates sentiment (green for bullish, red for bearish, gray for neutral)
- Node size represents viral potential score
- Interactive zoom, pan, and hover tooltips

**Sentiment Analysis Pipeline**
- VADER (Valence Aware Dictionary and sEntiment Reasoner) compound scoring
- Enhanced financial lexicon with terms: bullish, bearish, upgrade, downgrade, surge, plunge, breakout, crash, rally, selloff
- Real-time sentiment aggregation by ticker and time period

**Price-Narrative Correlation**
- Dual-axis charts showing price action alongside narrative contagion
- Candlestick OHLCV visualization from Yahoo Finance
- Viral volume overlay indicating headline intensity
- Sentiment trend line with zero-crossing detection

**Tactical Query System**
- Natural language search across headline database
- Results ranked by viral potential score
- Context-aware filtering by ticker symbol
- Real-time query results with sentiment indicators

#### Technical Implementation

```
Architecture:
    RSS Ingestion --> SQLite + NLP --> NetworkX Graph --> Plotly Render
    (Google News)    (VADER + KW)     (Constellation)    (Interactive)
```

**Classes:**
- `Config`: Centralized configuration management
- `DatabaseService`: SQLite backend with keyword storage
- `NLPEngine`: VADER sentiment analysis with financial lexicon
- `ConstellationEngine`: NetworkX graph construction
- `IngestionService`: RSS feed processing pipeline
- `VisualizationEngine`: Plotly chart generation

---

### Narrative Alpha Engine

**File:** `narrative_alpha.py`

A hybrid narrative engine focused on viral score calculation and dual-axis visualization.

#### Core Features

**Viral Score Algorithm**
```
viral_score = abs(sentiment) * brevity_factor
brevity_factor = 100 if headline_length < 80 else 80 if headline_length < 120 else 60
```

**Visual Design**
- Glassmorphism UI with backdrop blur effects
- SVG icon system (no emoji dependencies)
- Dual-axis charts with price and sentiment overlay
- Responsive three-column layout

**Data Pipeline**
- Multi-source RSS ingestion (Reuters, Bloomberg, Yahoo Finance, MarketWatch, CNBC)
- Hourly and daily sentiment aggregation
- Price-sentiment correlation analysis

---

### Narrative Monitor

**File:** `narrative_monitor.py`

Enterprise-grade MVP for continuous narrative monitoring with SQLite persistence.

#### Core Features

**RSS Feed Sources**
- Reuters Business
- Yahoo Finance
- MarketWatch Top Stories
- CNBC Top News
- Bloomberg Markets
- WSJ Markets
- FT Markets
- Seeking Alpha
- Investing.com
- Benzinga

**Database Operations**
- SHA-256 headline deduplication
- Timestamp-based indexing
- Sentiment compound score storage
- Keyword extraction and storage

**Aggregation Functions**
- Hourly sentiment averages
- Daily viral score totals
- Source-based filtering
- Time-range queries

---

### Shadow Supply Chain Hunter (GraphRAG)

**File:** `src/graph_rag.py`, `src/ingest.py`, `src/query.py`

Graph Retrieval-Augmented Generation pipeline for discovering hidden supply chain dependencies from SEC 10-K filings.

#### Core Features

**SEC Filing Processing**
- Automated 10-K filing download from SEC EDGAR
- Text chunking with 1024 token windows
- LLM-based triplet extraction (Subject, Predicate, Object)

**Neo4j Graph Database**
- Entity types: COMPANY, SUPPLIER, CUSTOMER, PRODUCT, REGION, RISK_FACTOR
- Relationship types: SUPPLIES, DEPENDS_ON, MANUFACTURES_IN, EXPOSED_TO
- Property support for confidence scores and timestamps

**Hybrid Query Engine**
- Vector similarity search using text-embedding-3-small
- Graph traversal for multi-hop reasoning
- Response synthesis with GPT-4o

**Query Examples**
```python
engine.query("What are the geopolitical risks for Apple based on its suppliers?")
engine.find_suppliers("Apple")
engine.multi_hop_risk_analysis("Apple", hops=3)
```

---

## Architecture

### System Components

```
+------------------+     +------------------+     +------------------+
|   Data Sources   |     |   Processing     |     |   Visualization  |
+------------------+     +------------------+     +------------------+
| Google News RSS  | --> | VADER Sentiment  | --> | Plotly Charts    |
| Yahoo Finance    |     | Keyword Extract  |     | NetworkX Graphs  |
| SEC EDGAR        |     | SQLite Storage   |     | Streamlit UI     |
| Reuters/Bloomberg|     | Neo4j (GraphRAG) |     | Interactive Tabs |
+------------------+     +------------------+     +------------------+
```

### Data Flow

1. **Ingestion Layer**: RSS feeds parsed via feedparser, SEC filings via EDGAR API
2. **Processing Layer**: VADER sentiment scoring, keyword extraction, deduplication
3. **Storage Layer**: SQLite for headlines, Neo4j for supply chain graphs
4. **Analysis Layer**: NetworkX graph construction, cluster detection, correlation analysis
5. **Presentation Layer**: Streamlit UI with Plotly visualizations

---

## Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager
- Neo4j database (for GraphRAG features)
- OpenAI API key (for GraphRAG features)

### Setup

```bash
# Clone repository
git clone https://github.com/tvbibayan/narrative-constellation.git
cd narrative-constellation

# Create virtual environment
python -m venv .venv311
source .venv311/bin/activate  # On Windows: .venv311\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

```
streamlit>=1.28.0
yfinance>=0.2.31
feedparser>=6.0.10
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.17.0
vaderSentiment>=3.3.2
networkx>=3.1
neo4j>=5.14.0
llama-index>=0.9.0
openai>=1.3.0
python-dotenv>=1.0.0
```

---

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# Neo4j Configuration (for GraphRAG)
NEO4J_URI=neo4j+s://xxxxxxxx.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password

# OpenAI Configuration (for GraphRAG)
OPENAI_API_KEY=sk-your_key_here

# SEC EDGAR Configuration
SEC_EDGAR_EMAIL=your_email@example.com
```

### Application Settings

Located in each application's `Config` class:

```python
class Config:
    DB_PATH = "narrative_constellation.db"
    RSS_BASE_URL = "https://news.google.com/rss/search?q={ticker}+stock"
    DEFAULT_TICKERS = ["AAPL", "NVDA", "TSLA", "MSFT", "GOOGL"]
    LOOKBACK_DAYS = 30
    MIN_KEYWORD_LENGTH = 3
    MIN_SHARED_KEYWORDS = 1
```

---

## Usage

### Running Applications

**Narrative Constellation Terminal**
```bash
streamlit run narrative_constellation.py --server.port 8501
```

**Narrative Alpha Engine**
```bash
streamlit run narrative_alpha.py --server.port 8502
```

**Narrative Monitor**
```bash
streamlit run narrative_monitor.py --server.port 8503
```

**GraphRAG Pipeline**
```bash
# Step 1: Download SEC filings
python src/ingest.py

# Step 2: Build knowledge graph
python src/graph_rag.py

# Step 3: Query the graph
python src/query.py
```

### Workflow

1. Select a ticker symbol from the dropdown or enter a custom ticker
2. Click "Ingest Data" to fetch latest headlines from Google News RSS
3. View the Narrative Constellation graph to identify headline clusters
4. Switch to Price Action tab to correlate sentiment with price movement
5. Use Tactical Query to search for specific topics or events
6. Monitor system status for database statistics

---

## Technical Specifications

### Sentiment Analysis

**VADER Implementation**
- Base lexicon: 7,500+ sentiment-rated words
- Financial enhancements: 20+ domain-specific terms
- Compound score range: -1.0 (most negative) to +1.0 (most positive)
- Threshold classification:
  - Bullish: compound > 0.15
  - Bearish: compound < -0.15
  - Neutral: -0.15 <= compound <= 0.15

**Keyword Extraction**
- Minimum word length: 3 characters
- Stop word filtering: 100+ common English words
- Financial stop words: stock, market, price, report, news, company

### Graph Construction

**Node Attributes**
- `headline`: Truncated headline text (60 characters)
- `full_headline`: Complete headline text
- `sentiment`: VADER compound score
- `viral_score`: Calculated viral potential
- `keywords`: Set of extracted keywords
- `source`: News source name
- `timestamp`: Publication time

**Edge Attributes**
- `weight`: Number of shared keywords
- `shared_keywords`: List of common terms

**Layout Algorithm**
- Spring layout with k=2 repulsion factor
- 50 iterations for convergence
- Seed=42 for reproducibility

### Database Schema

**SQLite Tables**

```sql
CREATE TABLE headlines (
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
);

CREATE INDEX idx_ticker ON headlines(ticker);
CREATE INDEX idx_timestamp ON headlines(timestamp);
CREATE INDEX idx_viral ON headlines(viral_score);
```

---

## API Reference

### DatabaseService

```python
class DatabaseService:
    def __init__(self, db_path: str)
    def headline_exists(self, headline_hash: str) -> bool
    def insert_record(self, record: HeadlineRecord) -> bool
    def get_ticker_headlines(self, ticker: str, days: int = 30) -> pd.DataFrame
    def get_daily_aggregates(self, ticker: str, days: int = 30) -> pd.DataFrame
    def search_headlines(self, query: str, ticker: str = None, limit: int = 10) -> pd.DataFrame
    def get_stats(self) -> Dict
```

### NLPEngine

```python
class NLPEngine:
    def __init__(self)
    def analyze_sentiment(self, text: str) -> float
    def calculate_viral_score(self, headline: str, sentiment: float) -> float
    def extract_keywords(self, headline: str) -> Set[str]
```

### ConstellationEngine

```python
class ConstellationEngine:
    def __init__(self)
    def build_graph(self, df: pd.DataFrame) -> nx.Graph
    def get_graph_stats(self) -> Dict
```

### IngestionService

```python
class IngestionService:
    def __init__(self, db: DatabaseService, nlp: NLPEngine)
    def ingest(self, ticker: str) -> Tuple[int, int, str]
```

### VisualizationEngine

```python
class VisualizationEngine:
    @staticmethod
    def create_constellation_chart(graph: nx.Graph) -> go.Figure
    @staticmethod
    def fetch_price_data(ticker: str, days: int = 30) -> pd.DataFrame
    @staticmethod
    def create_price_chart(price_df: pd.DataFrame, viral_df: pd.DataFrame, ticker: str) -> go.Figure
```

---

## Troubleshooting

### Common Issues

**"No documents found"**
- Ensure you have run the ingestion step first
- Check internet connectivity for RSS feeds
- Verify ticker symbol is valid

**"Empty graph results"**
- Ingest data for the selected ticker
- Check database file exists and has records
- Verify SQLite database is not corrupted

**"Neo4j connection failed"**
- Check NEO4J_URI format (neo4j+s:// for Aura)
- Verify credentials in .env file
- Ensure IP is whitelisted in Aura security settings

**"OpenAI rate limit"**
- Add delays between API calls
- Consider using gpt-4o-mini for development
- Check API key validity and quota

**"RSS feed error"**
- Some feeds may be temporarily unavailable
- Google News RSS may rate limit requests
- Try again after a few minutes

### Performance Optimization

- Use st.cache_data for expensive operations
- Limit headline lookback to 30 days
- Batch database inserts when possible
- Use connection pooling for high-throughput scenarios

---

## Project Structure

```
/narrative-constellation
|-- /data                     # Downloaded SEC 10-K filings
|-- /lib                      # Frontend JavaScript libraries
|   |-- /bindings            # Utility scripts
|   |-- /tom-select          # Select component library
|   |-- /vis-9.1.2           # Network visualization library
|-- /src                      # GraphRAG source code
|   |-- __init__.py
|   |-- ingest.py            # SEC EDGAR downloader
|   |-- graph_rag.py         # PropertyGraphIndex builder
|   |-- query.py             # Hybrid reasoning engine
|-- .env                      # Environment variables (not committed)
|-- .gitignore               # Git ignore rules
|-- app.py                   # Legacy application
|-- narrative_alpha.py       # Hybrid Narrative Engine
|-- narrative_constellation.py  # Main Terminal v3.0
|-- narrative_monitor.py     # Enterprise Monitor MVP
|-- narrative_flet.py        # Flet native app (experimental)
|-- requirements.txt         # Python dependencies
|-- README.md                # This documentation
```

---

## Security Notes

- Never commit .env files to version control
- Use environment variables in production deployments
- SEC EDGAR requires a valid email for API access
- Neo4j Aura encrypts data at rest and in transit
- SQLite databases may contain sensitive financial queries

---

## Quantitative Applications

This platform enables several quantitative trading strategies:

1. **Sentiment Momentum**: Trade based on aggregate sentiment shifts
2. **Narrative Clustering**: Identify coordinated news patterns
3. **Viral Breakout Detection**: Flag high viral score anomalies
4. **Supply Chain Risk Scoring**: Quantify dependency concentration
5. **Contagion Analysis**: Model information propagation through markets
6. **Geographic Concentration**: Assess regional risk exposure

---

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit changes with descriptive messages
4. Push to the branch
5. Open a pull request

---

## License

MIT License - Use at your own risk. This is not financial advice. Past performance does not guarantee future results. Always conduct your own due diligence before making investment decisions.

---

## Acknowledgments

- LlamaIndex for the GraphRAG framework
- Neo4j for the graph database infrastructure
- SEC EDGAR for public filings data
- VADER Sentiment for the sentiment analysis lexicon
- NetworkX for graph algorithms
- Plotly for interactive visualizations
- Streamlit for the web application framework
- Yahoo Finance for market data

---

## Version History

- v3.0.0: Narrative Constellation Terminal with NetworkX graph visualization
- v2.0.0: Narrative Alpha Engine with viral score algorithm
- v1.0.0: Initial Narrative Monitor with VADER sentiment analysis

---

## Contact

Repository: https://github.com/tvbibayan/narrative-constellation

For issues and feature requests, please use the GitHub Issues tracker.
