# 🕵️ Shadow Supply Chain Hunter

**A Graph Retrieval-Augmented Generation (GraphRAG) pipeline for discovering hidden supply chain dependencies from SEC 10-K filings.**

---

## 🎯 The Mission

This application extracts supply chain relationships (Entity → Relation → Entity) from unstructured SEC 10-K filings and stores them in a Neo4j Graph Database for complex reasoning and risk analysis.

### What It Does

1. **Ingests** SEC 10-K filings for target companies (AAPL, NVDA, TSM)
2. **Extracts** supply chain triplets: `Supplier → SUPPLIES → Customer`
3. **Stores** entities and relationships in Neo4j
4. **Queries** using hybrid search: Vector similarity + Graph traversal

### Why GraphRAG?

Traditional RAG only retrieves similar text chunks. GraphRAG enables:
- **Multi-hop reasoning**: Apple → TSMC → Taiwan → Geopolitical Risk
- **Hidden dependencies**: Discover risks not explicitly stated in a company's own filings
- **Structural queries**: "Find all companies exposed to Taiwan manufacturing"

---

## 📁 Project Structure

```
/shadow_supply_chain
├── /data                 # Downloaded SEC 10-K filings
├── /src
│   ├── ingest.py         # SEC EDGAR data downloader
│   ├── graph_rag.py      # PropertyGraphIndex builder
│   └── query.py          # Hybrid reasoning engine
├── .env                  # API keys (create from template)
├── requirements.txt      # Python dependencies
└── README.md
```

---

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.10+
- Neo4j AuraDB account (free tier works) or local Docker
- OpenAI API key

### 2. Setup Environment

```bash
# Navigate to project
cd shadow_supply_chain

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Credentials

Edit `.env` with your credentials:

```env
# Neo4j (Aura Cloud)
NEO4J_URI=neo4j+s://xxxxxxxx.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password

# OpenAI
OPENAI_API_KEY=sk-your_key_here

# SEC EDGAR (required - use your real email)
SEC_EDGAR_EMAIL=your_email@example.com
```

### 4. Run the Pipeline

```bash
# Step 1: Download SEC 10-K filings
python src/ingest.py

# Step 2: Build the knowledge graph
python src/graph_rag.py

# Step 3: Query the graph
python src/query.py
```

---

## 🔧 Neo4j Setup Options

### Option A: Neo4j Aura (Recommended)

1. Go to [Neo4j Aura](https://neo4j.com/cloud/aura/)
2. Create a free instance
3. Copy the connection URI and password to `.env`

### Option B: Local Docker

```bash
docker run \
    --name neo4j \
    -p 7474:7474 -p 7687:7687 \
    -e NEO4J_AUTH=neo4j/your_password \
    neo4j:latest
```

Update `.env`:
```env
NEO4J_URI=neo4j://localhost:7687
NEO4J_PASSWORD=your_password
```

---

## 📊 Schema Design

### Entity Types
- `COMPANY` - Corporate entities (Apple, TSMC)
- `SUPPLIER` - Explicit supplier designation
- `CUSTOMER` - Downstream customers
- `PRODUCT` - Products/Components
- `REGION` - Geographic locations
- `RISK_FACTOR` - Risk categories
- `MATERIAL` - Raw materials
- `FACILITY` - Manufacturing facilities

### Relationship Types
- `SUPPLIES` / `SUPPLIES_TO` - Supplier → Customer
- `DEPENDS_ON` - Customer → Supplier dependency
- `MANUFACTURES_IN` - Company → Region
- `EXPOSED_TO` - Entity → Risk Factor
- `SOURCES_FROM` - Company → Region (materials)

---

## 🔍 Query Examples

### Natural Language Queries

```python
from src.query import SupplyChainQueryEngine

engine = SupplyChainQueryEngine()

# Geopolitical risk assessment
engine.query("What are the geopolitical risks for Apple based on its suppliers?")

# Supply chain dependencies
engine.query("Which companies depend on TSMC for semiconductor manufacturing?")

# Risk exposure analysis
engine.query("What supply chain risks does NVIDIA disclose in their 10-K?")
```

### Direct Graph Queries

```python
# Find all suppliers for a company
engine.find_suppliers("Apple")

# Geographic exposure analysis
engine.find_geographic_exposure("Taiwan")

# Multi-hop risk discovery
engine.multi_hop_risk_analysis("Apple", hops=3)

# Path finding
engine.find_supply_chain_path("Apple", "TSMC")
```

---

## 🧠 Architecture Deep Dive

### Why PropertyGraphIndex?

We use LlamaIndex's `PropertyGraphIndex` (not the deprecated `KnowledgeGraphIndex`) because:

1. **Native hybrid retrieval**: Combines vector + graph search
2. **Schema-guided extraction**: Better precision with defined entity/relation types
3. **Property support**: Metadata on nodes and edges (confidence scores, timestamps)
4. **Neo4j integration**: Direct Cypher query support

### Extraction Pipeline

```
10-K Filing Text
      ↓
   Chunking (1024 tokens)
      ↓
   LLM Extraction (GPT-4o)
      ↓
   Triplets: (Subject, Predicate, Object)
      ↓
   Neo4j Property Graph
```

### Hybrid Query Flow

```
User Question
      ↓
   Embedding (text-embedding-3-small)
      ↓
   ┌─────────────────┬─────────────────┐
   │  Vector Search  │  Graph Traverse │
   │  (Similar text) │  (Entity hops)  │
   └─────────────────┴─────────────────┘
      ↓                    ↓
   Text Chunks         Related Entities
      ↓                    ↓
   └─────────┬─────────────┘
             ↓
      Response Synthesis (GPT-4o)
             ↓
         Answer
```

---

## 🔒 Security Notes

- Never commit `.env` to version control
- Use environment variables in production
- SEC requires a valid email for API access (rate limiting)
- Neo4j Aura encrypts data at rest and in transit

---

## 📈 Quant Applications

This pipeline enables several quantitative strategies:

1. **Supply Chain Risk Scoring**: Quantify single-source dependency risk
2. **Contagion Analysis**: Model how disruptions propagate through the supply chain
3. **Geographic Concentration**: Identify portfolio exposure to regional risks
4. **Hidden Correlations**: Discover non-obvious relationships between companies

---

## 🛠️ Troubleshooting

### "No documents found"
Run `python src/ingest.py` first to download the 10-K filings.

### "Neo4j connection failed"
- Check your `NEO4J_URI` format (should start with `neo4j+s://` for Aura)
- Verify credentials in `.env`
- Ensure your IP is whitelisted (Aura security settings)

### "OpenAI rate limit"
- The extraction process makes many API calls
- Consider using `gpt-4o-mini` for development (cheaper)
- Add delays between chunks if needed

### "Empty graph results"
- Check Neo4j Browser to verify data was indexed
- Run the Cypher query: `MATCH (n) RETURN n LIMIT 25`

---

## 📜 License

MIT License - Use at your own risk. This is not financial advice.

---

## 🙏 Acknowledgments

- [LlamaIndex](https://www.llamaindex.ai/) for the GraphRAG framework
- [Neo4j](https://neo4j.com/) for the graph database
- [SEC EDGAR](https://www.sec.gov/edgar) for public filings data
