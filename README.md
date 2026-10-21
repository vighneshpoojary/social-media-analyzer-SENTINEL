# NTRO Social Media Analytics

A real-time intelligence data ingestion, queuing, sentiment/emotion inference, trending topic detection, demographic profiling, social network link analysis, bot detection, and interactive dashboard UI for multi-platform social media monitoring (Telegram, Twitter/X, and more).

## Architecture Overview

```
                   +------------------------+
                   | Telegram Channels      |
                   | (telethon listener)    |
                   +-----------+------------+
                               |
                               v
+------------------+     +-----+----------------+     +-------------------+     +------------------+
| Twitter / X      | --> | RabbitMQ Queue       | --> | Queue Worker      | --> | ClickHouse       |
| (twikit scraper) |     | 'social_media_queue' |     | - SHA-256 Hash    |     | - 'social_posts' |
+------------------+     +----------------------+     | - Emotion Scorer  |     |   (with 5-D NLP) |
                                                      |   (Sarcasm, Fear, |     +--------+---------+
                                                      |    Joy, Stance)   |              |
                                                      +-------------------+              |
                                                                                         v
                                                      +-------------------+     +--------+---------+
                                                      | ClickHouse        | <-- | Trend Analyzer   |
                                                      | 'trending_topics' |     | Job (BERTopic &  |
                                                      | (with velocity)   |     |  TF-IDF/N-Gram)  |
                                                      +-------------------+     +--------+---------+
                                                                                         |
                                                                                         v
                                                      +-------------------+     +--------+---------+
                                                      | ClickHouse        | <-- | Demographic      |
                                                      | 'user_demographics|     | Profiler Job     |
                                                      | (Age, Loc, Role)  |     | (NER & Heuristic)|
                                                      +-------------------+     +--------+---------+
                                                                                         |
                                                                                         v
                                                      +-------------------+     +--------+---------+
                                                      | Neo4j Graph DB    | <-- | Network & Bot    |
                                                      | - Users, Posts    |     | Analyzer Job     |
                                                      | - PageRank & KOLs |     | (IsolationForest)|
                                                      +-------------------+     +--------+---------+
                                                                                         |
                                                                                         v
                                                      +-------------------+     +--------+---------+
                                                      | React Dashboard   | <-- | FastAPI Backend  |
                                                      | (Vite + Tailwind  |     | (REST Endpoints  |
                                                      |  + Recharts)      |     |  & Mock Fallback)|
                                                      +-------------------+     +------------------+
```

## Project Structure

```
├── .env.example                       # Environment variables template
├── .gitignore                         # Git ignore rules
├── docker-compose.yml                 # RabbitMQ, ClickHouse & Neo4j container setup
├── requirements.txt                   # Python dependency specifications
├── run_demo.py                        # Master launcher for FastAPI backend & React frontend
├── frontend/                          # React + Vite + Tailwind intelligence dashboard
│   ├── src/
│   │   ├── components/
│   │   │   ├── Header.jsx             # System status, health indicators, navigation
│   │   │   ├── SentimentTab.jsx       # 5-dimensional emotion area charts
│   │   │   ├── NetworkTab.jsx         # Force-directed interactive canvas social graph
│   │   │   ├── TrendsTab.jsx          # Trending topics & demographic distribution charts
│   │   │   └── BotTab.jsx             # Flagged bot accounts & anomaly metrics table
│   │   ├── api.js                     # Axios backend API client
│   │   ├── App.jsx                    # Root dashboard container
│   │   ├── index.css                  # Tailwind styles & dark mode glow theme
│   │   └── main.jsx                   # React DOM entry point
│   ├── tailwind.config.js             # NTRO intelligence color theme
│   └── vite.config.js                 # Dev server configuration with /api proxy
├── src/
│   ├── main.py                        # Master ingestion pipeline runner
│   ├── api/
│   │   ├── __init__.py
│   │   └── server.py                  # FastAPI server with 6 REST intelligence endpoints
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── rabbitmq_producer.py       # RabbitMQ publish utility
│   │   ├── telegram_listener.py       # Telethon Telegram channel listener
│   │   ├── x_scraper.py               # Twikit Twitter/X keyword scraper
│   │   └── queue_worker.py            # Consumer, SHA-256 hasher & ClickHouse batcher
│   ├── nlp/
│   │   ├── __init__.py
│   │   ├── sentiment.py               # Multi-dimensional emotion inference (5 scores)
│   │   ├── topic_modeling.py          # BERTopic & N-Gram trend extraction engine
│   │   ├── demographics.py            # NER location, age bracket & role profiler
│   │   └── bot_detector.py            # Isolation Forest & heuristic bot classifier
│   ├── graph/
│   │   ├── __init__.py
│   │   └── network_analysis.py        # NetworkX PageRank, betweenness & propagation
│   ├── jobs/
│   │   ├── __init__.py
│   │   ├── trend_analyzer.py          # Background trend & velocity analysis job
│   │   ├── demographic_profiler.py    # Background demographic profiling & upsert job
│   │   └── network_and_bot_analyzer.py# Background Neo4j sync & bot detection job
│   └── storage/
│       ├── __init__.py
│       ├── clickhouse_client.py       # ClickHouse storage, schemas, and migrations
│       └── neo4j_client.py            # Neo4j Cypher client for nodes and edge relations
└── tests/
    ├── test_storage.py                # Storage client unit tests
    ├── test_ingestion_pipeline.py     # Ingestion & queue worker unit tests
    ├── test_nlp.py                    # Multi-dimensional emotion unit tests
    ├── test_topic_modeling.py         # Topic modeling and trend analyzer unit tests
    ├── test_demographics.py           # Demographic profiling and storage unit tests
    ├── test_graph_and_bot.py          # Network graph & bot detector unit tests
    └── test_api.py                    # FastAPI endpoints & REST responses unit tests
```

## Quick Start

### 1. Environment Configuration
```bash
cp .env.example .env
```

### 2. Python Virtual Environment & Dependencies
```bash
python -m venv .venv
# On Windows PowerShell:
.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Spin Up Infrastructure Services (Docker)
```bash
docker compose up -d
```
- **RabbitMQ**: AMQP on `localhost:5672`, Management UI on `http://localhost:15672` (User: `guest`, Pass: `guest`)
- **ClickHouse**: HTTP on `http://localhost:8123`, Native protocol on `localhost:9000`
- **Neo4j**: HTTP Browser on `http://localhost:7474`, Bolt protocol on `localhost:7687` (User: `neo4j`, Pass: `ntro_password`)

### 4. Initialize Database Schema
```bash
python -m src.storage.clickhouse_client
```

### 5. Launch Interactive Dashboard & API
Start both the FastAPI backend and React Vite dashboard with a single command:
```bash
python run_demo.py
```
- **Interactive Intelligence Dashboard UI**: `http://localhost:5173`
- **FastAPI REST API**: `http://localhost:8000`
- **Interactive Swagger Docs**: `http://localhost:8000/docs`

### 6. Run Background Analytics Jobs

**Trend Analyzer:**
```bash
python src/jobs/trend_analyzer.py --once
```

**Demographic Profiler:**
```bash
python src/jobs/demographic_profiler.py --once
```

**Network & Bot Analyzer:**
```bash
python src/jobs/network_and_bot_analyzer.py --once --mock-data
```

### 7. Run Full Unit Test Suite
```bash
python -m unittest discover tests
```
