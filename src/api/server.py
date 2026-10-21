"""NTRO Social Media Analytics — FastAPI Backend Server with DB Connection."""

import hashlib
import os
import random
import sys
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

app = FastAPI(
    title="NTRO Social Media Analytics API Server",
    description="Real-time social media intelligence analytics.",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Database Clients (Graceful Degradation) ──────────────────────────────────

_clickhouse_client = None
_neo4j_client = None

def get_clickhouse():
    global _clickhouse_client
    if _clickhouse_client is not None:
        return _clickhouse_client
    try:
        from src.storage.clickhouse_client import ClickHouseStorageClient
        client = ClickHouseStorageClient()
        if client.check_connection():
            _clickhouse_client = client
            return _clickhouse_client
    except Exception:
        pass
    return None

def get_neo4j():
    global _neo4j_client
    if _neo4j_client is not None:
        return _neo4j_client
    try:
        from src.storage.neo4j_client import Neo4jClient
        client = Neo4jClient()
        if client.connect():
            _neo4j_client = client
            return _neo4j_client
    except Exception:
        pass
    return None


# ─── Mock Data Generators ─────────────────────────────────────────────────────

_PLATFORMS = ["telegram", "x"]
_ROLES      = ["Cybersecurity & Intelligence", "Software & Data Engineering",
                "Defence & Public Sector", "Journalism & Media", "Academic & Research", "Executive & Business"]
_LOCATIONS  = ["New Delhi", "Mumbai", "Bengaluru", "London", "Washington", "Singapore", "Dubai", "Unknown"]
_AGES       = ["18-24", "25-34", "35-54", "55+", "Unknown"]

def _ts(offset_hours: int = 0) -> str:
    return (datetime.now(timezone.utc) - timedelta(hours=offset_hours)).isoformat()


def _mock_stats_summary() -> Dict[str, Any]:
    return {
        "total_posts_ingested": random.randint(12000, 18000),
        "posts_last_hour": random.randint(200, 800),
        "active_platforms": _PLATFORMS,
        "active_topics": random.randint(8, 25),
        "flagged_bots": random.randint(3, 20),
        "kol_count": random.randint(5, 15),
        "services": {
            "clickhouse": get_clickhouse() is not None,
            "rabbitmq": False,
            "neo4j": get_neo4j() is not None,
        },
        "generated_at": _ts(),
    }


def _mock_feed_live(limit: int = 20) -> List[Dict[str, Any]]:
    feeds = []
    topics = ["Border Security Standoff", "Ransomware Alert", "Telecom 5G Infrastructure", "Cyber Disinfo"]
    for i in range(limit):
        uid = hashlib.sha256(f"user_{i}".encode()).hexdigest()[:12]
        feeds.append({
            "timestamp": _ts(i * 0.1),
            "platform": random.choice(_PLATFORMS),
            "user_id": uid,
            "raw_text": f"Crucial update regarding {random.choice(topics)}. Intelligence reports suggest potential anomalies.",
            "post_type": "post",
            "parent_id": None,
            "sarcasm_score": round(random.uniform(0.0, 0.4), 2),
            "anxiety_score": round(random.uniform(0.0, 0.8), 2),
            "excitement_score": round(random.uniform(0.0, 0.6), 2),
            "support_score": round(random.uniform(0.1, 0.9), 2),
            "opposition_score": round(random.uniform(0.0, 0.5), 2),
        })
    return feeds


def _mock_sentiment_timeline(hours: int = 24) -> List[Dict[str, Any]]:
    timeline = []
    for h in range(hours, 0, -1):
        timeline.append({
            "timestamp": _ts(h),
            "sarcasm_score":    round(random.uniform(0.05, 0.40), 3),
            "anxiety_score":    round(random.uniform(0.10, 0.55), 3),
            "excitement_score": round(random.uniform(0.20, 0.70), 3),
            "support_score":    round(random.uniform(0.15, 0.65), 3),
            "opposition_score": round(random.uniform(0.10, 0.50), 3),
            "post_count": random.randint(50, 400),
        })
    return timeline


def _mock_trends_top(limit: int = 10) -> List[Dict[str, Any]]:
    topic_seeds = [
        ("Cyber Attack Surge", ["cyber attack", "malware", "phishing"]),
        ("Defence Budget 2026", ["defence", "budget", "missiles"]),
        ("Telecom Security", ["5G", "spectrum", "surveillance"]),
        ("Border Tensions", ["border", "military", "standoff"]),
        ("AI Disinformation", ["deepfake", "AI", "disinformation"]),
    ]
    results = []
    for i, (label, kws) in enumerate(topic_seeds[:limit]):
        results.append({
            "topic_id": i + 1,
            "label": label,
            "keywords": kws,
            "post_count": random.randint(50, 800),
            "velocity_score": round(random.uniform(0.5, 8.0), 2),
            "timestamp": _ts(random.randint(0, 3)),
        })
    return sorted(results, key=lambda x: x["velocity_score"], reverse=True)


def _mock_demographics() -> Dict[str, Any]:
    return {
        "total_profiled": 1250,
        "age_brackets": {age: random.randint(50, 450) for age in _AGES},
        "professional_roles": {role: random.randint(30, 350) for role in _ROLES},
        "locations": {loc: random.randint(20, 300) for loc in _LOCATIONS},
        "languages": {"en": 950, "hi": 180, "ar": 60, "ru": 40, "zh": 20},
        "generated_at": _ts(),
    }


def _mock_network_graph() -> Dict[str, Any]:
    nodes = []
    edges = []
    user_ids = [hashlib.sha256(f"user_{i}".encode()).hexdigest()[:16] for i in range(25)]

    for i, uid in enumerate(user_ids):
        is_kol = i < 4
        is_bot = i >= 21
        nodes.append({
            "id": uid,
            "label": f"User {i+1}",
            "pagerank": round(random.uniform(0.05, 0.25) if is_kol else random.uniform(0.001, 0.05), 5),
            "betweenness": round(random.uniform(0.05, 0.3) if is_kol else random.uniform(0.0, 0.05), 5),
            "is_kol": is_kol,
            "is_bot": is_bot,
            "platform": random.choice(_PLATFORMS),
            "bot_score": round(random.uniform(0.65, 0.98) if is_bot else random.uniform(0.0, 0.3), 3),
        })

    for i in range(35):
        src = random.choice(user_ids)
        tgt = random.choice(user_ids)
        if src != tgt:
            edges.append({"from": src, "to": tgt, "weight": random.randint(1, 5)})

    return {"nodes": nodes, "edges": edges, "generated_at": _ts()}


# ─── API Routes ───────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {"service": "NTRO Social Media Analytics API Server", "version": "1.1.0", "status": "online"}


@app.get("/api/stats/summary")
async def stats_summary():
    """System-wide summary counters and health status."""
    ch = get_clickhouse()
    data = _mock_stats_summary()
    if ch:
        data["services"]["clickhouse"] = True
    neo = get_neo4j()
    if neo:
        data["services"]["neo4j"] = True
    return data


@app.get("/api/feed/live")
async def feed_live(limit: int = Query(default=20, ge=1, le=100)):
    """Live chronological feed of posts with sentiment/emotion badges."""
    ch = get_clickhouse()
    if ch:
        try:
            posts = ch.fetch_recent_posts(limit=limit)
            if posts:
                return {"feed": posts, "source": "clickhouse"}
        except Exception:
            pass
    return {"feed": _mock_feed_live(limit), "source": "mock"}


@app.get("/api/sentiment/timeline")
async def sentiment_timeline(hours: int = Query(default=24, ge=1, le=168)):
    """Sentiment timeline showing hourly emotion values."""
    ch = get_clickhouse()
    if ch:
        try:
            posts = ch.fetch_posts_in_time_window(hours=hours, limit=5000)
            if posts:
                from collections import defaultdict
                buckets = defaultdict(list)
                for p in posts:
                    ts = p.get("timestamp")
                    hour_key = str(ts)[:13] + ":00:00+00:00"
                    buckets[hour_key].append(p)

                timeline = []
                for hour_key in sorted(buckets.keys()):
                    bucket = buckets[hour_key]
                    n = len(bucket)
                    timeline.append({
                        "timestamp": hour_key,
                        "sarcasm_score":    round(sum(p.get("sarcasm_score", 0) for p in bucket) / n, 3),
                        "anxiety_score":    round(sum(p.get("anxiety_score", 0) for p in bucket) / n, 3),
                        "excitement_score": round(sum(p.get("excitement_score", 0) for p in bucket) / n, 3),
                        "support_score":    round(sum(p.get("support_score", 0) for p in bucket) / n, 3),
                        "opposition_score": round(sum(p.get("opposition_score", 0) for p in bucket) / n, 3),
                        "post_count": n,
                    })
                return {"hours": hours, "timeline": timeline, "source": "clickhouse"}
        except Exception:
            pass
    return {"hours": hours, "timeline": _mock_sentiment_timeline(hours), "source": "mock"}


@app.get("/api/trends/top")
async def trends_top(limit: int = Query(default=10, ge=1, le=50)):
    """Trending topics sorted by velocity score."""
    ch = get_clickhouse()
    if ch:
        try:
            topics = ch.fetch_trending_topics(limit=limit)
            if topics:
                return {"topics": topics, "count": len(topics), "source": "clickhouse"}
        except Exception:
            pass
    mock = _mock_trends_top(limit)
    return {"topics": mock, "count": len(mock), "source": "mock"}


@app.get("/api/demographics/summary")
async def demographics_summary():
    """Distribution of age brackets, professional roles, and GPE locations."""
    ch = get_clickhouse()
    if ch:
        try:
            records = ch.fetch_user_demographics(limit=2000)
            if records:
                from collections import Counter
                age_c = Counter(r["age_bracket"] for r in records)
                role_c = Counter(r["professional_role"] for r in records)
                loc_c = Counter(r["location_gpe"] for r in records)
                lang_c = Counter(r["primary_language"] for r in records)
                return {
                    "total_profiled": len(records),
                    "age_brackets": dict(age_c),
                    "professional_roles": dict(role_c),
                    "locations": dict(loc_c.most_common(10)),
                    "languages": dict(lang_c),
                    "source": "clickhouse",
                }
        except Exception:
            pass
    data = _mock_demographics()
    data["source"] = "mock"
    return data


@app.get("/api/network/graph")
async def network_graph():
    """Interaction topology nodes and edge links."""
    neo = get_neo4j()
    if neo:
        try:
            edges = neo.fetch_all_user_edges()
            users = neo.fetch_all_users()
            if users:
                nodes = []
                for u in users:
                    nodes.append({
                        "id": u["hashed_user_id"],
                        "label": f"User {u['hashed_user_id'][:6]}",
                        "platform": u["platform"],
                        "bot_score": u["bot_score"],
                        "is_bot": u["is_bot"],
                        "is_kol": u.get("is_kol", False),
                        "pagerank": u.get("pagerank", 0.0),
                        "betweenness": u.get("betweenness", 0.0),
                    })
                formatted_edges = [{"from": e["source"], "to": e["target"], "weight": e["weight"]} for e in edges]
                return {"nodes": nodes, "edges": formatted_edges, "source": "neo4j"}
        except Exception:
            pass
    data = _mock_network_graph()
    data["source"] = "mock"
    return data
