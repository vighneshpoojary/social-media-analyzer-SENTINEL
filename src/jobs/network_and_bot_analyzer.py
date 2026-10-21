"""Combined Network & Bot Analysis Job for NTRO Social Media Analytics."""

import argparse
import asyncio
import hashlib
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.graph.network_analysis import NetworkAnalyzer
from src.nlp.bot_detector import detect_bots
from src.storage.clickhouse_client import ClickHouseStorageClient
from src.storage.neo4j_client import Neo4jClient

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("NetworkAndBotJob")


def _sha256(uid: str) -> str:
    if len(uid) == 64 and all(c in "0123456789abcdefABCDEF" for c in uid):
        return uid.lower()
    return hashlib.sha256(uid.encode("utf-8")).hexdigest()


class NetworkAndBotAnalyzerJob:
    """
    Combined orchestration job that:
    1. Reads recent ClickHouse posts and builds interaction edges.
    2. Syncs User, Post, and interaction edges into Neo4j.
    3. Computes PageRank & Betweenness Centrality to identify KOLs.
    4. Runs Isolation Forest bot detection on user behavioral signals.
    5. Writes KOL and bot scores back into Neo4j User nodes.
    """

    def __init__(
        self,
        hours: float = 2.0,
        clickhouse: Optional[ClickHouseStorageClient] = None,
        neo4j: Optional[Neo4jClient] = None,
    ) -> None:
        self.hours = hours
        self.storage = clickhouse or ClickHouseStorageClient()
        self.neo4j = neo4j or Neo4jClient()
        self.analyzer = NetworkAnalyzer()

    # ─────────────────────── ClickHouse → Neo4j Sync ──────────────

    def _derive_activity_metrics(
        self,
        posts: List[Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compute behavioral metrics per hashed_user_id from post records.
        Returns: {hashed_user_id: {posts_per_hour, repetition_rate, ...}}
        """
        from collections import defaultdict

        user_posts: Dict[str, List[Dict]] = defaultdict(list)
        for p in posts:
            user_posts[p["user_id"]].append(p)

        metrics = {}
        for uid, uposts in user_posts.items():
            hashed = _sha256(uid)
            texts = [p.get("raw_text", "") for p in uposts]
            unique_texts = len(set(texts))
            rep_rate = 1.0 - (unique_texts / max(len(texts), 1))

            # Simple burst flag: >5 posts in the same platform
            platform_counts: Dict[str, int] = {}
            for p in uposts:
                pl = p.get("platform", "")
                platform_counts[pl] = platform_counts.get(pl, 0) + 1
            burst_flag = int(any(v >= 5 for v in platform_counts.values()))

            # Rough posts-per-hour using hour window
            pph = len(uposts) / max(self.hours, 0.1)

            metrics[hashed] = {
                "hashed_user_id": hashed,
                "posts_per_hour": round(pph, 2),
                "repetition_rate": round(rep_rate, 4),
                "unique_recipients": len(set(
                    p.get("parent_id", "") for p in uposts if p.get("parent_id")
                )),
                "avg_response_time_sec": 300.0,  # placeholder
                "follower_following_ratio": 1.0,  # placeholder
                "account_age_days": 365.0,         # placeholder
                "burst_flag": burst_flag,
            }
        return metrics

    def sync_to_neo4j(self, posts: List[Dict[str, Any]]) -> int:
        """Upsert post and user nodes + edges into Neo4j from ClickHouse posts."""
        synced = 0
        for i, post in enumerate(posts):
            uid_raw = post.get("user_id", "unknown")
            hashed_uid = _sha256(uid_raw)
            platform = post.get("platform", "unknown")
            raw_text = post.get("raw_text", "")
            timestamp = str(post.get("timestamp", ""))
            post_type = post.get("post_type", "post")
            parent_id = post.get("parent_id")

            post_id = f"{platform}_{hashed_uid[:8]}_{i}"

            # Upsert User node
            self.neo4j.upsert_user(hashed_uid, platform)

            # Upsert Post node
            self.neo4j.upsert_post(
                post_id=post_id,
                platform=platform,
                raw_text=raw_text,
                timestamp=timestamp,
                post_type=post_type,
                sarcasm_score=post.get("sarcasm_score", 0.0),
                anxiety_score=post.get("anxiety_score", 0.0),
                excitement_score=post.get("excitement_score", 0.0),
                support_score=post.get("support_score", 0.0),
                opposition_score=post.get("opposition_score", 0.0),
            )

            # POSTED edge
            self.neo4j.link_user_posted(hashed_uid, post_id, timestamp)

            # REPLIED_TO edge: link user who posted the reply to the post author
            if parent_id:
                parent_hashed = _sha256(parent_id)
                # Ensure parent user node exists
                self.neo4j.upsert_user(parent_hashed, platform)
                self.neo4j.link_user_interaction(hashed_uid, parent_hashed, "INTERACTS_WITH")

            synced += 1

        logger.info(f"Synced {synced} posts to Neo4j.")
        return synced

    # ─────────────────────── Main Analysis ─────────────────────────

    def run_once(self, mock_posts: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Execute a single network + bot analysis cycle."""
        logger.info(f"Starting network & bot analysis cycle (window={self.hours}h)...")

        # 1. Fetch posts
        if mock_posts is not None:
            posts = mock_posts
            logger.info(f"Using {len(posts)} mock posts.")
        else:
            if not self.storage.check_connection():
                logger.warning("ClickHouse unavailable. Skipping cycle.")
                return {}
            posts = self.storage.fetch_posts_in_time_window(hours=self.hours)
            logger.info(f"Fetched {len(posts)} posts from ClickHouse.")

        if not posts:
            logger.info("No posts available for analysis.")
            return {}

        # 2. Derive behavioral metrics per user
        user_metrics = self._derive_activity_metrics(posts)

        # 3. Bot detection
        logger.info(f"Running bot detection on {len(user_metrics)} unique users...")
        bot_results = detect_bots(list(user_metrics.values()))

        # 4. Build interaction graph edges
        # Each reply/parent_id pair is an interaction edge
        edges: List[Dict[str, Any]] = []
        hashed_users: Dict[str, int] = {}  # uid -> post count

        for i, p in enumerate(posts):
            uid_raw = p.get("user_id", "unknown")
            hashed_uid = _sha256(uid_raw)
            hashed_users[hashed_uid] = hashed_users.get(hashed_uid, 0) + 1

            parent_id = p.get("parent_id")
            if parent_id:
                parent_hashed = _sha256(parent_id)
                edges.append({"source": hashed_uid, "target": parent_hashed, "weight": 1.0})

        # Add self-edges so isolated nodes appear in graph
        for uid in hashed_users:
            if not any(e["source"] == uid or e["target"] == uid for e in edges):
                edges.append({"source": uid, "target": uid, "weight": 0.01})

        # 5. Network analysis
        logger.info(f"Computing network metrics ({len(edges)} edges)...")
        analysis = self.analyzer.run_full_analysis(edges, top_n_kols=10)
        kols = analysis["kols"]

        # 6. Sync to Neo4j (if available)
        neo4j_available = False
        if mock_posts is None:
            try:
                neo4j_available = self.neo4j.connect()
            except Exception:
                pass

        if neo4j_available:
            self.neo4j.create_indexes()
            self.sync_to_neo4j(posts)

            # Write centrality scores back to Neo4j user nodes
            pagerank = analysis["pagerank"]
            betweenness = analysis["betweenness"]
            for kol in kols:
                uid = kol["hashed_user_id"]
                self.neo4j.update_user_centrality(
                    uid,
                    pagerank=kol["pagerank"],
                    betweenness=kol["betweenness"],
                    is_kol=kol["is_kol"],
                )

            # Write bot scores to Neo4j
            for uid, bot_result in bot_results.items():
                self.neo4j.update_user_bot_score(
                    uid,
                    bot_score=bot_result["bot_score"],
                    is_bot=bot_result["is_bot"],
                    anomaly_flags=bot_result["anomaly_flags"],
                )
        else:
            logger.info("Neo4j not available — skipping graph sync (run `docker compose up -d`).")

        # 7. Summary
        bot_count = sum(1 for r in bot_results.values() if r["is_bot"])
        kol_count = sum(1 for k in kols if k["is_kol"])
        logger.info(
            f"Analysis complete → Nodes: {analysis['node_count']}, "
            f"Edges: {analysis['edge_count']}, KOLs: {kol_count}, Bots: {bot_count}"
        )

        if kols:
            logger.info("Top KOLs identified:")
            for k in kols[:5]:
                logger.info(
                    f"  [{k['hashed_user_id'][:10]}...] "
                    f"PR={k['pagerank']:.5f} | BW={k['betweenness']:.5f} | KOL={k['is_kol']}"
                )

        return {
            "network": analysis,
            "bot_results": bot_results,
            "kols": kols,
            "bot_count": bot_count,
            "kol_count": kol_count,
        }

    async def run_loop(self, interval_seconds: int = 600) -> None:
        """Run periodic analysis loop."""
        logger.info(f"Starting Network & Bot Analyzer loop (interval={interval_seconds}s)...")
        try:
            while True:
                self.run_once()
                await asyncio.sleep(interval_seconds)
        except asyncio.CancelledError:
            logger.info("Network & Bot Analyzer loop cancelled.")
        finally:
            self.neo4j.close()


def main():
    parser = argparse.ArgumentParser(
        description="NTRO Social Media Analytics - Network & Bot Analyzer Job"
    )
    parser.add_argument("--hours", type=float, default=2.0,
                        help="Time window in hours for post retrieval (default: 2.0).")
    parser.add_argument("--interval", type=int, default=600,
                        help="Loop interval in seconds (default: 600).")
    parser.add_argument("--once", action="store_true",
                        help="Run a single cycle and exit.")
    parser.add_argument("--mock-data", action="store_true",
                        help="Use simulated posts for testing.")

    args = parser.parse_args()

    mock_posts = None
    if args.mock_data:
        mock_posts = _generate_mock_posts()

    job = NetworkAndBotAnalyzerJob(hours=args.hours)

    if args.once:
        job.run_once(mock_posts=mock_posts)
    else:
        asyncio.run(job.run_loop(interval_seconds=args.interval))


def _generate_mock_posts() -> List[Dict[str, Any]]:
    """Generate synthetic post data covering human and bot-like users."""
    from datetime import timedelta

    now = datetime.now(timezone.utc)
    posts = []

    # Normal users
    for i in range(5):
        posts.append({
            "user_id": f"human_user_{i}",
            "platform": "telegram",
            "raw_text": f"Discussing national security topic {i}.",
            "timestamp": (now - timedelta(minutes=i * 10)).isoformat(),
            "post_type": "post",
            "parent_id": f"human_user_{(i+1) % 5}" if i % 2 == 0 else None,
        })

    # Bot-like user (high frequency, repetitive)
    for j in range(25):
        posts.append({
            "user_id": "bot_suspect_99",
            "platform": "x",
            "raw_text": "BREAKING: Urgent update! Share now! #alert #news",
            "timestamp": (now - timedelta(minutes=j)).isoformat(),
            "post_type": "post",
            "parent_id": None,
        })

    # A second influencer-like user with many replies
    for k in range(8):
        posts.append({
            "user_id": "influencer_user_42",
            "platform": "telegram",
            "raw_text": f"Analysis of evolving threat landscape — perspective {k}.",
            "timestamp": (now - timedelta(minutes=k * 5)).isoformat(),
            "post_type": "reply" if k % 2 == 0 else "post",
            "parent_id": f"human_user_{k % 5}" if k % 2 == 0 else None,
        })

    return posts


if __name__ == "__main__":
    main()
