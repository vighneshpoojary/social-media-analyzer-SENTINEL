"""Automated Job for Real-Time Trend & Topic Detection from ClickHouse Social Posts."""

import argparse
import asyncio
import logging
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.nlp.topic_modeling import extract_trending_topics
from src.storage.clickhouse_client import ClickHouseStorageClient

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("TrendAnalyzerJob")


class TrendAnalyzerJob:
    """
    Automated background analyzer that:
    1. Fetches recent posts from ClickHouse for the past N hours.
    2. Runs semantic and n-gram topic clustering.
    3. Calculates real-time velocity metrics.
    4. Persists trending topic records into ClickHouse `trending_topics` table.
    """

    def __init__(
        self,
        hours_window: float = 2.0,
        top_n: int = 10,
        clickhouse_client: Optional[ClickHouseStorageClient] = None,
    ) -> None:
        self.hours_window = hours_window
        self.top_n = top_n
        self.storage = clickhouse_client or ClickHouseStorageClient()

    def run_once(self, mock_posts: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """Execute a single cycle of trend extraction and storage."""
        logger.info(f"Running trend analysis over past {self.hours_window} hour(s)...")

        # 1. Fetch posts from ClickHouse
        if mock_posts is not None:
            posts = mock_posts
            logger.info(f"Using {len(posts)} provided posts for analysis.")
        else:
            if not self.storage.check_connection():
                logger.warning("ClickHouse server is unreachable. Aborting trend analysis cycle.")
                return []
            posts = self.storage.fetch_posts_in_time_window(hours=self.hours_window)
            logger.info(f"Fetched {len(posts)} posts from ClickHouse.")

        if not posts:
            logger.info("No posts available in the specified time window for topic extraction.")
            return []

        # 2. Extract trending topics and compute velocity
        logger.info(f"Extracting top {self.top_n} trending topics...")
        topics = extract_trending_topics(
            posts, top_n=self.top_n, time_window_hours=self.hours_window
        )

        if not topics:
            logger.info("No distinct topics formed from the current post volume.")
            return []

        # 3. Attach timestamp
        now = datetime.now(timezone.utc)
        for t in topics:
            t["timestamp"] = now

        logger.info(f"Identified {len(topics)} trending topic(s):")
        for t in topics:
            logger.info(
                f"  -> [{t['topic_id']}] '{t['label']}' | Count: {t['post_count']} | Velocity: {t['velocity_score']}/hr | Keywords: {t['keywords']}"
            )

        # 4. Save to ClickHouse if connected
        if mock_posts is None and self.storage.check_connection():
            try:
                self.storage.insert_trending_topics(topics)
                logger.info(f"Persisted {len(topics)} topics to ClickHouse 'trending_topics'.")
            except Exception as e:
                logger.error(f"Failed to persist trending topics to ClickHouse: {e}")

        return topics

    async def run_loop(self, interval_seconds: int = 300) -> None:
        """Run periodic trend analysis loop asynchronously."""
        logger.info(
            f"Starting TrendAnalyzer periodic loop (interval={interval_seconds}s, window={self.hours_window}h)..."
        )
        try:
            while True:
                self.run_once()
                logger.info(f"Sleeping for {interval_seconds}s until next trend cycle...")
                await asyncio.sleep(interval_seconds)
        except asyncio.CancelledError:
            logger.info("TrendAnalyzer periodic loop cancelled.")


def main():
    parser = argparse.ArgumentParser(
        description="NTRO Social Media Analytics - Real-Time Trend & Topic Detection Job"
    )
    parser.add_argument(
        "--hours",
        type=float,
        default=2.0,
        help="Time window in hours to analyze (default: 2.0).",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Maximum number of trending topics to extract (default: 10).",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=300,
        help="Loop interval in seconds between analysis cycles (default: 300).",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Execute single analysis cycle and exit immediately.",
    )
    parser.add_argument(
        "--mock-data",
        action="store_true",
        help="Run analysis using simulated cybersecurity and regional news posts.",
    )

    args = parser.parse_args()

    job = TrendAnalyzerJob(hours_window=args.hours, top_n=args.top_n)

    mock_posts = None
    if args.mock_data:
        mock_posts = [
            {"raw_text": "Severe cybersecurity vulnerability in critical power grid infrastructure."},
            {"raw_text": "Zero-day exploit detected targeting government communications network."},
            {"raw_text": "Emergency patch released for zero-day cybersecurity vulnerability."},
            {"raw_text": "Critical alert: ransomware attacks escalating on telecom infrastructure."},
            {"raw_text": "National defence and border surveillance systems enhanced with AI radar."},
            {"raw_text": "Strategic defence agreement signed for regional maritime security cooperation."},
            {"raw_text": "Economic markets surge with record growth in domestic technology sectors."},
            {"raw_text": "Quarterly inflation rate drops, boosting regional economic confidence."},
        ]

    if args.once:
        job.run_once(mock_posts=mock_posts)
    else:
        asyncio.run(job.run_loop(interval_seconds=args.interval))


if __name__ == "__main__":
    main()
