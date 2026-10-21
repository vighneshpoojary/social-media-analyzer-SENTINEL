"""Automated Job for User Demographic Profiling and Aggregation in ClickHouse."""

import argparse
import asyncio
import hashlib
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

from src.nlp.demographics import infer_user_demographics
from src.storage.clickhouse_client import ClickHouseStorageClient

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("DemographicProfilerJob")


def ensure_sha256(user_id: str) -> str:
    """Ensure user ID is a 64-char SHA-256 hex string."""
    if len(user_id) == 64 and all(c in "0123456789abcdefABCDEF" for c in user_id):
        return user_id.lower()
    return hashlib.sha256(str(user_id).encode("utf-8")).hexdigest()


class DemographicProfilerJob:
    """
    Automated background profiler that:
    1. Fetches unique user activity records from ClickHouse.
    2. Runs demographic inference (NER location, age bracket, role, language).
    3. Hashes user IDs for privacy compliance.
    4. Upserts records into ClickHouse `user_demographics` table.
    """

    def __init__(
        self,
        batch_limit: int = 500,
        clickhouse_client: Optional[ClickHouseStorageClient] = None,
    ) -> None:
        self.batch_limit = batch_limit
        self.storage = clickhouse_client or ClickHouseStorageClient()

    def run_once(self, mock_users: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """Execute single demographic profiling cycle."""
        logger.info("Running automated demographic profiling cycle...")

        # 1. Fetch user data
        if mock_users is not None:
            user_records = mock_users
            logger.info(f"Using {len(user_records)} mock user profile(s).")
        else:
            if not self.storage.check_connection():
                logger.warning("ClickHouse is unreachable. Skipping demographic profiling.")
                return []
            user_records = self.storage.fetch_unique_user_profiles(limit=self.batch_limit)
            logger.info(f"Fetched {len(user_records)} unique user profiles from ClickHouse.")

        if not user_records:
            logger.info("No user profiles available for demographic inference.")
            return []

        # 2. Infer demographics for each user
        demographic_results = []
        now = datetime.now(timezone.utc)

        for u in user_records:
            raw_uid = u.get("user_id", "unknown")
            hashed_uid = ensure_sha256(raw_uid)

            inferred = infer_user_demographics(u)
            record = {
                "hashed_user_id": hashed_uid,
                "age_bracket": inferred["age_bracket"],
                "location_gpe": inferred["location_gpe"],
                "primary_language": inferred["primary_language"],
                "professional_role": inferred["professional_role"],
                "updated_at": now,
            }
            demographic_results.append(record)
            logger.info(
                f"  -> Profiled User [{hashed_uid[:8]}...] | Age: {record['age_bracket']} | "
                f"Loc: {record['location_gpe']} | Role: {record['professional_role']} | Lang: {record['primary_language']}"
            )

        # 3. Upsert into ClickHouse
        if mock_users is None and self.storage.check_connection():
            try:
                self.storage.upsert_user_demographics(demographic_results)
                logger.info(f"Successfully upserted {len(demographic_results)} records into 'user_demographics'.")
            except Exception as e:
                logger.error(f"Failed to upsert demographic records into ClickHouse: {e}")

        return demographic_results

    async def run_loop(self, interval_seconds: int = 600) -> None:
        """Run periodic profiling loop asynchronously."""
        logger.info(f"Starting Demographic Profiler loop (interval={interval_seconds}s)...")
        try:
            while True:
                self.run_once()
                logger.info(f"Sleeping for {interval_seconds}s until next profiling round...")
                await asyncio.sleep(interval_seconds)
        except asyncio.CancelledError:
            logger.info("Demographic Profiler loop cancelled.")


def main():
    parser = argparse.ArgumentParser(
        description="NTRO Social Media Analytics - Automated Demographic Profiler Job"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Maximum unique users to profile per run (default: 500).",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=600,
        help="Loop interval in seconds between profiling cycles (default: 600).",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Execute single profiling cycle and exit immediately.",
    )
    parser.add_argument(
        "--mock-data",
        action="store_true",
        help="Run analysis using simulated user profiles.",
    )

    args = parser.parse_args()

    job = DemographicProfilerJob(batch_limit=args.limit)

    mock_users = None
    if args.mock_data:
        mock_users = [
            {
                "user_id": "user_alpha_1",
                "bio": "24yo Cyber Security Analyst @CERT-In. Based in New Delhi 📍. Ethical Hacking.",
                "raw_text": "Monitoring new threat vectors across telecom sectors.",
            },
            {
                "user_id": "user_beta_2",
                "bio": "Defence policy researcher and military correspondent. Class of 2018. Living in London.",
                "raw_text": "Evaluating missile defence and strategic regional agreements.",
            },
            {
                "user_id": "user_gamma_3",
                "bio": "Senior Software Engineer with 18 years in cloud architecture. Father of 2. Bengaluru, India.",
                "raw_text": "Deploying scalable distributed microservices.",
            },
            {
                "user_id": "user_delta_4",
                "bio": "Retired Army Colonel (ret.), veteran 35+ years. Mumbai.",
                "raw_text": "National security and border preparedness insights.",
            },
        ]

    if args.once:
        job.run_once(mock_users=mock_users)
    else:
        asyncio.run(job.run_loop(interval_seconds=args.interval))


if __name__ == "__main__":
    main()
