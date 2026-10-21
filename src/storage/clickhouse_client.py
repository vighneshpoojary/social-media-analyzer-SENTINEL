"""ClickHouse Client, Schema Initializer, and Multi-Table Storage for NTRO Social Media Analytics."""

import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("ClickHouseStorage")


class ClickHouseStorageClient:
    """Manages ClickHouse connections, table schema initialization, migrations, and data operations."""

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None,
    ) -> None:
        self.host = host or os.getenv("CLICKHOUSE_HOST", "localhost")
        self.port = int(port or os.getenv("CLICKHOUSE_PORT", "9000"))
        self.user = user or os.getenv("CLICKHOUSE_USER", "default")
        self.password = password if password is not None else os.getenv("CLICKHOUSE_PASSWORD", "")
        self.database = database or os.getenv("CLICKHOUSE_DB", "ntro_analytics")
        self._client = None

    def get_client(self, use_db: bool = True):
        """Lazy load and return a clickhouse_driver Client instance."""
        try:
            from clickhouse_driver import Client
        except ImportError:
            logger.error("clickhouse-driver is not installed. Please run: pip install clickhouse-driver")
            raise

        db_name = self.database if use_db else "default"
        return Client(
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password,
            database=db_name,
            connect_timeout=10,
            send_receive_timeout=30,
        )

    def check_connection(self) -> bool:
        """Verify connectivity to ClickHouse server."""
        try:
            client = self.get_client(use_db=False)
            res = client.execute("SELECT 1")
            return res == [(1,)]
        except Exception as e:
            logger.warning(f"Connection check failed: {e}")
            return False

    def init_database(self) -> None:
        """Create database if it does not already exist."""
        client = self.get_client(use_db=False)
        query = f"CREATE DATABASE IF NOT EXISTS {self.database}"
        logger.info(f"Ensuring database exists: {self.database}")
        client.execute(query)
        logger.info(f"Database '{self.database}' ready.")

    def init_tables(self) -> None:
        """Initialize and migrate social_posts, trending_topics, and user_demographics tables."""
        self.init_database()
        client = self.get_client(use_db=True)

        # 1. Social Posts Table
        create_posts_query = f"""
        CREATE TABLE IF NOT EXISTS {self.database}.social_posts (
            timestamp DateTime,
            platform LowCardinality(String),
            user_id String,
            raw_text String,
            post_type LowCardinality(String),
            parent_id Nullable(String),
            sarcasm_score Float32 DEFAULT 0.0,
            anxiety_score Float32 DEFAULT 0.0,
            excitement_score Float32 DEFAULT 0.0,
            support_score Float32 DEFAULT 0.0,
            opposition_score Float32 DEFAULT 0.0
        ) ENGINE = MergeTree()
        ORDER BY (platform, timestamp, user_id)
        """
        logger.info("Creating or verifying 'social_posts' table...")
        client.execute(create_posts_query)

        # Non-destructive column migrations for social_posts
        columns_to_add = [
            ("sarcasm_score", "Float32 DEFAULT 0.0"),
            ("anxiety_score", "Float32 DEFAULT 0.0"),
            ("excitement_score", "Float32 DEFAULT 0.0"),
            ("support_score", "Float32 DEFAULT 0.0"),
            ("opposition_score", "Float32 DEFAULT 0.0"),
        ]
        for col_name, col_type in columns_to_add:
            try:
                alter_query = (
                    f"ALTER TABLE {self.database}.social_posts "
                    f"ADD COLUMN IF NOT EXISTS {col_name} {col_type}"
                )
                client.execute(alter_query)
            except Exception as e:
                logger.debug(f"Column migration notice for '{col_name}': {e}")

        # 2. Trending Topics Table
        create_topics_query = f"""
        CREATE TABLE IF NOT EXISTS {self.database}.trending_topics (
            timestamp DateTime,
            topic_id Int32,
            label String,
            keywords Array(String),
            post_count UInt32,
            velocity_score Float32
        ) ENGINE = MergeTree()
        ORDER BY (timestamp, topic_id)
        """
        logger.info("Creating or verifying 'trending_topics' table...")
        client.execute(create_topics_query)

        # 3. User Demographics Table
        create_demographics_query = f"""
        CREATE TABLE IF NOT EXISTS {self.database}.user_demographics (
            hashed_user_id String,
            age_bracket String,
            location_gpe String,
            primary_language String,
            professional_role String,
            updated_at DateTime
        ) ENGINE = ReplacingMergeTree(updated_at)
        ORDER BY (hashed_user_id)
        """
        logger.info("Creating or verifying 'user_demographics' table...")
        client.execute(create_demographics_query)

        logger.info("All storage tables ('social_posts', 'trending_topics', 'user_demographics') successfully initialized.")

    def insert_post(
        self,
        timestamp: datetime,
        platform: str,
        user_id: str,
        raw_text: str,
        post_type: str,
        parent_id: Optional[str] = None,
        sarcasm_score: float = 0.0,
        anxiety_score: float = 0.0,
        excitement_score: float = 0.0,
        support_score: float = 0.0,
        opposition_score: float = 0.0,
    ) -> None:
        """Insert a single social media post record with emotion scores."""
        client = self.get_client(use_db=True)
        query = f"""
        INSERT INTO {self.database}.social_posts
        (timestamp, platform, user_id, raw_text, post_type, parent_id,
         sarcasm_score, anxiety_score, excitement_score, support_score, opposition_score)
        VALUES
        """
        data = [(
            timestamp,
            platform,
            user_id,
            raw_text,
            post_type,
            parent_id,
            float(sarcasm_score),
            float(anxiety_score),
            float(excitement_score),
            float(support_score),
            float(opposition_score),
        )]
        client.execute(query, data)
        logger.debug(f"Inserted post from {platform} by {user_id}")

    def insert_posts_batch(self, posts: List[Dict[str, Any]]) -> int:
        """Batch insert multiple social media posts with emotion scores."""
        if not posts:
            return 0

        client = self.get_client(use_db=True)
        formatted_rows = []
        for p in posts:
            ts = p["timestamp"]
            if isinstance(ts, str):
                ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            
            formatted_rows.append((
                ts,
                str(p.get("platform", "unknown")),
                str(p.get("user_id", "unknown")),
                str(p.get("raw_text", "")),
                str(p.get("post_type", "post")),
                p.get("parent_id"),
                float(p.get("sarcasm_score", 0.0)),
                float(p.get("anxiety_score", 0.0)),
                float(p.get("excitement_score", 0.0)),
                float(p.get("support_score", 0.0)),
                float(p.get("opposition_score", 0.0)),
            ))

        query = f"""
        INSERT INTO {self.database}.social_posts
        (timestamp, platform, user_id, raw_text, post_type, parent_id,
         sarcasm_score, anxiety_score, excitement_score, support_score, opposition_score)
        VALUES
        """
        client.execute(query, formatted_rows)
        logger.info(f"Batch inserted {len(formatted_rows)} posts into 'social_posts'.")
        return len(formatted_rows)

    def insert_trending_topics(self, topics: List[Dict[str, Any]]) -> int:
        """Insert extracted trending topics into the trending_topics table."""
        if not topics:
            return 0

        client = self.get_client(use_db=True)
        now = datetime.now(timezone.utc)
        formatted_rows = []

        for t in topics:
            ts = t.get("timestamp", now)
            if isinstance(ts, str):
                ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            
            formatted_rows.append((
                ts,
                int(t.get("topic_id", 0)),
                str(t.get("label", "Untitled Topic")),
                [str(k) for k in t.get("keywords", [])],
                int(t.get("post_count", 0)),
                float(t.get("velocity_score", 0.0)),
            ))

        query = f"""
        INSERT INTO {self.database}.trending_topics
        (timestamp, topic_id, label, keywords, post_count, velocity_score)
        VALUES
        """
        client.execute(query, formatted_rows)
        logger.info(f"Inserted {len(formatted_rows)} trending topics into 'trending_topics'.")
        return len(formatted_rows)

    def upsert_user_demographics(self, demographic_records: List[Dict[str, Any]]) -> int:
        """
        Upsert demographic records into the user_demographics table.
        Each record contains:
        - hashed_user_id (str)
        - age_bracket (str)
        - location_gpe (str)
        - primary_language (str)
        - professional_role (str)
        - updated_at (optional datetime, defaults to now)
        """
        if not demographic_records:
            return 0

        client = self.get_client(use_db=True)
        now = datetime.now(timezone.utc)
        formatted_rows = []

        for d in demographic_records:
            updated_at = d.get("updated_at", now)
            if isinstance(updated_at, str):
                updated_at = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))

            formatted_rows.append((
                str(d.get("hashed_user_id", "unknown")),
                str(d.get("age_bracket", "Unknown")),
                str(d.get("location_gpe", "Unknown")),
                str(d.get("primary_language", "en")),
                str(d.get("professional_role", "General / Other")),
                updated_at,
            ))

        query = f"""
        INSERT INTO {self.database}.user_demographics
        (hashed_user_id, age_bracket, location_gpe, primary_language, professional_role, updated_at)
        VALUES
        """
        client.execute(query, formatted_rows)
        logger.info(f"Upserted {len(formatted_rows)} records into 'user_demographics'.")
        return len(formatted_rows)

    def fetch_user_demographics(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Fetch latest user demographic profiles."""
        client = self.get_client(use_db=True)
        query = f"""
        SELECT hashed_user_id, age_bracket, location_gpe, primary_language, professional_role, updated_at
        FROM {self.database}.user_demographics
        ORDER BY updated_at DESC
        LIMIT {limit}
        """
        rows = client.execute(query)
        results = []
        for r in rows:
            results.append({
                "hashed_user_id": r[0],
                "age_bracket": r[1],
                "location_gpe": r[2],
                "primary_language": r[3],
                "professional_role": r[4],
                "updated_at": r[5],
            })
        return results

    def fetch_unique_user_profiles(self, limit: int = 500) -> List[Dict[str, Any]]:
        """Fetch distinct user activity and aggregated text from social_posts for profiling."""
        client = self.get_client(use_db=True)
        query = f"""
        SELECT user_id, count() as post_count, groupArray(raw_text) as sample_texts
        FROM {self.database}.social_posts
        GROUP BY user_id
        ORDER BY post_count DESC
        LIMIT {limit}
        """
        try:
            rows = client.execute(query)
            results = []
            for r in rows:
                texts = r[2][:10] if isinstance(r[2], list) else [str(r[2])]
                combined = " ".join(texts)
                results.append({
                    "user_id": r[0],
                    "post_count": r[1],
                    "raw_text": combined,
                })
            return results
        except Exception as e:
            logger.error(f"Error fetching unique user profiles: {e}")
            return []

    def fetch_recent_posts(self, limit: int = 10, platform: Optional[str] = None) -> List[Dict[str, Any]]:
        """Fetch recent posts with emotion metrics from the database."""
        client = self.get_client(use_db=True)
        where_clause = f"WHERE platform = '{platform}'" if platform else ""
        query = f"""
        SELECT timestamp, platform, user_id, raw_text, post_type, parent_id,
               sarcasm_score, anxiety_score, excitement_score, support_score, opposition_score
        FROM {self.database}.social_posts
        {where_clause}
        ORDER BY timestamp DESC
        LIMIT {limit}
        """
        rows = client.execute(query)
        results = []
        for r in rows:
            results.append({
                "timestamp": r[0],
                "platform": r[1],
                "user_id": r[2],
                "raw_text": r[3],
                "post_type": r[4],
                "parent_id": r[5],
                "sarcasm_score": r[6],
                "anxiety_score": r[7],
                "excitement_score": r[8],
                "support_score": r[9],
                "opposition_score": r[10],
            })
        return results

    def fetch_posts_in_time_window(self, hours: float = 2.0, limit: int = 1000) -> List[Dict[str, Any]]:
        """Fetch posts within the past N hours for time-window analytics and topic modeling."""
        client = self.get_client(use_db=True)
        seconds_back = int(hours * 3600)
        query = f"""
        SELECT timestamp, platform, user_id, raw_text, post_type, parent_id,
               sarcasm_score, anxiety_score, excitement_score, support_score, opposition_score
        FROM {self.database}.social_posts
        WHERE timestamp >= now() - INTERVAL {seconds_back} SECOND
        ORDER BY timestamp DESC
        LIMIT {limit}
        """
        try:
            rows = client.execute(query)
            results = []
            for r in rows:
                results.append({
                    "timestamp": r[0],
                    "platform": r[1],
                    "user_id": r[2],
                    "raw_text": r[3],
                    "post_type": r[4],
                    "parent_id": r[5],
                    "sarcasm_score": r[6],
                    "anxiety_score": r[7],
                    "excitement_score": r[8],
                    "support_score": r[9],
                    "opposition_score": r[10],
                })
            return results
        except Exception as e:
            logger.error(f"Error fetching posts in time window: {e}")
            return []

    def fetch_trending_topics(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Fetch latest extracted trending topics."""
        client = self.get_client(use_db=True)
        query = f"""
        SELECT timestamp, topic_id, label, keywords, post_count, velocity_score
        FROM {self.database}.trending_topics
        ORDER BY timestamp DESC, velocity_score DESC
        LIMIT {limit}
        """
        rows = client.execute(query)
        results = []
        for r in rows:
            results.append({
                "timestamp": r[0],
                "topic_id": r[1],
                "label": r[2],
                "keywords": r[3],
                "post_count": r[4],
                "velocity_score": r[5],
            })
        return results

    def describe_table(self, table_name: str = "social_posts") -> List[Dict[str, str]]:
        """Return the column schema of a ClickHouse table."""
        client = self.get_client(use_db=True)
        query = f"DESCRIBE TABLE {self.database}.{table_name}"
        rows = client.execute(query)
        return [{"name": r[0], "type": r[1]} for r in rows]


def initialize_storage():
    """CLI initialization entry point."""
    logger.info("Initializing ClickHouse Storage Client...")
    client = ClickHouseStorageClient()

    logger.info(f"Target ClickHouse Host: {client.host}:{client.port} (DB: {client.database})")
    if not client.check_connection():
        logger.warning(
            "Could not connect to ClickHouse server. "
            "Please ensure the ClickHouse container is running: `docker compose up -d`"
        )
        return False

    client.init_tables()
    for tbl in ("social_posts", "trending_topics", "user_demographics"):
        schema = client.describe_table(tbl)
        logger.info(f"Verified Schema for '{tbl}':")
        for col in schema:
            logger.info(f"  - {col['name']}: {col['type']}")

    logger.info("ClickHouse Storage initialized successfully!")
    return True


if __name__ == "__main__":
    success = initialize_storage()
    if not success:
        sys.exit(1)
