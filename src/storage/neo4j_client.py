"""Neo4j Graph Database Client for NTRO Social Media Analytics."""

import logging
import os
import sys
from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional
from dotenv import load_dotenv

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

load_dotenv()

logger = logging.getLogger("Neo4jClient")


class Neo4jClient:
    """
    Manages Neo4j connectivity and Cypher operations for the NTRO social graph.

    Graph Schema:
    - Nodes: (:User), (:Post), (:Topic)
    - Edges: (:User)-[:POSTED]->(:Post),
             (:Post)-[:REPLIED_TO]->(:Post),
             (:Post)-[:RETWEETED]->(:Post),
             (:Post)-[:TAGGED_WITH]->(:Topic),
             (:User)-[:INTERACTS_WITH]->(:User)
    """

    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
    ) -> None:
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.getenv("NEO4J_USER", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "ntro_password")
        self._driver = None

    def connect(self) -> bool:
        """Open the Neo4j driver connection. Returns True if successful."""
        try:
            from neo4j import GraphDatabase
            self._driver = GraphDatabase.driver(
                self.uri, auth=(self.user, self.password)
            )
            # Verify connectivity
            self._driver.verify_connectivity()
            logger.info(f"Connected to Neo4j at {self.uri}")
            return True
        except Exception as e:
            logger.warning(f"Neo4j connection failed ({self.uri}): {e}")
            self._driver = None
            return False

    def close(self) -> None:
        """Close the Neo4j driver connection."""
        if self._driver:
            self._driver.close()
            self._driver = None

    @contextmanager
    def session(self) -> Generator:
        """Context manager yielding a Neo4j session."""
        if self._driver is None:
            if not self.connect():
                raise RuntimeError("Neo4j is not connected. Run `docker compose up -d`.")
        session = self._driver.session()
        try:
            yield session
        finally:
            session.close()

    # ─────────────────────── Schema Setup ─────────────────────────

    def create_indexes(self) -> None:
        """Create indexes and uniqueness constraints for fast lookup."""
        constraints = [
            "CREATE CONSTRAINT user_id_unique IF NOT EXISTS FOR (u:User) REQUIRE u.hashed_user_id IS UNIQUE",
            "CREATE CONSTRAINT post_id_unique IF NOT EXISTS FOR (p:Post) REQUIRE p.post_id IS UNIQUE",
            "CREATE CONSTRAINT topic_id_unique IF NOT EXISTS FOR (t:Topic) REQUIRE t.topic_id IS UNIQUE",
        ]
        with self.session() as s:
            for stmt in constraints:
                try:
                    s.run(stmt)
                except Exception as e:
                    logger.debug(f"Index/constraint notice: {e}")
        logger.info("Neo4j indexes and constraints initialized.")

    # ─────────────────────── Node Upserts ─────────────────────────

    def upsert_user(self, hashed_user_id: str, platform: str, **props) -> None:
        """Create or update a User node."""
        cypher = """
        MERGE (u:User {hashed_user_id: $hashed_user_id})
        SET u.platform = $platform,
            u.updated_at = $updated_at,
            u.post_count  = coalesce(u.post_count, 0) + $post_count_inc,
            u.bot_score   = $bot_score,
            u.is_bot      = $is_bot,
            u.age_bracket = $age_bracket,
            u.location_gpe = $location_gpe,
            u.professional_role = $professional_role
        """
        params = {
            "hashed_user_id": hashed_user_id,
            "platform": platform,
            "updated_at": props.get("updated_at", ""),
            "post_count_inc": props.get("post_count_inc", 1),
            "bot_score": props.get("bot_score", 0.0),
            "is_bot": props.get("is_bot", False),
            "age_bracket": props.get("age_bracket", "Unknown"),
            "location_gpe": props.get("location_gpe", "Unknown"),
            "professional_role": props.get("professional_role", "General / Other"),
        }
        with self.session() as s:
            s.run(cypher, params)

    def upsert_post(
        self,
        post_id: str,
        platform: str,
        raw_text: str,
        timestamp: str,
        post_type: str = "post",
        **scores,
    ) -> None:
        """Create or update a Post node."""
        cypher = """
        MERGE (p:Post {post_id: $post_id})
        SET p.platform      = $platform,
            p.raw_text      = $raw_text,
            p.timestamp     = $timestamp,
            p.post_type     = $post_type,
            p.sarcasm_score = $sarcasm_score,
            p.anxiety_score = $anxiety_score,
            p.excitement_score = $excitement_score,
            p.support_score = $support_score,
            p.opposition_score = $opposition_score
        """
        params = {
            "post_id": post_id,
            "platform": platform,
            "raw_text": raw_text[:2000],
            "timestamp": str(timestamp),
            "post_type": post_type,
            "sarcasm_score": scores.get("sarcasm_score", 0.0),
            "anxiety_score": scores.get("anxiety_score", 0.0),
            "excitement_score": scores.get("excitement_score", 0.0),
            "support_score": scores.get("support_score", 0.0),
            "opposition_score": scores.get("opposition_score", 0.0),
        }
        with self.session() as s:
            s.run(cypher, params)

    def upsert_topic(self, topic_id: int, label: str, keywords: List[str]) -> None:
        """Create or update a Topic node."""
        cypher = """
        MERGE (t:Topic {topic_id: $topic_id})
        SET t.label    = $label,
            t.keywords = $keywords
        """
        with self.session() as s:
            s.run(cypher, {"topic_id": topic_id, "label": label, "keywords": keywords})

    # ─────────────────────── Edge Creation ─────────────────────────

    def link_user_posted(self, hashed_user_id: str, post_id: str, timestamp: str) -> None:
        """Create (:User)-[:POSTED]->(:Post) edge."""
        cypher = """
        MATCH (u:User {hashed_user_id: $hashed_user_id})
        MATCH (p:Post {post_id: $post_id})
        MERGE (u)-[r:POSTED]->(p)
        SET r.timestamp = $timestamp
        """
        with self.session() as s:
            s.run(cypher, {"hashed_user_id": hashed_user_id, "post_id": post_id, "timestamp": timestamp})

    def link_reply(self, child_post_id: str, parent_post_id: str) -> None:
        """Create (:Post)-[:REPLIED_TO]->(:Post) edge."""
        cypher = """
        MATCH (child:Post  {post_id: $child_id})
        MATCH (parent:Post {post_id: $parent_id})
        MERGE (child)-[:REPLIED_TO]->(parent)
        """
        with self.session() as s:
            s.run(cypher, {"child_id": child_post_id, "parent_id": parent_post_id})

    def link_retweet(self, retweet_post_id: str, original_post_id: str) -> None:
        """Create (:Post)-[:RETWEETED]->(:Post) edge."""
        cypher = """
        MATCH (rt:Post  {post_id: $rt_id})
        MATCH (orig:Post {post_id: $orig_id})
        MERGE (rt)-[:RETWEETED]->(orig)
        """
        with self.session() as s:
            s.run(cypher, {"rt_id": retweet_post_id, "orig_id": original_post_id})

    def link_post_topic(self, post_id: str, topic_id: int) -> None:
        """Create (:Post)-[:TAGGED_WITH]->(:Topic) edge."""
        cypher = """
        MATCH (p:Post {post_id: $post_id})
        MATCH (t:Topic {topic_id: $topic_id})
        MERGE (p)-[:TAGGED_WITH]->(t)
        """
        with self.session() as s:
            s.run(cypher, {"post_id": post_id, "topic_id": topic_id})

    def link_user_interaction(
        self, from_user_id: str, to_user_id: str, interaction_type: str = "INTERACTS_WITH"
    ) -> None:
        """Create a user-to-user interaction edge (INTERACTS_WITH / RETWEETED_USER etc.)."""
        cypher = f"""
        MATCH (a:User {{hashed_user_id: $from_id}})
        MATCH (b:User {{hashed_user_id: $to_id}})
        MERGE (a)-[r:{interaction_type}]->(b)
        ON CREATE SET r.weight = 1
        ON MATCH  SET r.weight = r.weight + 1
        """
        with self.session() as s:
            s.run(cypher, {"from_id": from_user_id, "to_id": to_user_id})

    # ─────────────────────── Queries ──────────────────────────────

    def fetch_all_user_edges(self) -> List[Dict[str, Any]]:
        """Return all user INTERACTS_WITH edges for graph analysis."""
        cypher = """
        MATCH (a:User)-[r:INTERACTS_WITH]->(b:User)
        RETURN a.hashed_user_id AS source,
               b.hashed_user_id AS target,
               r.weight          AS weight
        """
        with self.session() as s:
            result = s.run(cypher)
            return [{"source": r["source"], "target": r["target"], "weight": r["weight"]}
                    for r in result]

    def fetch_all_users(self) -> List[Dict[str, Any]]:
        """Return all user nodes with metadata."""
        cypher = """
        MATCH (u:User)
        RETURN u.hashed_user_id AS hashed_user_id,
               u.platform        AS platform,
               u.post_count      AS post_count,
               u.bot_score       AS bot_score,
               u.is_bot          AS is_bot
        """
        with self.session() as s:
            result = s.run(cypher)
            return [dict(r) for r in result]

    def update_user_centrality(
        self,
        hashed_user_id: str,
        pagerank: float,
        betweenness: float,
        is_kol: bool,
    ) -> None:
        """Write PageRank and betweenness centrality scores back to a User node."""
        cypher = """
        MATCH (u:User {hashed_user_id: $uid})
        SET u.pagerank     = $pagerank,
            u.betweenness  = $betweenness,
            u.is_kol       = $is_kol
        """
        with self.session() as s:
            s.run(cypher, {
                "uid": hashed_user_id,
                "pagerank": pagerank,
                "betweenness": betweenness,
                "is_kol": is_kol,
            })

    def update_user_bot_score(
        self,
        hashed_user_id: str,
        bot_score: float,
        is_bot: bool,
        anomaly_flags: List[str],
    ) -> None:
        """Write bot detection results back to a User node."""
        cypher = """
        MATCH (u:User {hashed_user_id: $uid})
        SET u.bot_score     = $bot_score,
            u.is_bot        = $is_bot,
            u.anomaly_flags = $anomaly_flags
        """
        with self.session() as s:
            s.run(cypher, {
                "uid": hashed_user_id,
                "bot_score": bot_score,
                "is_bot": is_bot,
                "anomaly_flags": anomaly_flags,
            })

    def count_nodes(self, label: str) -> int:
        """Count nodes of a given label."""
        with self.session() as s:
            result = s.run(f"MATCH (n:{label}) RETURN count(n) AS cnt")
            return result.single()["cnt"]
