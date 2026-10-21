"""Unit tests for Graph Network Analysis and Bot Detection."""

import hashlib
import unittest
from unittest.mock import MagicMock, patch

from src.graph.network_analysis import NetworkAnalyzer
from src.jobs.network_and_bot_analyzer import NetworkAndBotAnalyzerJob, _generate_mock_posts
from src.nlp.bot_detector import BotDetector, detect_bots
from src.storage.neo4j_client import Neo4jClient


def _sha256(uid: str) -> str:
    return hashlib.sha256(uid.encode()).hexdigest()


# ──────────────────────────────────────────────────────────────────
# Network Analysis Tests
# ──────────────────────────────────────────────────────────────────

class TestNetworkAnalysis(unittest.TestCase):

    def _sample_edges(self):
        return [
            {"source": "user_A", "target": "user_B", "weight": 3.0},
            {"source": "user_A", "target": "user_C", "weight": 1.0},
            {"source": "user_B", "target": "user_C", "weight": 2.0},
            {"source": "user_C", "target": "user_D", "weight": 1.0},
            {"source": "user_D", "target": "user_E", "weight": 1.0},
            {"source": "user_B", "target": "user_E", "weight": 1.0},
        ]

    def test_graph_construction(self):
        """Graph should have correct node/edge counts."""
        analyzer = NetworkAnalyzer()
        G = analyzer.build_graph(self._sample_edges())
        self.assertEqual(G.number_of_nodes(), 5)
        self.assertEqual(G.number_of_edges(), 6)

    def test_pagerank_returns_all_nodes(self):
        """PageRank should return a score for every node."""
        analyzer = NetworkAnalyzer()
        G = analyzer.build_graph(self._sample_edges())
        pr = analyzer.compute_pagerank(G)
        self.assertEqual(set(pr.keys()), {"user_A", "user_B", "user_C", "user_D", "user_E"})

    def test_pagerank_sums_to_one(self):
        """PageRank scores should sum approximately to 1.0."""
        analyzer = NetworkAnalyzer()
        G = analyzer.build_graph(self._sample_edges())
        pr = analyzer.compute_pagerank(G)
        self.assertAlmostEqual(sum(pr.values()), 1.0, places=4)

    def test_betweenness_centrality(self):
        """Betweenness centrality: node B or C should rank high (bridge node)."""
        analyzer = NetworkAnalyzer()
        G = analyzer.build_graph(self._sample_edges())
        bw = analyzer.compute_betweenness_centrality(G)
        # user_C is a bridge (A→C, B→C→D→E), expect high betweenness
        top_node = max(bw, key=bw.get)
        self.assertIn(top_node, {"user_B", "user_C", "user_D"})

    def test_kol_identification(self):
        """KOL list should be non-empty and correctly sorted by influence."""
        analyzer = NetworkAnalyzer()
        G = analyzer.build_graph(self._sample_edges())
        pr = analyzer.compute_pagerank(G)
        bw = analyzer.compute_betweenness_centrality(G)
        kols = analyzer.identify_kols(pr, bw, top_n=3)
        self.assertLessEqual(len(kols), 3)
        # Influence scores should be descending
        scores = [k["influence_score"] for k in kols]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_narrative_propagation_trace(self):
        """Propagation trace from user_A should reach user_E within 5 hops."""
        analyzer = NetworkAnalyzer()
        G = analyzer.build_graph(self._sample_edges())
        result = analyzer.trace_narrative_propagation(G, "user_A", max_hops=5)
        self.assertEqual(result["source"], "user_A")
        self.assertIn("user_E", result["reachable_nodes"])
        self.assertGreater(len(result["clusters"]), 0)

    def test_empty_graph(self):
        """Empty edge list should return empty metrics without error."""
        analyzer = NetworkAnalyzer()
        G = analyzer.build_graph([])
        pr = analyzer.compute_pagerank(G)
        bw = analyzer.compute_betweenness_centrality(G)
        self.assertEqual(pr, {})
        self.assertEqual(bw, {})


# ──────────────────────────────────────────────────────────────────
# Bot Detection Tests
# ──────────────────────────────────────────────────────────────────

class TestBotDetection(unittest.TestCase):

    def _human_user(self, uid="human_1"):
        return {
            "hashed_user_id": _sha256(uid),
            "posts_per_hour": 1.5,
            "repetition_rate": 0.05,
            "unique_recipients": 12,
            "avg_response_time_sec": 320.0,
            "follower_following_ratio": 1.2,
            "account_age_days": 730.0,
            "burst_flag": 0,
        }

    def _bot_user(self, uid="bot_99"):
        return {
            "hashed_user_id": _sha256(uid),
            "posts_per_hour": 45.0,
            "repetition_rate": 0.92,
            "unique_recipients": 1,
            "avg_response_time_sec": 2.0,
            "follower_following_ratio": 80.0,
            "account_age_days": 5.0,
            "burst_flag": 1,
        }

    def test_human_user_low_bot_score(self):
        """Normal user should have a low bot score."""
        detector = BotDetector()
        result = detector.score_user(self._human_user())
        self.assertLess(result["bot_score"], 0.6)

    def test_bot_user_high_bot_score(self):
        """Bot-like user should have a high bot score and be flagged."""
        population = [self._human_user(f"h{i}") for i in range(8)] + [self._bot_user()]
        detector = BotDetector()
        detector.fit(population)
        result = detector.score_user(self._bot_user())
        self.assertGreaterEqual(result["bot_score"], 0.6)
        self.assertTrue(result["is_bot"])

    def test_anomaly_flags_populated(self):
        """Bot-like user should trigger multiple anomaly flags."""
        detector = BotDetector()
        result = detector.score_user(self._bot_user())
        self.assertGreater(len(result["anomaly_flags"]), 0)

    def test_detect_bots_batch(self):
        """detect_bots() should correctly label bots and humans in a batch."""
        logs = [self._human_user(f"h{i}") for i in range(5)] + [self._bot_user("bot_x")]
        results = detect_bots(logs)

        self.assertEqual(len(results), 6)
        bot_uid = _sha256("bot_x")
        self.assertIn(bot_uid, results)
        self.assertTrue(results[bot_uid]["is_bot"])

    def test_detect_bots_empty_input(self):
        """Empty input should return empty dict without crashing."""
        results = detect_bots([])
        self.assertEqual(results, {})

    def test_burst_flag_triggers_anomaly(self):
        """Burst posting should be flagged as anomaly."""
        user = self._human_user()
        user["burst_flag"] = 1
        detector = BotDetector()
        result = detector.score_user(user)
        self.assertIn("BURST_POSTING_DETECTED", result["anomaly_flags"])


# ──────────────────────────────────────────────────────────────────
# Neo4j Client Tests (Mocked)
# ──────────────────────────────────────────────────────────────────

class TestNeo4jClient(unittest.TestCase):

    def test_upsert_user_runs_cypher(self):
        """upsert_user should run a MERGE cypher statement."""
        client = Neo4jClient(uri="bolt://localhost:7687")
        mock_session = MagicMock()
        with patch.object(client, "session") as mock_ctx:
            mock_ctx.return_value.__enter__ = lambda s: mock_session
            mock_ctx.return_value.__exit__ = MagicMock(return_value=False)
            client.upsert_user("a" * 64, "telegram")
            mock_session.run.assert_called_once()
            query = mock_session.run.call_args[0][0]
            self.assertIn("MERGE (u:User", query)

    def test_link_user_posted_cypher(self):
        """link_user_posted should create a POSTED edge."""
        client = Neo4jClient()
        mock_session = MagicMock()
        with patch.object(client, "session") as mock_ctx:
            mock_ctx.return_value.__enter__ = lambda s: mock_session
            mock_ctx.return_value.__exit__ = MagicMock(return_value=False)
            client.link_user_posted("a" * 64, "post_001", "2026-08-29")
            query = mock_session.run.call_args[0][0]
            self.assertIn("POSTED", query)


# ──────────────────────────────────────────────────────────────────
# Integration: Full Job with Mock Posts
# ──────────────────────────────────────────────────────────────────

class TestNetworkAndBotJob(unittest.TestCase):

    def test_job_with_mock_data(self):
        """Job should complete without error on mock posts and return analysis results."""
        mock_ch = MagicMock()
        mock_neo4j = MagicMock()
        job = NetworkAndBotAnalyzerJob(hours=2.0, clickhouse=mock_ch, neo4j=mock_neo4j)

        mock_posts = _generate_mock_posts()
        result = job.run_once(mock_posts=mock_posts)

        self.assertIn("network", result)
        self.assertIn("bot_results", result)
        self.assertIn("kols", result)
        # Bot suspect should be flagged
        bot_uid = _sha256("bot_suspect_99")
        self.assertIn(bot_uid, result["bot_results"])
        self.assertTrue(result["bot_results"][bot_uid]["is_bot"])

    def test_bot_count_detected(self):
        """At least one bot should be detected in mock data."""
        mock_ch = MagicMock()
        mock_neo4j = MagicMock()
        job = NetworkAndBotAnalyzerJob(hours=2.0, clickhouse=mock_ch, neo4j=mock_neo4j)
        result = job.run_once(mock_posts=_generate_mock_posts())
        self.assertGreater(result["bot_count"], 0)

    def test_empty_posts_returns_empty(self):
        """Empty post list should return empty dict without error."""
        mock_ch = MagicMock()
        mock_neo4j = MagicMock()
        job = NetworkAndBotAnalyzerJob(hours=2.0, clickhouse=mock_ch, neo4j=mock_neo4j)
        result = job.run_once(mock_posts=[])
        self.assertEqual(result, {})


if __name__ == "__main__":
    unittest.main()
