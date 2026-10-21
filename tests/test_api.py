"""Integration and unit tests for FastAPI Analytics Server and Endpoint Routing."""

import unittest
from fastapi.testclient import TestClient
from src.api.server import app


class TestFastAPIEndpoints(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_root_endpoint(self):
        """Root status endpoint should return status online."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "online")

    def test_stats_summary(self):
        """Stats summary endpoint should return summary indicators and counts."""
        response = self.client.get("/api/stats/summary")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("total_posts_ingested", data)
        self.assertIn("active_platforms", data)
        self.assertIn("services", data)

    def test_feed_live(self):
        """Live feed endpoint should return chronological posts with sentiment scores."""
        response = self.client.get("/api/feed/live?limit=5")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("feed", data)
        self.assertGreater(len(data["feed"]), 0)
        sample = data["feed"][0]
        self.assertIn("user_id", sample)
        self.assertIn("platform", sample)
        self.assertIn("raw_text", sample)
        self.assertIn("sarcasm_score", sample)

    def test_sentiment_timeline(self):
        """Sentiment timeline should return time-window statistics for emotion categories."""
        response = self.client.get("/api/sentiment/timeline?hours=6")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("timeline", data)
        self.assertGreater(len(data["timeline"]), 0)
        sample = data["timeline"][0]
        self.assertIn("timestamp", sample)
        self.assertIn("sarcasm_score", sample)
        self.assertIn("anxiety_score", sample)

    def test_trends_top(self):
        """Trends endpoint should return top trending topics with velocity scores."""
        response = self.client.get("/api/trends/top?limit=3")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("topics", data)
        self.assertLessEqual(len(data["topics"]), 3)

    def test_demographics_summary(self):
        """Demographics summary should return age, role, and location allocations."""
        response = self.client.get("/api/demographics/summary")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("age_brackets", data)
        self.assertIn("professional_roles", data)
        self.assertIn("locations", data)

    def test_network_graph(self):
        """Network graph endpoint should return node topology and interaction links."""
        response = self.client.get("/api/network/graph")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("nodes", data)
        self.assertIn("edges", data)
        self.assertGreater(len(data["nodes"]), 0)


if __name__ == "__main__":
    unittest.main()
