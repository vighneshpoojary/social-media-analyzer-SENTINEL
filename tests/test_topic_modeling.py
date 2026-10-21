"""Unit tests for Real-Time Trend & Topic Modeling Engine and ClickHouse Topic Storage."""

import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from src.jobs.trend_analyzer import TrendAnalyzerJob
from src.nlp.topic_modeling import (
    TopicModeler,
    extract_ngrams,
    extract_trending_topics,
    preprocess_text,
)
from src.storage.clickhouse_client import ClickHouseStorageClient


class TestTopicModeling(unittest.TestCase):
    def setUp(self):
        self.sample_posts = [
            {"raw_text": "Massive cybersecurity breach detected. Vulnerability in firewall software exploited by hackers."},
            {"raw_text": "Emergency patch deployed for zero-day cybersecurity vulnerability in core router network."},
            {"raw_text": "Ransomware gang claims responsibility for cyber attack on critical telecom servers."},
            {"raw_text": "Cybersecurity defense teams isolate infected endpoints following network breach."},
            {"raw_text": "National defence forces conduct joint maritime patrol exercises in regional waters."},
            {"raw_text": "Strategic missile defence systems deployed along northern border sectors."},
            {"raw_text": "Joint naval fleet drill concludes with high operational readiness."},
            {"raw_text": "Central bank cuts benchmark interest rates to stimulate economic growth."},
            {"raw_text": "Stock market index hits record high fueled by economic optimism and tech earnings."},
            {"raw_text": "Foreign direct investment reaches new quarterly high across manufacturing sectors."},
        ]

    def test_preprocess_and_ngram_extraction(self):
        """Test text cleaning and n-gram generation."""
        raw = "Breaking: Check https://example.com/news @admin Urgent alert on cyber defense!"
        cleaned = preprocess_text(raw)
        self.assertNotIn("https://", cleaned)
        self.assertNotIn("@admin", cleaned)
        self.assertIn("urgent", cleaned)
        self.assertIn("defense", cleaned)

        unigrams = extract_ngrams(cleaned, n=1)
        self.assertIn("defense", unigrams)

        bigrams = extract_ngrams(cleaned, n=2)
        self.assertIn("cyber defense", bigrams)

    def test_topic_extraction_clusters(self):
        """Verify cluster formation and keyword extraction across distinct domains."""
        topics = extract_trending_topics(self.sample_posts, top_n=5, time_window_hours=2.0)

        self.assertIsInstance(topics, list)
        self.assertGreater(len(topics), 0)

        for t in topics:
            self.assertIn("topic_id", t)
            self.assertIn("label", t)
            self.assertIn("keywords", t)
            self.assertIn("post_count", t)
            self.assertIn("velocity_score", t)

            self.assertIsInstance(t["topic_id"], int)
            self.assertIsInstance(t["label"], str)
            self.assertIsInstance(t["keywords"], list)
            self.assertIsInstance(t["post_count"], int)
            self.assertIsInstance(t["velocity_score"], float)

            self.assertGreater(t["post_count"], 0)
            self.assertGreater(t["velocity_score"], 0.0)
            self.assertGreater(len(t["keywords"]), 0)

        # Check that cybersecurity or defence appears among top topic keywords
        all_extracted_keywords = [k.lower() for t in topics for k in t["keywords"]]
        has_security_or_defense = any(
            w in all_extracted_keywords
            for w in ["cybersecurity", "vulnerability", "breach", "defence", "economic", "rates"]
        )
        self.assertTrue(has_security_or_defense)

    def test_empty_and_small_posts(self):
        """Verify graceful handling for empty or low-signal post lists."""
        empty_topics = extract_trending_topics([])
        self.assertEqual(empty_topics, [])

        single_post = [{"raw_text": "Single announcement about satellite launch."}]
        single_topic = extract_trending_topics(single_post, top_n=3)
        self.assertIsInstance(single_topic, list)


class TestTopicStorage(unittest.TestCase):
    def setUp(self):
        self.client = ClickHouseStorageClient(
            host="localhost", port=9000, user="default", password="", database="ntro_analytics"
        )

    def test_trending_topics_table_init(self):
        """Verify trending_topics table create query."""
        with patch.object(self.client, "get_client") as mock_get_client:
            mock_ch = MagicMock()
            mock_get_client.return_value = mock_ch

            self.client.init_tables()

            calls = [call[0][0] for call in mock_ch.execute.call_args_list]
            topic_query = [c for c in calls if "CREATE TABLE IF NOT EXISTS ntro_analytics.trending_topics" in c][0]

            self.assertIn("topic_id Int32", topic_query)
            self.assertIn("label String", topic_query)
            self.assertIn("keywords Array(String)", topic_query)
            self.assertIn("post_count UInt32", topic_query)
            self.assertIn("velocity_score Float32", topic_query)

    def test_insert_trending_topics(self):
        """Verify batch insertion of trending topics."""
        with patch.object(self.client, "get_client") as mock_get_client:
            mock_ch = MagicMock()
            mock_get_client.return_value = mock_ch

            topics = [
                {
                    "timestamp": "2026-08-29T10:00:00Z",
                    "topic_id": 1,
                    "label": "Cybersecurity Vulnerability",
                    "keywords": ["cybersecurity", "vulnerability", "zero-day"],
                    "post_count": 25,
                    "velocity_score": 12.5,
                }
            ]

            count = self.client.insert_trending_topics(topics)
            self.assertEqual(count, 1)
            mock_ch.execute.assert_called()

            call_args = mock_ch.execute.call_args
            query = call_args[0][0]
            data = call_args[0][1]

            self.assertIn("INSERT INTO ntro_analytics.trending_topics", query)
            self.assertEqual(data[0][1], 1)
            self.assertEqual(data[0][2], "Cybersecurity Vulnerability")
            self.assertEqual(data[0][3], ["cybersecurity", "vulnerability", "zero-day"])
            self.assertEqual(data[0][4], 25)
            self.assertEqual(data[0][5], 12.5)


class TestTrendAnalyzerJob(unittest.TestCase):
    def test_trend_analyzer_run_once(self):
        """Test end-to-end execution of TrendAnalyzerJob."""
        mock_storage = MagicMock()
        mock_storage.check_connection.return_value = True

        mock_posts = [
            {"raw_text": "Cyber attack targets critical telecommunication infrastructure."},
            {"raw_text": "Emergency patch deployed for zero-day cyber attack."},
            {"raw_text": "Telecom infrastructure restored following cyber security intervention."},
        ]

        job = TrendAnalyzerJob(hours_window=2.0, top_n=3, clickhouse_client=mock_storage)
        topics = job.run_once(mock_posts=mock_posts)

        self.assertIsInstance(topics, list)
        self.assertGreater(len(topics), 0)
        self.assertIn("topic_id", topics[0])
        self.assertIn("velocity_score", topics[0])


if __name__ == "__main__":
    unittest.main()
