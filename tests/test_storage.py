"""Unit tests for ClickHouse storage client and schema validation with emotion scores."""

import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
from src.storage.clickhouse_client import ClickHouseStorageClient


class TestClickHouseStorage(unittest.TestCase):
    def setUp(self):
        self.client = ClickHouseStorageClient(
            host="localhost",
            port=9000,
            user="default",
            password="",
            database="ntro_analytics"
        )

    def test_schema_definition(self):
        """Verify table creation SQL contains all required fields including emotion scores."""
        with patch.object(self.client, "get_client") as mock_get_client:
            mock_ch = MagicMock()
            mock_get_client.return_value = mock_ch

            self.client.init_tables()

            # Verify database creation
            mock_ch.execute.assert_any_call("CREATE DATABASE IF NOT EXISTS ntro_analytics")

            # Check table create query contains all required fields
            calls = [call[0][0] for call in mock_ch.execute.call_args_list]
            create_query = [c for c in calls if "CREATE TABLE IF NOT EXISTS" in c][0]
            
            self.assertIn("timestamp DateTime", create_query)
            self.assertIn("platform LowCardinality(String)", create_query)
            self.assertIn("user_id String", create_query)
            self.assertIn("raw_text String", create_query)
            self.assertIn("post_type LowCardinality(String)", create_query)
            self.assertIn("parent_id Nullable(String)", create_query)
            self.assertIn("sarcasm_score Float32", create_query)
            self.assertIn("anxiety_score Float32", create_query)
            self.assertIn("excitement_score Float32", create_query)
            self.assertIn("support_score Float32", create_query)
            self.assertIn("opposition_score Float32", create_query)
            self.assertIn("ENGINE = MergeTree()", create_query)

    def test_insert_post(self):
        """Test single post insertion with emotion scores."""
        with patch.object(self.client, "get_client") as mock_get_client:
            mock_ch = MagicMock()
            mock_get_client.return_value = mock_ch

            ts = datetime.now(timezone.utc)
            self.client.insert_post(
                timestamp=ts,
                platform="telegram",
                user_id="user_123",
                raw_text="Test social media post",
                post_type="channel_post",
                parent_id="msg_999",
                sarcasm_score=0.1,
                anxiety_score=0.2,
                excitement_score=0.8,
                support_score=0.7,
                opposition_score=0.05,
            )

            mock_ch.execute.assert_called()
            call_args = mock_ch.execute.call_args
            query = call_args[0][0]
            data = call_args[0][1]

            self.assertIn("INSERT INTO ntro_analytics.social_posts", query)
            self.assertEqual(data, [(
                ts, "telegram", "user_123", "Test social media post", "channel_post", "msg_999",
                0.1, 0.2, 0.8, 0.7, 0.05
            )])

    def test_batch_insert(self):
        """Test batch post insertion with emotion scores."""
        with patch.object(self.client, "get_client") as mock_get_client:
            mock_ch = MagicMock()
            mock_get_client.return_value = mock_ch

            posts = [
                {
                    "timestamp": "2026-08-27T12:00:00Z",
                    "platform": "twitter",
                    "user_id": "user_abc",
                    "raw_text": "Sample tweet #1",
                    "post_type": "tweet",
                    "parent_id": None,
                    "sarcasm_score": 0.0,
                    "anxiety_score": 0.1,
                    "excitement_score": 0.9,
                    "support_score": 0.85,
                    "opposition_score": 0.05,
                },
                {
                    "timestamp": "2026-08-27T12:05:00Z",
                    "platform": "telegram",
                    "user_id": "user_xyz",
                    "raw_text": "Sample message #2",
                    "post_type": "reply",
                    "parent_id": "12345",
                    "sarcasm_score": 0.5,
                    "anxiety_score": 0.0,
                    "excitement_score": 0.1,
                    "support_score": 0.2,
                    "opposition_score": 0.6,
                }
            ]

            count = self.client.insert_posts_batch(posts)
            self.assertEqual(count, 2)
            mock_ch.execute.assert_called()


if __name__ == "__main__":
    unittest.main()
