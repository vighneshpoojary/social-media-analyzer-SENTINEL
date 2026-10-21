"""Unit tests for Data Ingestion, RabbitMQ Queue Worker, and Formatters."""

import hashlib
import json
import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from src.ingestion.queue_worker import QueueWorker, anonymize_user_id
from src.ingestion.rabbitmq_producer import RabbitMQProducer
from src.ingestion.telegram_listener import TelegramListener, format_telegram_payload
from src.ingestion.x_scraper import XScraper, format_tweet_payload


class TestPayloadFormatters(unittest.TestCase):
    def test_anonymize_user_id(self):
        """Verify SHA-256 user anonymization."""
        uid = "user_secret_12345"
        hashed = anonymize_user_id(uid)
        
        expected = hashlib.sha256(uid.encode("utf-8")).hexdigest()
        self.assertEqual(hashed, expected)
        self.assertEqual(len(hashed), 64)
        # Verify non-reversibility property
        self.assertNotEqual(hashed, uid)

    def test_format_telegram_payload_basic(self):
        """Test formatting standard Telegram channel message."""
        mock_msg = MagicMock()
        mock_msg.date = datetime(2026, 8, 28, 10, 30, 0, tzinfo=timezone.utc)
        mock_msg.sender_id = 987654321
        mock_msg.message = "Urgent news announcement"
        mock_msg.text = "Urgent news announcement"
        mock_msg.reply_to = None
        mock_msg.reply_to_msg_id = None
        mock_msg.fwd_from = None

        payload = format_telegram_payload(mock_msg)
        self.assertEqual(payload["platform"], "telegram")
        self.assertEqual(payload["user_id"], "987654321")
        self.assertEqual(payload["raw_text"], "Urgent news announcement")
        self.assertEqual(payload["post_type"], "channel_post")
        self.assertIsNone(payload["parent_id"])
        self.assertIn("2026-08-28", payload["timestamp"])

    def test_format_telegram_payload_reply(self):
        """Test formatting Telegram reply message."""
        mock_msg = MagicMock()
        mock_msg.date = datetime(2026, 8, 28, 10, 31, 0, tzinfo=timezone.utc)
        mock_msg.sender_id = 111222333
        mock_msg.message = "This is a reply to the post"
        mock_msg.text = "This is a reply to the post"
        mock_msg.reply_to = MagicMock()
        mock_msg.reply_to.reply_to_msg_id = 55500
        mock_msg.fwd_from = None

        payload = format_telegram_payload(mock_msg)
        self.assertEqual(payload["platform"], "telegram")
        self.assertEqual(payload["post_type"], "reply")
        self.assertEqual(payload["parent_id"], "55500")

    def test_format_tweet_payload_basic(self):
        """Test formatting standard Twitter / X tweet."""
        mock_tweet = MagicMock()
        mock_tweet.created_at = datetime(2026, 8, 28, 11, 0, 0, tzinfo=timezone.utc)
        mock_tweet.id = 1234567890
        mock_tweet.text = "Breaking analysis on tech trends"
        mock_tweet.full_text = "Breaking analysis on tech trends"
        mock_tweet.user = MagicMock()
        mock_tweet.user.id = "user_twitter_42"
        mock_tweet.is_retweet = False
        mock_tweet.is_quote = False
        mock_tweet.retweeted_tweet = None
        mock_tweet.quoted_tweet = None
        mock_tweet.reply_to = None
        mock_tweet.in_reply_to = None

        payload = format_tweet_payload(mock_tweet)
        self.assertEqual(payload["platform"], "twitter")
        self.assertEqual(payload["user_id"], "user_twitter_42")
        self.assertEqual(payload["raw_text"], "Breaking analysis on tech trends")
        self.assertEqual(payload["post_type"], "tweet")
        self.assertIsNone(payload["parent_id"])

    def test_format_tweet_payload_retweet(self):
        """Test formatting Twitter retweet."""
        mock_tweet = MagicMock()
        mock_tweet.created_at = "2026-08-28T11:05:00Z"
        mock_tweet.text = "RT @source: Important alert"
        mock_tweet.full_text = "RT @source: Important alert"
        mock_tweet.user = MagicMock()
        mock_tweet.user.id = "user_rt_99"
        mock_tweet.is_retweet = True
        mock_tweet.retweeted_tweet = MagicMock()
        mock_tweet.retweeted_tweet.id = 99887766
        mock_tweet.is_quote = False
        mock_tweet.reply_to = None
        mock_tweet.in_reply_to = None

        payload = format_tweet_payload(mock_tweet)
        self.assertEqual(payload["platform"], "twitter")
        self.assertEqual(payload["post_type"], "retweet")
        self.assertEqual(payload["parent_id"], "99887766")

    def test_format_tweet_payload_reply(self):
        """Test formatting Twitter reply."""
        mock_tweet = MagicMock()
        mock_tweet.created_at = datetime(2026, 8, 28, 11, 10, 0, tzinfo=timezone.utc)
        mock_tweet.text = "Agreed with this point"
        mock_tweet.full_text = "Agreed with this point"
        mock_tweet.user = MagicMock()
        mock_tweet.user.id = "user_reply_88"
        mock_tweet.is_retweet = False
        mock_tweet.is_quote = False
        mock_tweet.retweeted_tweet = None
        mock_tweet.quoted_tweet = None
        mock_tweet.reply_to = "target_tweet_1001"
        mock_tweet.in_reply_to = None

        payload = format_tweet_payload(mock_tweet)
        self.assertEqual(payload["platform"], "twitter")
        self.assertEqual(payload["post_type"], "reply")
        self.assertEqual(payload["parent_id"], "target_tweet_1001")


class TestRabbitMQProducer(unittest.TestCase):
    @patch("pika.BlockingConnection")
    def test_publish_post(self, mock_connection_cls):
        """Test publishing standardized post payload."""
        mock_conn = MagicMock()
        mock_channel = MagicMock()
        mock_connection_cls.return_value = mock_conn
        mock_conn.channel.return_value = mock_channel
        mock_channel.is_closed = False
        mock_conn.is_closed = False

        producer = RabbitMQProducer(queue_name="social_media_queue")
        producer.connect()

        payload = {
            "timestamp": "2026-08-28T12:00:00Z",
            "platform": "telegram",
            "user_id": "test_user",
            "raw_text": "Sample text",
            "post_type": "channel_post",
            "parent_id": None,
        }

        success = producer.publish_post(payload)
        self.assertTrue(success)
        mock_channel.basic_publish.assert_called_once()
        call_kwargs = mock_channel.basic_publish.call_args[1]
        self.assertEqual(call_kwargs["routing_key"], "social_media_queue")
        
        sent_body = json.loads(call_kwargs["body"].decode("utf-8"))
        self.assertEqual(sent_body["user_id"], "test_user")


class TestQueueWorker(unittest.TestCase):
    def setUp(self):
        self.mock_storage = MagicMock()
        self.worker = QueueWorker(
            batch_size=10,
            flush_interval=5.0,
            clickhouse_client=self.mock_storage,
        )

    def test_process_message_anonymization(self):
        """Verify process_message parses payload and hashes user_id with SHA-256."""
        raw_payload = {
            "timestamp": "2026-08-28T12:00:00Z",
            "platform": "telegram",
            "user_id": "secret_agent_007",
            "raw_text": "Classified bulletin",
            "post_type": "channel_post",
            "parent_id": None,
        }
        raw_bytes = json.dumps(raw_payload).encode("utf-8")
        processed = self.worker.process_message(raw_bytes)

        self.assertIsNotNone(processed)
        expected_hash = hashlib.sha256("secret_agent_007".encode("utf-8")).hexdigest()
        self.assertEqual(processed["user_id"], expected_hash)
        self.assertEqual(processed["raw_text"], "Classified bulletin")
        self.assertIn("sarcasm_score", processed)
        self.assertIn("anxiety_score", processed)
        self.assertIn("excitement_score", processed)
        self.assertIn("support_score", processed)
        self.assertIn("opposition_score", processed)

    def test_batch_flush_at_batch_size(self):
        """Verify batch flushes to ClickHouse when batch_size is reached."""
        self.worker._channel = MagicMock()
        self.worker._channel.is_closed = False

        # Simulate 10 messages arriving
        for i in range(10):
            payload = {
                "timestamp": "2026-08-28T12:00:00Z",
                "platform": "twitter",
                "user_id": anonymize_user_id(f"user_{i}"),
                "raw_text": f"Tweet #{i}",
                "post_type": "tweet",
                "parent_id": None,
            }
            self.worker._batch_buffer.append((i + 1, payload))

        self.assertEqual(len(self.worker._batch_buffer), 10)
        flushed_count = self.worker.flush_batch()

        self.assertEqual(flushed_count, 10)
        self.mock_storage.insert_posts_batch.assert_called_once()
        inserted_records = self.mock_storage.insert_posts_batch.call_args[0][0]
        self.assertEqual(len(inserted_records), 10)
        self.worker._channel.basic_ack.assert_called_once_with(delivery_tag=10, multiple=True)
        self.assertEqual(len(self.worker._batch_buffer), 0)

    def test_batch_flush_timeout(self):
        """Verify partial batch flushes after flush_interval."""
        self.worker._channel = MagicMock()
        self.worker._channel.is_closed = False

        # Add 3 items (under batch_size of 10)
        for i in range(3):
            payload = {
                "timestamp": "2026-08-28T12:00:00Z",
                "platform": "telegram",
                "user_id": anonymize_user_id(f"user_{i}"),
                "raw_text": f"Post #{i}",
                "post_type": "channel_post",
                "parent_id": None,
            }
            self.worker._batch_buffer.append((i + 1, payload))

        flushed = self.worker.flush_batch()
        self.assertEqual(flushed, 3)
        self.mock_storage.insert_posts_batch.assert_called_once()
        self.worker._channel.basic_ack.assert_called_once_with(delivery_tag=3, multiple=True)
        self.assertEqual(len(self.worker._batch_buffer), 0)


if __name__ == "__main__":
    unittest.main()
