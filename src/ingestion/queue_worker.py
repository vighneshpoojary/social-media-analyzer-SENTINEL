"""RabbitMQ Queue Worker with SHA-256 User Anonymization, NLP Emotion Inference, and ClickHouse Batch Ingestion."""

import asyncio
import hashlib
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import pika
from dotenv import load_dotenv

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.nlp.sentiment import analyze_emotion
from src.storage.clickhouse_client import ClickHouseStorageClient

load_dotenv()

logger = logging.getLogger("QueueWorker")


def anonymize_user_id(user_id: Any) -> str:
    """
    Hash user identifiers with SHA-256 to ensure complete privacy and anonymization.
    """
    raw_str = str(user_id) if user_id is not None else "unknown"
    return hashlib.sha256(raw_str.encode("utf-8")).hexdigest()


class QueueWorker:
    """
    Consumes social media posts from RabbitMQ, anonymizes user IDs,
    runs multi-dimensional emotion inference, and batch inserts records
    into ClickHouse every 10 posts or 5 seconds.
    """

    def __init__(
        self,
        rabbitmq_host: Optional[str] = None,
        rabbitmq_port: Optional[int] = None,
        rabbitmq_user: Optional[str] = None,
        rabbitmq_password: Optional[str] = None,
        queue_name: str = "social_media_queue",
        batch_size: int = 10,
        flush_interval: float = 5.0,
        clickhouse_client: Optional[ClickHouseStorageClient] = None,
    ) -> None:
        self.host = rabbitmq_host or os.getenv("RABBITMQ_HOST", "localhost")
        self.port = int(rabbitmq_port or os.getenv("RABBITMQ_PORT", "5672"))
        self.user = rabbitmq_user or os.getenv("RABBITMQ_USER", "guest")
        self.password = rabbitmq_password if rabbitmq_password is not None else os.getenv("RABBITMQ_PASSWORD", "guest")
        self.queue_name = queue_name
        self.batch_size = batch_size
        self.flush_interval = flush_interval

        self.storage = clickhouse_client or ClickHouseStorageClient()
        self._connection: Optional[pika.BlockingConnection] = None
        self._channel = None
        self._is_running = False

        # In-memory buffer: list of tuples (delivery_tag, processed_payload_dict)
        self._batch_buffer: List[Tuple[int, Dict[str, Any]]] = []
        self._last_flush_time: float = time.time()

    def connect(self) -> bool:
        """Establish connection to RabbitMQ."""
        try:
            credentials = pika.PlainCredentials(self.user, self.password)
            parameters = pika.ConnectionParameters(
                host=self.host,
                port=self.port,
                credentials=credentials,
                connection_attempts=3,
                retry_delay=2,
                socket_timeout=10,
            )
            self._connection = pika.BlockingConnection(parameters)
            self._channel = self._connection.channel()
            self._channel.queue_declare(queue=self.queue_name, durable=True)
            # Set prefetch count to batch_size * 2 for efficient throughput
            self._channel.basic_qos(prefetch_count=self.batch_size * 2)
            logger.info(f"QueueWorker connected to RabbitMQ '{self.queue_name}' at {self.host}:{self.port}")
            return True
        except Exception as e:
            logger.warning(f"QueueWorker failed to connect to RabbitMQ: {e}")
            self._connection = None
            self._channel = None
            return False

    def process_message(self, body: bytes) -> Optional[Dict[str, Any]]:
        """
        Parse raw JSON message:
        1. Anonymize user_id with SHA-256.
        2. Run emotion inference on raw_text (Sarcasm, Anxiety, Excitement, Support, Opposition).
        3. Return unified enriched payload.
        """
        try:
            payload = json.loads(body.decode("utf-8"))
            raw_user_id = payload.get("user_id", "unknown")
            payload["user_id"] = anonymize_user_id(raw_user_id)

            raw_text = payload.get("raw_text", "")
            emotion_scores = analyze_emotion(raw_text)
            payload.update(emotion_scores)

            logger.debug(
                f"Enriched post from platform '{payload.get('platform')}' with emotion scores: {emotion_scores}"
            )
            return payload
        except Exception as e:
            logger.error(f"Failed to parse, anonymize, or analyze payload: {e}")
            return None

    def flush_batch(self) -> int:
        """Flush buffered messages to ClickHouse and ACK in RabbitMQ."""
        if not self._batch_buffer:
            self._last_flush_time = time.time()
            return 0

        batch_to_insert = [item[1] for item in self._batch_buffer]
        highest_delivery_tag = self._batch_buffer[-1][0]
        count = len(batch_to_insert)

        try:
            # Batch insert into ClickHouse
            self.storage.insert_posts_batch(batch_to_insert)
            logger.info(f"Successfully flushed {count} posts to ClickHouse 'social_posts'.")

            # Acknowledge messages in RabbitMQ
            if self._channel and not self._channel.is_closed:
                self._channel.basic_ack(delivery_tag=highest_delivery_tag, multiple=True)

            self._batch_buffer.clear()
            self._last_flush_time = time.time()
            return count
        except Exception as e:
            logger.error(f"Failed to flush batch to ClickHouse: {e}")
            # Keep buffer to retry next round
            return 0

    def run_sync(self, max_batches: Optional[int] = None) -> None:
        """Synchronously poll and consume messages with batch timeout handling."""
        if not self.connect():
            logger.error("Cannot start QueueWorker: RabbitMQ unavailable.")
            return

        self._is_running = True
        self._last_flush_time = time.time()
        batches_flushed = 0

        logger.info(
            f"QueueWorker running (batch_size={self.batch_size}, flush_interval={self.flush_interval}s)..."
        )

        try:
            while self._is_running:
                # Non-blocking fetch of message
                method_frame, properties, body = self._channel.basic_get(queue=self.queue_name, auto_ack=False)
                now = time.time()

                if method_frame:
                    payload = self.process_message(body)
                    if payload:
                        self._batch_buffer.append((method_frame.delivery_tag, payload))

                    if len(self._batch_buffer) >= self.batch_size:
                        flushed = self.flush_batch()
                        if flushed > 0:
                            batches_flushed += 1
                else:
                    # No message available in this tick
                    if self._batch_buffer and (now - self._last_flush_time >= self.flush_interval):
                        flushed = self.flush_batch()
                        if flushed > 0:
                            batches_flushed += 1
                    
                    time.sleep(0.2)  # Prevent CPU spinning when idle

                if max_batches and batches_flushed >= max_batches:
                    break

        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received in QueueWorker.")
        finally:
            self.flush_batch()
            self.stop()

    async def start(self) -> None:
        """Asynchronous entry point for running QueueWorker within an asyncio loop."""
        loop = asyncio.get_running_loop()
        self._is_running = True
        logger.info("Starting QueueWorker in async background executor...")
        try:
            await loop.run_in_executor(None, self.run_sync)
        except asyncio.CancelledError:
            logger.info("QueueWorker async task cancelled.")
        finally:
            self.stop()

    def stop(self) -> None:
        """Stop worker and close connections."""
        self._is_running = False
        try:
            if self._batch_buffer:
                self.flush_batch()
            if self._channel and not self._channel.is_closed:
                self._channel.close()
            if self._connection and not self._connection.is_closed:
                self._connection.close()
            logger.info("QueueWorker stopped.")
        except Exception as e:
            logger.warning(f"Error closing QueueWorker connection: {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    worker = QueueWorker()
    worker.run_sync()
