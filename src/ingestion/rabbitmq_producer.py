"""RabbitMQ Producer Utility for Social Media Ingestion."""

import json
import logging
import os
from typing import Any, Dict, Optional
import pika
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("RabbitMQProducer")


class RabbitMQProducer:
    """Handles connection and message publishing to RabbitMQ queues."""

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        queue_name: str = "social_media_queue",
    ) -> None:
        self.host = host or os.getenv("RABBITMQ_HOST", "localhost")
        self.port = int(port or os.getenv("RABBITMQ_PORT", "5672"))
        self.user = user or os.getenv("RABBITMQ_USER", "guest")
        self.password = password if password is not None else os.getenv("RABBITMQ_PASSWORD", "guest")
        self.queue_name = queue_name
        self._connection = None
        self._channel = None

    def connect(self) -> bool:
        """Establish connection and declare durable queue."""
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
            logger.info(f"Connected to RabbitMQ at {self.host}:{self.port}, queue: '{self.queue_name}'")
            return True
        except Exception as e:
            logger.warning(f"Failed to connect to RabbitMQ ({self.host}:{self.port}): {e}")
            self._connection = None
            self._channel = None
            return False

    def publish_post(self, payload: Dict[str, Any]) -> bool:
        """
        Publish a standardized social media post payload to the queue.

        Payload schema:
        - timestamp: str (ISO format)
        - platform: str ('telegram', 'twitter', etc.)
        - user_id: str
        - raw_text: str
        - post_type: str
        - parent_id: Optional[str]
        """
        try:
            if not self._channel or self._channel.is_closed or not self._connection or self._connection.is_closed:
                if not self.connect():
                    logger.error(f"Cannot publish message; RabbitMQ is unreachable.")
                    return False

            message_body = json.dumps(payload, default=str)
            self._channel.basic_publish(
                exchange="",
                routing_key=self.queue_name,
                body=message_body.encode("utf-8"),
                properties=pika.BasicProperties(
                    delivery_mode=pika.DeliveryMode.Persistent,
                    content_type="application/json",
                ),
            )
            logger.debug(f"Published post to '{self.queue_name}' from platform '{payload.get('platform')}'")
            return True
        except Exception as e:
            logger.error(f"Failed to publish post to RabbitMQ: {e}")
            return False

    def close(self) -> None:
        """Close RabbitMQ channel and connection."""
        try:
            if self._channel and not self._channel.is_closed:
                self._channel.close()
            if self._connection and not self._connection.is_closed:
                self._connection.close()
            logger.info("RabbitMQ Producer connection closed.")
        except Exception as e:
            logger.warning(f"Error while closing RabbitMQ connection: {e}")
