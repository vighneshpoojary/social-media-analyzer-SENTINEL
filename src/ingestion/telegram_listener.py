"""Telegram Channel Listener for Social Media Ingestion using Telethon."""

import asyncio
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

from telethon import TelegramClient, events

from src.ingestion.rabbitmq_producer import RabbitMQProducer

load_dotenv()

logger = logging.getLogger("TelegramListener")


def format_telegram_payload(message: Any, chat: Optional[Any] = None) -> Dict[str, Any]:
    """
    Standardize a Telethon message into the universal social post schema:
    - timestamp (ISO string)
    - platform: 'telegram'
    - user_id: sender identifier (or channel ID if posted anonymously)
    - raw_text: text message content
    - post_type: 'channel_post', 'reply', 'forward', etc.
    - parent_id: reply target ID if applicable
    """
    # Timestamp
    msg_date = getattr(message, "date", None) or datetime.now(timezone.utc)
    if isinstance(msg_date, datetime):
        iso_timestamp = msg_date.isoformat()
    else:
        iso_timestamp = str(msg_date)

    # Determine user_id
    user_id = "unknown"
    sender_id = getattr(message, "sender_id", None)
    from_id = getattr(message, "from_id", None)
    peer_id = getattr(message, "peer_id", None)

    if sender_id is not None and not isinstance(sender_id, (bool,)):
        user_id = str(sender_id)
    elif from_id is not None and not isinstance(from_id, (bool,)):
        user_id = str(getattr(from_id, "user_id", from_id))
    elif chat and hasattr(chat, "id") and getattr(chat, "id") is not None:
        user_id = f"channel_{chat.id}"
    elif peer_id is not None and not isinstance(peer_id, (bool,)):
        user_id = str(peer_id)

    # Determine raw_text
    raw_text = ""
    msg_attr = getattr(message, "message", None)
    text_attr = getattr(message, "text", None)
    if isinstance(msg_attr, str) and msg_attr:
        raw_text = msg_attr
    elif isinstance(text_attr, str):
        raw_text = text_attr
    elif msg_attr is not None:
        raw_text = str(msg_attr)

    # Determine post_type and parent_id
    post_type = "channel_post"
    parent_id = None

    reply_to = getattr(message, "reply_to", None)
    reply_to_msg_id = getattr(message, "reply_to_msg_id", None)
    fwd_from = getattr(message, "fwd_from", None)

    if reply_to is not None and not isinstance(reply_to, (bool,)):
        parent_id_val = getattr(reply_to, "reply_to_msg_id", None)
        if parent_id_val is not None:
            post_type = "reply"
            parent_id = str(parent_id_val)
    elif reply_to_msg_id is not None and not isinstance(reply_to_msg_id, (bool,)):
        post_type = "reply"
        parent_id = str(reply_to_msg_id)
    elif fwd_from is not None and not isinstance(fwd_from, (bool,)):
        post_type = "forward"

    return {
        "timestamp": iso_timestamp,
        "platform": "telegram",
        "user_id": user_id,
        "raw_text": raw_text,
        "post_type": post_type,
        "parent_id": parent_id,
    }


class TelegramListener:
    """Listens to public Telegram channels and forwards new posts to RabbitMQ."""

    def __init__(
        self,
        api_id: Optional[int] = None,
        api_hash: Optional[str] = None,
        phone: Optional[str] = None,
        channels: Optional[List[str]] = None,
        producer: Optional[RabbitMQProducer] = None,
        session_name: str = "ntro_telegram_session",
    ) -> None:
        raw_api_id = api_id or os.getenv("TELEGRAM_API_ID")
        self.api_id = int(raw_api_id) if raw_api_id else None
        self.api_hash = api_hash or os.getenv("TELEGRAM_API_HASH")
        self.phone = phone or os.getenv("TELEGRAM_PHONE")
        
        env_channels = os.getenv("TELEGRAM_CHANNELS", "")
        if channels:
            self.channels = channels
        elif env_channels:
            self.channels = [c.strip() for c in env_channels.split(",") if c.strip()]
        else:
            self.channels = []

        self.session_name = session_name
        self.producer = producer or RabbitMQProducer(queue_name="social_media_queue")
        self._client: Optional[TelegramClient] = None
        self._is_running = False

    def is_configured(self) -> bool:
        """Check if minimum Telegram credentials are provided."""
        return bool(self.api_id and self.api_hash)

    async def start(self) -> None:
        """Start listening to Telegram channels."""
        if not self.is_configured():
            logger.warning(
                "Telegram API credentials (TELEGRAM_API_ID, TELEGRAM_API_HASH) are not set. "
                "Telegram listener will remain dormant."
            )
            return

        logger.info(f"Initializing Telegram client for channels: {self.channels or 'all incoming'}")
        self._client = TelegramClient(self.session_name, self.api_id, self.api_hash)

        # Connect producer
        self.producer.connect()

        # Register event handler for target channels
        chats_filter = self.channels if self.channels else None

        @self._client.on(events.NewMessage(chats=chats_filter))
        async def on_new_message(event):
            try:
                chat = await event.get_chat()
                payload = format_telegram_payload(event.message, chat=chat)
                logger.info(
                    f"[Telegram] Captured post from chat='{getattr(chat, 'title', getattr(chat, 'username', 'unknown'))}', "
                    f"user_id={payload['user_id']}, length={len(payload['raw_text'])}"
                )
                self.producer.publish_post(payload)
            except Exception as ex:
                logger.error(f"Error processing Telegram message: {ex}", exc_info=True)

        self._is_running = True
        logger.info("Connecting to Telegram...")
        await self._client.start(phone=self.phone)
        logger.info("Telegram listener is active and listening for new messages.")

        try:
            await self._client.run_until_disconnected()
        except asyncio.CancelledError:
            logger.info("Telegram listener task was cancelled.")
        finally:
            await self.stop()

    async def stop(self) -> None:
        """Gracefully disconnect Telegram client and producer."""
        self._is_running = False
        if self._client and self._client.is_connected():
            logger.info("Disconnecting Telegram client...")
            await self._client.disconnect()
        if self.producer:
            self.producer.close()
        logger.info("Telegram listener stopped.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    listener = TelegramListener()
    if listener.is_configured():
        asyncio.run(listener.start())
    else:
        logger.warning("Please configure TELEGRAM_API_ID and TELEGRAM_API_HASH in your .env file.")
