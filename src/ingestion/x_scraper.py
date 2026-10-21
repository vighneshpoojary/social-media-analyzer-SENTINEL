"""Twitter / X Scraper for Social Media Ingestion using Twikit."""

import asyncio
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set
from dotenv import load_dotenv

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from twikit import Client as TwikitClient

from src.ingestion.rabbitmq_producer import RabbitMQProducer

load_dotenv()

logger = logging.getLogger("XScraper")


def format_tweet_payload(tweet: Any) -> Dict[str, Any]:
    """
    Standardize a Twikit Tweet object into the universal social post schema:
    - timestamp (ISO string)
    - platform: 'twitter'
    - user_id: author ID or screen name
    - raw_text: tweet body text
    - post_type: 'tweet', 'reply', 'retweet', 'quote'
    - parent_id: in_reply_to_tweet_id or quoted_tweet_id if applicable
    """
    # Timestamp
    raw_created_at = getattr(tweet, "created_at", None)
    if isinstance(raw_created_at, datetime):
        iso_timestamp = raw_created_at.isoformat()
    elif isinstance(raw_created_at, str):
        try:
            parsed = datetime.fromisoformat(raw_created_at.replace("Z", "+00:00"))
            iso_timestamp = parsed.isoformat()
        except Exception:
            iso_timestamp = raw_created_at
    else:
        iso_timestamp = datetime.now(timezone.utc).isoformat()

    # User identification
    user_id = "unknown"
    user_obj = getattr(tweet, "user", None)
    user_id_attr = getattr(tweet, "user_id", None)
    if user_obj is not None and not isinstance(user_obj, bool):
        uid = getattr(user_obj, "id", getattr(user_obj, "screen_name", None))
        if uid is not None and not isinstance(uid, (bool,)):
            user_id = str(uid)
    elif user_id_attr is not None and not isinstance(user_id_attr, bool):
        user_id = str(user_id_attr)

    # Raw text
    raw_text = ""
    full_text_attr = getattr(tweet, "full_text", None)
    text_attr = getattr(tweet, "text", None)
    if isinstance(full_text_attr, str) and full_text_attr:
        raw_text = full_text_attr
    elif isinstance(text_attr, str):
        raw_text = text_attr
    elif text_attr is not None:
        raw_text = str(text_attr)

    # Post type & Parent ID
    post_type = "tweet"
    parent_id = None

    is_rt = getattr(tweet, "is_retweet", False)
    rt = getattr(tweet, "retweeted_tweet", None)
    is_quote = getattr(tweet, "is_quote", False)
    qt = getattr(tweet, "quoted_tweet", None)
    reply_to = getattr(tweet, "reply_to", None)
    in_reply_to = getattr(tweet, "in_reply_to", None)

    if (is_rt is True) or (rt is not None and not isinstance(rt, bool) and hasattr(rt, "id") and getattr(rt, "id") is not None):
        post_type = "retweet"
        if rt and hasattr(rt, "id") and getattr(rt, "id") is not None:
            parent_id = str(rt.id)
    elif (is_quote is True) or (qt is not None and not isinstance(qt, bool) and hasattr(qt, "id") and getattr(qt, "id") is not None):
        post_type = "quote"
        if qt and hasattr(qt, "id") and getattr(qt, "id") is not None:
            parent_id = str(qt.id)
    elif isinstance(reply_to, (str, int)) and reply_to:
        post_type = "reply"
        parent_id = str(reply_to)
    elif isinstance(in_reply_to, (str, int)) and in_reply_to:
        post_type = "reply"
        parent_id = str(in_reply_to)

    return {
        "timestamp": iso_timestamp,
        "platform": "twitter",
        "user_id": user_id,
        "raw_text": raw_text,
        "post_type": post_type,
        "parent_id": parent_id,
    }


class XScraper:
    """Scrapes recent public tweets matching keywords and sends them to RabbitMQ."""

    def __init__(
        self,
        keywords: Optional[List[str]] = None,
        username: Optional[str] = None,
        email: Optional[str] = None,
        password: Optional[str] = None,
        cookies_file: str = "cookies.json",
        scrape_interval: int = 60,
        producer: Optional[RabbitMQProducer] = None,
    ) -> None:
        self.username = username or os.getenv("TWITTER_USERNAME")
        self.email = email or os.getenv("TWITTER_EMAIL")
        self.password = password or os.getenv("TWITTER_PASSWORD")
        self.cookies_file = cookies_file
        self.scrape_interval = scrape_interval

        env_keywords = os.getenv("TWITTER_SEARCH_KEYWORDS", "")
        if keywords:
            self.keywords = keywords
        elif env_keywords:
            self.keywords = [k.strip() for k in env_keywords.split(",") if k.strip()]
        else:
            self.keywords = ["#news", "technology", "security"]

        self.producer = producer or RabbitMQProducer(queue_name="social_media_queue")
        self._client: Optional[TwikitClient] = None
        self._seen_tweet_ids: Set[str] = set()
        self._is_running = False

    def is_configured(self) -> bool:
        """Check if login credentials or cookies file are available."""
        return bool((self.username and self.password) or os.path.exists(self.cookies_file))

    async def authenticate(self) -> bool:
        """Initialize Twikit client and authenticate using cookies or credentials."""
        self._client = TwikitClient("en-US")

        if os.path.exists(self.cookies_file):
            try:
                logger.info(f"Loading Twitter session cookies from {self.cookies_file}...")
                self._client.load_cookies(self.cookies_file)
                logger.info("Successfully loaded Twitter session cookies.")
                return True
            except Exception as e:
                logger.warning(f"Failed to load cookies from {self.cookies_file}: {e}")

        if self.username and self.password:
            try:
                logger.info(f"Logging in to Twitter / X as @{self.username}...")
                await self._client.login(
                    auth_info_1=self.username,
                    auth_info_2=self.email,
                    password=self.password
                )
                self._client.save_cookies(self.cookies_file)
                logger.info("Twitter login successful; session saved.")
                return True
            except Exception as e:
                logger.error(f"Twitter login failed: {e}")
                return False

        logger.warning(
            "Twitter credentials (TWITTER_USERNAME, TWITTER_PASSWORD) or cookies.json not found."
        )
        return False

    async def search_and_publish(self, query: str, count: int = 20) -> int:
        """Execute a search query on Twitter and publish new tweets to RabbitMQ."""
        if not self._client:
            logger.error("XScraper client is not initialized.")
            return 0

        logger.info(f"Searching Twitter for recent tweets matching: '{query}'...")
        published_count = 0
        try:
            tweets = await self._client.search_tweet(query, product="Latest", count=count)
            for tweet in tweets:
                tweet_id = str(getattr(tweet, "id", ""))
                if tweet_id in self._seen_tweet_ids:
                    continue

                self._seen_tweet_ids.add(tweet_id)
                payload = format_tweet_payload(tweet)
                logger.info(
                    f"[Twitter/X] Captured tweet ID={tweet_id} by user={payload['user_id']}: {payload['raw_text'][:50]}..."
                )
                if self.producer.publish_post(payload):
                    published_count += 1

            # Cap seen cache size to prevent memory growth
            if len(self._seen_tweet_ids) > 10000:
                self._seen_tweet_ids = set(list(self._seen_tweet_ids)[-5000:])

            logger.info(f"Published {published_count} new tweets for query '{query}'.")
        except Exception as e:
            logger.error(f"Error while searching tweets for '{query}': {e}")

        return published_count

    async def start(self) -> None:
        """Start the periodic scraping loop."""
        if not self.is_configured():
            logger.warning(
                "Twitter credentials are not configured in .env. "
                "X / Twitter scraper will remain dormant."
            )
            return

        authenticated = await self.authenticate()
        if not authenticated:
            logger.warning("Could not authenticate with Twitter. XScraper stopping.")
            return

        self.producer.connect()
        self._is_running = True
        logger.info(f"XScraper active. Monitoring keywords: {self.keywords}")

        try:
            while self._is_running:
                for kw in self.keywords:
                    await self.search_and_publish(kw)
                    await asyncio.sleep(2)  # Short pause between queries to avoid rate limits

                logger.info(f"Sleeping for {self.scrape_interval}s before next scrape round...")
                await asyncio.sleep(self.scrape_interval)
        except asyncio.CancelledError:
            logger.info("XScraper task cancelled.")
        finally:
            await self.stop()

    async def stop(self) -> None:
        """Stop scraper and close resources."""
        self._is_running = False
        if self.producer:
            self.producer.close()
        logger.info("XScraper stopped.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    scraper = XScraper()
    if scraper.is_configured():
        asyncio.run(scraper.start())
    else:
        logger.warning("Please configure TWITTER_USERNAME and TWITTER_PASSWORD in your .env file.")
