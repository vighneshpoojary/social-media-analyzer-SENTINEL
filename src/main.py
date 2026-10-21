"""Master runner script for NTRO Social Media Analytics pipeline."""

import argparse
import asyncio
import logging
import os
import signal
import sys
from datetime import datetime, timezone
from dotenv import load_dotenv

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.ingestion.queue_worker import QueueWorker
from src.ingestion.rabbitmq_producer import RabbitMQProducer
from src.ingestion.telegram_listener import TelegramListener
from src.ingestion.x_scraper import XScraper
from src.storage.clickhouse_client import ClickHouseStorageClient

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("NTRO_Pipeline_Master")


async def run_mock_ingestion(producer: RabbitMQProducer, count: int = 15) -> None:
    """Send simulated posts to RabbitMQ to test pipeline throughput and batching."""
    logger.info(f"Injecting {count} sample posts into RabbitMQ for pipeline verification...")
    if not producer.connect():
        logger.error("Could not connect to RabbitMQ for mock ingestion.")
        return

    platforms = ["telegram", "twitter"]
    for i in range(1, count + 1):
        platform = platforms[i % 2]
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "platform": platform,
            "user_id": f"test_user_{i}_{platform}",
            "raw_text": f"Simulated intelligence post #{i} about cybersecurity and regional updates.",
            "post_type": "channel_post" if platform == "telegram" else "tweet",
            "parent_id": f"parent_msg_{i-1}" if i % 3 == 0 else None,
        }
        producer.publish_post(payload)
        await asyncio.sleep(0.1)

    logger.info(f"Successfully published {count} mock posts into RabbitMQ.")


async def main():
    parser = argparse.ArgumentParser(
        description="NTRO Social Media Analytics - Ingestion & Queue Pipeline"
    )
    parser.add_argument(
        "--worker-only",
        action="store_true",
        help="Run only the RabbitMQ Queue Worker without scrapers.",
    )
    parser.add_argument(
        "--mock-ingest",
        action="store_true",
        help="Publish mock posts into the queue for testing before starting listeners.",
    )
    parser.add_argument(
        "--channels",
        type=str,
        default="",
        help="Comma-separated Telegram channel usernames/IDs to monitor.",
    )
    parser.add_argument(
        "--keywords",
        type=str,
        default="",
        help="Comma-separated Twitter/X search keywords.",
    )
    parser.add_argument(
        "--api",
        action="store_true",
        help="Launch the FastAPI server concurrently with the pipeline workers.",
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("NTRO Social Media Analytics - Starting Pipeline")
    logger.info("=" * 60)

    # Initialize ClickHouse table check
    storage = ClickHouseStorageClient()
    logger.info(f"Target ClickHouse DB: {storage.host}:{storage.port}/{storage.database}")

    # Shared RabbitMQ producer
    producer = RabbitMQProducer(queue_name="social_media_queue")

    # If mock ingestion is requested, publish sample messages
    if args.mock_ingest:
        await run_mock_ingestion(producer, count=15)

    # Instantiate services
    worker = QueueWorker(clickhouse_client=storage)
    
    channels_list = [c.strip() for c in args.channels.split(",") if c.strip()] if args.channels else None
    keywords_list = [k.strip() for k in args.keywords.split(",") if k.strip()] if args.keywords else None

    telegram_listener = TelegramListener(channels=channels_list, producer=producer)
    x_scraper = XScraper(keywords=keywords_list, producer=producer)

    tasks = []

    # 1. Queue Worker Task
    logger.info("Registering Queue Worker task...")
    tasks.append(asyncio.create_task(worker.start(), name="QueueWorker"))

    # 2. Ingestion Tasks (if not worker-only)
    if not args.worker_only:
        if telegram_listener.is_configured():
            logger.info("Registering Telegram Listener task...")
            tasks.append(asyncio.create_task(telegram_listener.start(), name="TelegramListener"))
        else:
            logger.info(
                "[Status] Telegram Listener skipped (TELEGRAM_API_ID / TELEGRAM_API_HASH not set in .env)."
            )

        if x_scraper.is_configured():
            logger.info("Registering Twitter/X Scraper task...")
            tasks.append(asyncio.create_task(x_scraper.start(), name="XScraper"))
        else:
            logger.info(
                "[Status] Twitter/X Scraper skipped (TWITTER_USERNAME / TWITTER_PASSWORD not set in .env)."
            )

        if args.api:
            import uvicorn
            logger.info("Registering FastAPI API Server task on http://localhost:8000...")
            config = uvicorn.Config("src.api.server:app", host="0.0.0.0", port=8000, log_level="warning")
            server = uvicorn.Server(config)
            tasks.append(asyncio.create_task(server.serve(), name="FastAPIServer"))

    logger.info(f"Pipeline running with {len(tasks)} active task(s). Press Ctrl+C to terminate.")

    # Graceful shutdown handler
    stop_event = asyncio.Event()

    def signal_handler():
        logger.info("Shutdown signal received. Cancelling pipeline tasks...")
        stop_event.set()
        for t in tasks:
            t.cancel()

    # Register signals if on supported platforms
    loop = asyncio.get_running_loop()
    if sys.platform != "win32":
        for s in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(s, signal_handler)

    try:
        # Wait for all tasks or stop event
        await asyncio.gather(*tasks, return_exceptions=True)
    except (asyncio.CancelledError, KeyboardInterrupt):
        logger.info("Pipeline execution cancelled.")
    finally:
        logger.info("Cleaning up pipeline resources...")
        worker.stop()
        await telegram_listener.stop()
        await x_scraper.stop()
        producer.close()
        logger.info("Pipeline successfully terminated.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Terminated by user.")
