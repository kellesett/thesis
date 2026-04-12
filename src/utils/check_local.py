"""
Check which models are available on local server (LOCAL_API_BASE).

Usage:
    python src/utils/check_local.py
    python src/utils/check_local.py --ping        # only check availability
    python src/utils/check_local.py --model foo   # check specific model
"""

import os
import sys
import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent.parent / ".env")
except ImportError:
    pass

from openai import OpenAI, APIConnectionError, APIStatusError


def make_client(base_url: str, api_key: str) -> OpenAI:
    return OpenAI(base_url=base_url, api_key=api_key or "local")


def list_models(client: OpenAI) -> list[tuple[str, str]]:
    """Return list of (id, status) sorted by id."""
    response = client.models.list()
    return sorted(
        (m.id, getattr(m, "status", "—")) for m in response.data
    )


def ping(client: OpenAI, model: str) -> tuple[bool, str, float]:
    """Send minimal request to verify model responds.
    Return (success, reply, elapsed_sec)."""
    import time
    t0 = time.perf_counter()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Say 'ok'"}],
            max_tokens=5,
            timeout=30,
        )
        elapsed = time.perf_counter() - t0
        reply = resp.choices[0].message.content or ""
        return True, reply.strip(), elapsed
    except APIStatusError as e:
        elapsed = time.perf_counter() - t0
        logger.error(f"HTTP error for {model} at {client.base_url}: {e.status_code}")
        return False, f"HTTP {e.status_code}: {e.message}", elapsed
    except Exception as e:
        elapsed = time.perf_counter() - t0
        logger.error(f"Connection error for {model} at {client.base_url}: {e}")
        return False, str(e), elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Check models on LOCAL_API_BASE")
    parser.add_argument("--base",  default=None, help="Server URL (default: LOCAL_API_BASE from .env)")
    parser.add_argument("--key",   default=None, help="API key (default: LOCAL_API_KEY from .env)")
    parser.add_argument("--ping",  action="store_true", help="Ping each model with short request")
    parser.add_argument("--model", default=None, help="Check specific model (implies --ping)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    base_url = args.base or os.getenv("LOCAL_API_BASE")
    api_key  = args.key  or os.getenv("LOCAL_API_KEY", "local")

    if not base_url:
        logger.error("LOCAL_API_BASE not set. Set in .env or via --base")
        sys.exit(1)

    logger.info(f"Server: {base_url}")
    client = make_client(base_url, api_key)

    # List models
    try:
        models = list_models(client)
    except APIConnectionError:
        logger.error(f"Could not connect to {base_url}")
        logger.error("Ensure server is running and URL is correct")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error retrieving model list: {e}")
        sys.exit(1)

    if not models:
        logger.warning("Server responds but model list is empty")
        sys.exit(0)

    logger.info(f"Available models ({len(models)}):")
    for mid, status in models:
        logger.info(f"      * {mid}  [{status}]")

    # Ping
    to_ping: list[str] = []
    if args.model:
        to_ping = [args.model]
    elif args.ping:
        to_ping = [mid for mid, _ in models]

    if to_ping:
        logger.info("Pinging models...")
        for m in to_ping:
            ok, reply, elapsed = ping(client, m)
            status = "[OK]" if ok else "[FAIL]"
            logger.info(f"      {status}  {m}  >>  {reply!r}  ({elapsed:.2f}s)")


if __name__ == "__main__":
    main()
