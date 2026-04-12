"""Shared utilities for all metric modules."""
import logging
import os
from pathlib import Path

import yaml
from openai import OpenAI

logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict:
    """Load YAML configuration from the given path.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If config file does not exist.
        yaml.YAMLError: If config file is malformed.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path) as f:
        return yaml.safe_load(f)


def make_client(cfg: dict) -> OpenAI:
    """Create an OpenAI client from config.

    Expects cfg to contain 'judge_api_key_env' and 'judge_base_url'.

    Args:
        cfg: Configuration dict with judge API settings.

    Returns:
        Configured OpenAI client.

    Raises:
        RuntimeError: If the API key environment variable is not set.
    """
    key_env = cfg.get("judge_api_key_env")
    if not key_env:
        raise RuntimeError("Config missing 'judge_api_key_env'")
    api_key = os.environ.get(key_env, "")
    if not api_key:
        raise RuntimeError(f"API key not set: env var '{key_env}'")
    base_url = cfg.get("judge_base_url")
    if not base_url:
        raise RuntimeError("Config missing 'judge_base_url'")
    return OpenAI(api_key=api_key, base_url=base_url)
