"""Custom exception hierarchy for the thesis evaluation pipeline."""


class ThesisError(Exception):
    """Base exception for all thesis pipeline errors."""


class DataLoadError(ThesisError):
    """Failed to load or parse a dataset/file."""


class LLMCallError(ThesisError):
    """LLM API call failed after retries."""


class EvaluationError(ThesisError):
    """Evaluation step failed."""


class ConfigError(ThesisError):
    """Invalid or missing configuration."""


class JudgeFailedError(EvaluationError):
    """Judge evaluation failed (retryable)."""
