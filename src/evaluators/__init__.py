"""
src/evaluators/__init__.py
Evaluator registry: maps evaluator_id → class, provides load_evaluator() factory.

To add a new evaluator:
  1. Create src/evaluators/<name>.py with a class inheriting BaseEvaluator
  2. Add an entry to REGISTRY below
"""
from .base import BaseEvaluator
from .surge import SurGEEvaluator
from ..exceptions import JudgeFailedError

REGISTRY: dict[str, type[BaseEvaluator]] = {
    "surge": SurGEEvaluator,
}


def load_evaluator(evaluator_id: str, config: dict, dataset) -> BaseEvaluator:
    """
    Instantiate an evaluator by its id.

    Args:
        evaluator_id: Evaluator identifier (key in REGISTRY)
        config: Configuration dict with required keys for the evaluator
        dataset: Dataset instance to evaluate against

    Returns:
        Instantiated evaluator

    Raises:
        ValueError: If evaluator_id is not in REGISTRY
    """
    cls = REGISTRY.get(evaluator_id)
    if cls is None:
        raise ValueError(
            f"Unknown evaluator id: '{evaluator_id}'. "
            f"Available: {list(REGISTRY)}"
        )
    # Validate that required config keys are present
    required_keys = []
    if evaluator_id == "surge":
        required_keys = ["eval_list", "judge_model", "judge_base_url", "judge_api_key_env"]
    for key in required_keys:
        if key not in config:
            raise ValueError(
                f"Missing required config key '{key}' for evaluator '{evaluator_id}'"
            )
    return cls(config, dataset)
