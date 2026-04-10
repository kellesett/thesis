"""
src/evaluators/base.py
Abstract base class for all evaluators.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from src.datasets.base import BaseDataset


class BaseEvaluator(ABC):
    """
    Abstract base for all evaluation methods.

    Each evaluator is initialised once per run with a config dict and
    a loaded dataset instance.  The dataset is used inside evaluate()
    to look up reference data via dataset.get_by_id(generation["id"]).
    """

    evaluator_id: str  # class-level constant, e.g. "surge"

    def __init__(self, config: dict, dataset: BaseDataset) -> None:
        self.config  = config
        self.dataset = dataset

    @abstractmethod
    def evaluate(self, generation: dict) -> dict:
        """
        Evaluate one unified generation dict.

        Args:
            generation: unified generation JSON with keys
                        {id, dataset_id, model_id, query, text, success, meta}

        Returns:
            dict with keys:
                "scores"    — {metric_name: float | None}
                "judge_log" — list of per-attempt log entries (may be empty)
        """
