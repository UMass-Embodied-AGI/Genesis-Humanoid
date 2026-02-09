from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from gs_agent.bases.policy import Policy


class BaseRunner(ABC):
    @abstractmethod
    def train(self, metric_logger: Any) -> dict[str, Any]:
        """
        Train the algorithm for a given number of episodes.
        """
        ...

    @abstractmethod
    def load_checkpoint(self, path: Path) -> None:
        """
        Load the checkpoint of the algorithm.
        """
        ...

    @abstractmethod
    def get_inference_policy(self) -> Policy:
        """
        Get the policy for evaluation.
        """
        ...
