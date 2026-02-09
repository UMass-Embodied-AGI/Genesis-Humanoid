from abc import ABC, abstractmethod
from typing import Any

import torch


class BaseBuffer(ABC):
    """
    Abstract base class for on-policy and imitation learning buffers.
    """

    @abstractmethod
    def reset(self) -> None:
        """Reset the buffer state."""
        ...

    # TODO: avoid using Any
    @abstractmethod
    def append(self, transition: dict[Any, torch.Tensor]) -> None:
        """Append a transition to the buffer."""
        ...

    @abstractmethod
    def is_full(self) -> bool:
        """Check if the buffer is full."""
        ...

    @abstractmethod
    def __len__(self) -> int:
        """Return the total number of data points stored."""
        ...
