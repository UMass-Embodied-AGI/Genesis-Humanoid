from abc import ABC, abstractmethod

import torch
from torch import nn


class BaseStateValueFunction(nn.Module, ABC):
    """State value function V(s)."""

    @abstractmethod
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """obs: [B, O]  ->  returns V: [B, 1]."""
        ...


class BaseQValueFunction(nn.Module, ABC):
    """Q-value function Q(s, a)."""

    @abstractmethod
    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        """obs: [B, O], act: [B, A]  ->  returns Q: [B, 1]."""
        ...
