from abc import ABC, abstractmethod

import torch
from torch import nn


class NetworkBackbone(nn.Module, ABC):
    """Abstract base class for network backbones."""

    @abstractmethod
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Process observations into latent features.

        Returns:
            Tensor of shape [batch_size, feature_dim]
        """
        ...

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Dimension of the output latent vector."""
        ...
