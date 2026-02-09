from abc import ABC
from typing import Final

from torch import nn

from gs_agent.bases.network_backbone import NetworkBackbone


class Policy(nn.Module, ABC):
    """Policy = Backbone (obs->feat) + PolicyHead (feat->action dist/output)."""

    def __init__(self, backbone: NetworkBackbone, action_dim: int) -> None:
        super().__init__()
        self.backbone: Final[NetworkBackbone] = backbone
        self.action_dim: Final[int] = action_dim
