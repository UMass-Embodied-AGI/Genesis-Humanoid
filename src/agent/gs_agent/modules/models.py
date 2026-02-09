import torch
from torch import nn

from gs_agent.bases.network_backbone import NetworkBackbone
from gs_agent.modules.config.schema import (
    ActivationType,
    MLPConfig,
    NetworkBackboneConfig,
    NetworkBackboneType,
)


def get_activation(
    activation: ActivationType,
) -> nn.Module:
    """Get activation function by typed name."""
    match activation:
        case ActivationType.RELU:
            return nn.ReLU()
        case ActivationType.TANH:
            return nn.Tanh()
        case ActivationType.GELU:
            return nn.GELU()
        case ActivationType.SWISH:
            return nn.SiLU()


class MLPBackbone(NetworkBackbone):
    """Multi-layer perceptron."""

    def __init__(
        self,
        input_dim: int,
        config: MLPConfig,
        device: torch.device,
        output_dim: int | None = None,
    ) -> None:
        super().__init__()

        self.device = device
        self._input_dim = input_dim
        self._hidden_dims = config.hidden_dims
        self._activation = config.activation

        # Build layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in self._hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim, bias=True))
            layers.append(get_activation(self._activation))
            prev_dim = hidden_dim

        if output_dim is not None:
            layers.append(nn.Linear(prev_dim, output_dim, bias=True))

        self._output_dim = output_dim if output_dim is not None else prev_dim

        self._network = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self._network(obs)

    @property
    def output_dim(self) -> int:
        """Output dimension of the network."""
        return self._output_dim

    @property
    def input_dim(self) -> int:
        """Input dimension of the network."""
        return self._input_dim

    @property
    def hidden_dims(self) -> tuple[int, ...]:
        """Hidden dimensions of the network."""
        return self._hidden_dims

    @property
    def activation(self) -> ActivationType:
        """Activation function of the network."""
        return self._activation


class NetworkFactory:
    """Factory for creating different types of neural networks."""

    @staticmethod
    def create_network(
        network_backbone_args: NetworkBackboneConfig,
        input_dim: int,
        device: torch.device,
        output_dim: int | None = None,
    ) -> NetworkBackbone:
        """Create a network based on the provided configuration.

        Args:
            network_backbone_args: Network configuration
            input_dim: Input dimension
            output_dim: Output dimension
            device: Device to place the network on

        Returns:
            Neural network module
        """
        match network_backbone_args.network_type:
            case NetworkBackboneType.MLP:
                return MLPBackbone(
                    input_dim=input_dim,
                    config=network_backbone_args,
                    device=device,
                    output_dim=output_dim,
                )
            case _:
                raise ValueError(f"Unsupported network type: {network_backbone_args.network_type}")
