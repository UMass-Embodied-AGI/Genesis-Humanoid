from gs_schemas.base_types import GenesisEnum, genesis_pydantic_config
from pydantic import BaseModel


class ActivationType(GenesisEnum):
    RELU = "RELU"
    TANH = "TANH"
    GELU = "GELU"
    SWISH = "SWISH"


class NetworkBackboneType(GenesisEnum):
    MLP = "MLP"
    RNN = "RNN"
    CNN = "CNN"


class PolicyType(GenesisEnum):
    GAUSSIAN = "GAUSSIAN"


class ValueFunctionType(GenesisEnum):
    STATE_VALUE = "STATE_VALUE"
    Q_VALUE = "Q_VALUE"


class MLPConfig(BaseModel):
    """Configuration for Multi-Layer Perceptron networks."""

    model_config = genesis_pydantic_config(frozen=True)

    hidden_dims: tuple[int, ...] = (256, 256, 128)
    activation: ActivationType = ActivationType.RELU

    network_type: NetworkBackboneType = NetworkBackboneType.MLP


NetworkBackboneConfig = MLPConfig
