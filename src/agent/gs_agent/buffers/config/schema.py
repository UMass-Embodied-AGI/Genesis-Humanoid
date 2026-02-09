from enum import Enum

from gs_schemas.base_types import genesis_pydantic_config
from pydantic import BaseModel


class GAEBufferKey(str, Enum):
    ACTOR_OBS = "ACTOR_OBS"
    CRITIC_OBS = "CRITIC_OBS"
    ACTIONS = "ACTIONS"
    REWARDS = "REWARDS"
    DONES = "DONES"
    VALUES = "VALUES"
    ACTION_LOGPROBS = "ACTION_LOGPROBS"
    MU = "MU"
    SIGMA = "SIGMA"
    ADVANTAGES = "ADVANTAGES"
    RETURNS = "RETURNS"


class BCBufferKey(str, Enum):
    OBSERVATIONS = "OBSERVATIONS"
    ACTIONS = "ACTIONS"


class GAEBufferArgs(BaseModel):
    model_config = genesis_pydantic_config(frozen=True)
    pass


class BCBufferArgs(BaseModel):
    model_config = genesis_pydantic_config(frozen=True)
    pass
