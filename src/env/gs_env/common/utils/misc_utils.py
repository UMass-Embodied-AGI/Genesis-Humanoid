from typing import Any

import numpy as np

#
from gymnasium import spaces


def get_space_dim(space: spaces.Space[Any]) -> int:
    if isinstance(space, spaces.Box):
        return int(np.prod(space.shape))
    elif isinstance(space, spaces.Discrete):
        return 1
    elif isinstance(space, spaces.Dict):
        return sum(get_space_dim(subspace) for subspace in space.spaces.values())
    else:
        raise NotImplementedError(f"Unsupported space type: {type(space)}")
