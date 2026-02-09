import abc

import torch


class BaseObject(abc.ABC):
    """
    Base class for all objects.
    """

    _name: str

    @abc.abstractmethod
    def reset(self, envs_idx: torch.Tensor) -> None:
        """
        Reset the object.
        """
        ...

    @property
    def name(self) -> str:
        return self._name
