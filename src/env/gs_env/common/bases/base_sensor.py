import abc

import torch


class BaseSensor(abc.ABC):
    """
    Base class for all sensors
    """

    @abc.abstractmethod
    def get_observation(self, envs_idx: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Get the observation from the sensor.

        Args:
            envs_idx (torch.Tensor): Indices of environments in the batch to get observation.
        """

    @abc.abstractmethod
    def initialize(self) -> None:
        """Initialize the sensor."""


class BaseCamera(BaseSensor, abc.ABC):
    """
    Base class for all camera devices.
    """

    def __init__(self, silent: bool = True) -> None:
        """
        Initialize the camera device.
        :param silent: If True, suppress output.
        """
        super().__init__()

        self._silent = silent

    def initialize(self) -> None: ...

    @abc.abstractmethod
    def get_frame(self) -> dict[str, torch.Tensor]:
        """
        Get a frame from the camera device.
        """
        ...

    @property
    @abc.abstractmethod
    def resolution(self) -> tuple[int, int]:
        """
        Get the resolution of the camera device.
        """
        ...

    @property
    @abc.abstractmethod
    def fps(self) -> int:
        """
        Get the frames per second of the camera device.
        """
        ...
