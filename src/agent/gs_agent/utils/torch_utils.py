import torch
import torch.nn as nn


def get_activation(activation: str | type[nn.Module] | nn.Module) -> nn.Module:
    """Return an activation module instance given a name, class, or instance."""
    # If the user passed in an nn.Module instance, just return it
    if isinstance(activation, nn.Module):
        return activation

    # If they passed in a class (e.g. nn.ReLU), instantiate it
    if isinstance(activation, type) and issubclass(activation, nn.Module):  # type: ignore
        return activation()

    # Otherwise assume it's a string
    name = activation.lower().strip()  # type: ignore[union-attr]
    activations: dict[str, nn.Module] = {
        "relu": nn.ReLU(),
        "sigmoid": nn.Sigmoid(),
        "tanh": nn.Tanh(),
        "leaky_relu": nn.LeakyReLU(),
        "elu": nn.ELU(),
        "softmax": nn.Softmax(dim=-1),
        "none": nn.Identity(),
    }
    if name not in activations:
        raise ValueError(
            f"Unsupported activation: {activation!r}. "
            f"Supported options: {list(activations.keys()) + ['nn.Module subclass', 'nn.Module instance']}"
        )
    return activations[name]


def get_torch_device() -> torch.device:
    # Properly detect available device (CUDA, MPS, or CPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA is available")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("MPS is available")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device
