from pathlib import Path
from typing import Any

import matplotlib.ticker as ticker
import numpy as np
import yaml


def cross_correlation(
    a: np.typing.NDArray[Any], b: np.typing.NDArray[Any], allow_flip: bool = False
) -> float:
    """
    Compute the sub-sample cross-correlation lag between two 1D signals a and b.
    Handles both normal and flipped correlation (max or min peak).
    Returns lag in samples (positive means b lags behind a).
    """
    assert a.ndim == 1 and b.ndim == 1 and len(a) == len(b), (
        "Inputs must be 1D arrays of the same length"
    )
    a = a - np.mean(a)
    b = b - np.mean(b)
    corr = np.correlate(a, b, mode="full")
    lags = np.arange(-len(a) + 1, len(a))
    overlap = len(a) - np.abs(lags)
    corr_unbiased = corr / overlap
    k_max = np.argmax(corr)
    k_min = np.argmin(corr)
    k_peak = k_min if abs(corr[k_min]) > abs(corr[k_max]) and allow_flip else k_max
    lag_int = lags[k_peak]

    # Parabolic interpolation: use corr[k-1], corr[k], corr[k+1]
    if 0 < k_peak < len(corr) - 1:
        c_minus = corr_unbiased[k_peak - 1]
        c_0 = corr_unbiased[k_peak]
        c_plus = corr_unbiased[k_peak + 1]

        denom = -2 * c_0 + c_minus + c_plus
        if denom != 0:
            delta = 0.5 * (c_minus - c_plus) / denom
        else:
            delta = 0.0
    else:
        delta = 0.0

    lag_subsample = lag_int + delta
    return -lag_subsample.item()


def compute_SRD(a: np.typing.NDArray[Any], eps: float = 1e-3) -> float:
    """
    symmetric relative difference
    """
    a0 = a[:-1]
    a1 = a[1:]
    return np.mean(np.abs(a0 - a1) / (np.abs(a0) + np.abs(a1) + eps)).item()


def compute_SD(a: np.typing.NDArray[Any]) -> float:
    """
    symmetric difference
    """
    a0 = a[:-1]
    a1 = a[1:]
    return np.mean(np.abs(a0 - a1)).item()


def measure_lag_and_noise(
    target: np.ndarray, measured: np.ndarray, allow_flip: bool = False
) -> tuple[float, float]:
    """
    Align two 1D signals given lag_samples (positive => measured lags behind target).
    Returns (target_aligned, measured_aligned).
    """
    assert target.ndim == 1 and measured.ndim == 1 and len(target) == len(measured), (
        "Inputs must be 1D arrays of the same length"
    )
    lag_samples = cross_correlation(target, measured, allow_flip=allow_flip)
    n = len(target)
    target = target[:n]
    measured = measured[:n]

    t = np.arange(n, dtype=float)
    # measured_aligned(t) = measured(t - lag)
    measured_aligned = np.interp(t - lag_samples, t, measured, left=np.nan, right=np.nan)

    valid = ~np.isnan(measured_aligned)
    target_aligned = target[valid]
    measured_aligned = measured_aligned[valid]

    err = measured_aligned - target_aligned
    amp = (np.max(target_aligned) - np.min(target_aligned)) / 2
    nrms_err = np.sqrt(np.mean(err**2)) / amp
    return lag_samples, nrms_err


def plot_metric_on_axis(
    ax: Any,
    steps: Any,
    data_lists: list[list[float]],
    labels: list[str],
    ylabel: str,
    title: str,
    yscale: str = "log",
    xlabel: str | None = None,
    show_mean: bool = True,
) -> None:
    """Plot multiple data series on a single axis with optional log scale.

    Args:
        ax: Matplotlib axis to plot on
        steps: X-axis values (step numbers)
        data_lists: List of data arrays to plot (e.g., [mean_data, max_data])
        labels: List of labels for each data series
        ylabel: Label for y-axis
        title: Title for the subplot
        yscale: Scale for y-axis ('log' or 'linear')
        xlabel: Label for x-axis (optional)
        show_mean: Whether to display mean values as text (default: True)
    """
    # Plot each data series and their means
    for data, label in zip(data_lists, labels, strict=False):
        line = ax.plot(steps, data, label=label, linewidth=1.5, alpha=0.8)[0]

        # Add horizontal dotted line for mean if requested
        if show_mean:
            data_mean = np.mean(data)
            ax.axhline(
                y=data_mean,
                color=line.get_color(),
                linestyle="--",
                linewidth=1.2,
                alpha=0.6,
                label=f"mean: {data_mean:.4f}",
            )

    # Set scale
    ax.set_yscale(yscale)

    # Set ticks and formatter for log scale
    if yscale == "log":
        # Define all possible tick locations
        all_yticks = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]

        # Filter ticks based on data range
        all_data = np.concatenate([np.array(d) for d in data_lists])
        all_data = all_data[all_data > 0]  # Filter out zeros
        if len(all_data) > 0:
            data_min = np.min(all_data)
            data_max = np.max(all_data)
            # Only include ticks within data range
            filtered_ticks = [tick for tick in all_yticks if data_min <= tick <= data_max]
            if not filtered_ticks:
                # If no ticks in range, expand slightly
                filtered_ticks = [
                    tick for tick in all_yticks if tick >= data_min * 0.5 and tick <= data_max * 2
                ][:5]
            ax.set_yticks(filtered_ticks)

        # Use decimal formatter
        formatter = ticker.FuncFormatter(lambda y, _: f"{y:.10g}")
        ax.yaxis.set_major_formatter(formatter)

    # Set labels and title
    ax.set_ylabel(ylabel, fontsize=11)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")

    # Add legend and grid
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3, which="both")


def parse_scalar(value: Any) -> Any:
    """Best-effort parse of CLI string into bool/int/float, else return as-is."""
    if isinstance(value, int | float | bool):
        return value
    if isinstance(value, str):
        v = value.strip()
        if v.lower() in {"true", "false"}:
            return v.lower() == "true"
        try:
            if v.isdigit() or (v.startswith("-") and v[1:].isdigit()):
                return int(v)
            return float(v)
        except ValueError:
            return value
    return value


def strip_prefixes(path: str, prefixes: tuple[str, ...]) -> str:
    # repeatedly strip any matching prefix until none match
    changed = True
    while changed:
        changed = False
        for pref in prefixes:
            if path.startswith(pref):
                path = path[len(pref) :]
                changed = True
    return path


def tokenize_path(path: str, prefixes: tuple[str, ...]) -> list[str]:
    """Split a dot/bracket path into tokens after removing known prefixes."""
    path = strip_prefixes(path, prefixes)
    out: list[str] = []
    buf = ""
    i = 0
    while i < len(path):
        c = path[i]
        if c == ".":
            if buf:
                out.append(buf)
                buf = ""
            i += 1
            continue
        if c == "[":
            if buf:
                out.append(buf)
                buf = ""
            j = path.find("]", i)
            if j == -1:
                buf += c
                i += 1
            else:
                out.append(path[i + 1 : j])
                i = j + 1
            continue
        buf += c
        i += 1
    if buf:
        out.append(buf)
    return out


def parse_sequence_like(raw: Any, target_example: Any) -> Any:
    if not isinstance(raw, str):
        return raw
    parts = [p.strip() for p in raw.split(",")]
    parsed = [parse_scalar(p) for p in parts]
    if isinstance(target_example, tuple):
        return tuple(parsed)
    return parsed


def deep_apply(obj: Any, tokens: list[str], raw_val: Any) -> Any:
    """Deeply apply a single override value addressed by tokens to a Pydantic model.

    Handles nested BaseModel, dicts (one-level key), and lists/tuples (by index).
    """
    from pydantic import BaseModel as _BM  # type: ignore

    if not tokens:
        return obj

    field = tokens[0]
    rest = tokens[1:]

    # getattr guard
    try:
        current = getattr(obj, field)
    except AttributeError:
        return obj

    # Leaf assignment
    if not rest:
        new_value = parse_scalar(raw_val)
        if isinstance(current, tuple | list) and isinstance(raw_val, str) and "," in raw_val:
            new_value = parse_sequence_like(raw_val, current)
        if isinstance(current, tuple) and isinstance(new_value, list):
            new_value = tuple(new_value)
        return obj.model_copy(update={field: new_value}) if isinstance(obj, _BM) else obj

    # Dict field (e.g., reward_args)
    if isinstance(current, dict):
        key = rest[0]
        new_dict = dict(current)
        if len(rest) == 1:
            new_dict[key] = parse_scalar(raw_val)
        else:
            # deeper dict nesting not supported
            new_dict[".".join(rest)] = parse_scalar(raw_val)
        return obj.model_copy(update={field: new_dict})

    # List/Tuple field
    if isinstance(current, list | tuple):
        try:
            idx = int(rest[0])
        except ValueError:
            return obj
        if idx < 0 or idx >= len(current):
            return obj
        seq_list = list(current)
        elem = seq_list[idx]
        if len(rest) == 1:
            new_elem = parse_scalar(raw_val)
        else:
            if hasattr(elem, "model_copy"):
                new_elem = deep_apply(elem, rest[1:], raw_val)
            elif isinstance(elem, dict):
                new_elem = dict(elem)
                new_elem[rest[1]] = parse_scalar(raw_val)
            else:
                new_elem = elem
        seq_list[idx] = new_elem
        new_seq = tuple(seq_list) if isinstance(current, tuple) else seq_list
        return obj.model_copy(update={field: new_seq})

    # Nested Pydantic model
    if hasattr(current, "model_copy"):
        updated_sub = deep_apply(current, rest, raw_val)
        return obj.model_copy(update={field: updated_sub})

    return obj


def apply_overrides_generic(
    base_obj: Any, overrides: dict[str, Any] | None, prefixes: tuple[str, ...]
) -> Any:
    if not overrides:
        return base_obj
    updated = base_obj
    for raw_key, raw_val in overrides.items():
        tokens = tokenize_path(raw_key, prefixes)
        updated = deep_apply(updated, tokens, raw_val)
    return updated


def _convert_to_serializable(obj: Any) -> Any:
    """Convert objects to YAML-serializable format.

    Args:
        obj: Object to convert

    Returns:
        YAML-serializable representation
    """
    if obj is None or isinstance(obj, bool | int | float | str):
        return obj
    elif isinstance(obj, dict):
        return {k: _convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_to_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        # Convert tuple to list but mark it with a special tag
        return {"__tuple__": [_convert_to_serializable(item) for item in obj]}
    else:
        # For other types, convert to string
        return str(obj)


def _restore_tuples(obj: Any) -> Any:
    """Restore tuples from YAML data.

    Args:
        obj: Object loaded from YAML

    Returns:
        Object with tuples restored
    """
    if isinstance(obj, dict):
        if "__tuple__" in obj and len(obj) == 1:
            # This is a tuple marker
            return tuple(_restore_tuples(item) for item in obj["__tuple__"])
        else:
            return {k: _restore_tuples(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_restore_tuples(item) for item in obj]
    else:
        return obj


def config_to_yaml(config: Any, yaml_path: str | Path) -> None:
    """Convert a config/args class (typically Pydantic model) to YAML file.

    Args:
        config: Configuration object (Pydantic model or similar)
        yaml_path: Path where YAML file should be saved

    Example:
        >>> config_to_yaml(env_args, "logs/experiment/env_args.yaml")
    """
    from pydantic import BaseModel as _BM  # type: ignore

    # Convert to dictionary
    if isinstance(config, _BM):
        config_dict = config.model_dump()
    elif hasattr(config, "__dict__"):
        config_dict = config.__dict__
    elif isinstance(config, dict):
        config_dict = config
    else:
        raise ValueError(f"Cannot convert config of type {type(config)} to YAML")

    # Convert to serializable format
    serializable_dict = _convert_to_serializable(config_dict)

    # Ensure parent directory exists
    yaml_path = Path(yaml_path)
    yaml_path.parent.mkdir(parents=True, exist_ok=True)

    # Write to YAML file
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(serializable_dict, f, default_flow_style=False, sort_keys=False, indent=2)

    print(f"Config saved to YAML: {yaml_path}")


def yaml_to_config(yaml_path: str | Path, config_class: type | None = None) -> Any:
    """Convert YAML file back to config/args class.

    Args:
        yaml_path: Path to YAML file
        config_class: Optional class type to instantiate. If provided and it's a
                     Pydantic model, will return an instance of that class.
                     If None, returns a dictionary.

    Returns:
        Config object (Pydantic model instance) or dictionary

    Example:
        >>> from gs_env.sim.envs.config.schema import EnvArgs
        >>> env_args = yaml_to_config("logs/experiment/env_args.yaml", EnvArgs)
    """
    from pydantic import BaseModel as _BM  # type: ignore

    # Load YAML file
    with open(yaml_path, encoding="utf-8") as f:
        config_dict = yaml.load(f, Loader=yaml.UnsafeLoader)

    # Restore tuples
    config_dict = _restore_tuples(config_dict)

    # If config_class provided and is Pydantic model, instantiate it
    if config_class is not None and issubclass(config_class, _BM):
        return config_class(**config_dict)
    elif config_class is not None:
        # Try to instantiate with the dictionary
        return config_class(**config_dict)
    else:
        # Return as dictionary
        return config_dict
