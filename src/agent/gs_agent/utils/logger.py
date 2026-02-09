# =============================================================================
# SB3 Source:   https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/logger.py
# SB3 License:  MIT License (Copyright (c) 2019–2025 Antonin Raffin et al.)
#
# Modifications:
#   • 2025-06-09, Yunlong Song: Refactored API, added W&B support, cleaned imports.
# =============================================================================
import datetime
import json
import os
import sys
import tempfile
import warnings
from collections import defaultdict
from collections.abc import Mapping, Sequence
from io import TextIOBase
from typing import Any, TextIO, Literal

import numpy as np
import pandas
import torch as th
import wandb
from matplotlib import pyplot as plt

try:
    from torch.utils.tensorboard import SummaryWriter
    from torch.utils.tensorboard.summary import hparams
except ImportError:
    SummaryWriter = None  # type: ignore[misc, assignment]

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

DEBUG = 10
INFO = 20
WARN = 30
ERROR = 40
DISABLED = 50


class Video:
    def __init__(self, frames: th.Tensor, fps: float) -> None:
        """
        Args:
            frames (th.Tensor): the video frames, of shape (T, H, W, C)
            fps (float): frames per second
        """
        self.frames = frames
        self.fps = fps


class Figure:
    def __init__(self, figure: plt.figure, close: bool) -> None:  # type: ignore
        """
        Args:
            figure (plt.figure): the figure to log
            close (bool): whether to close the figure
        """
        self.figure = figure
        self.close = close


class Image:
    def __init__(self, image: th.Tensor | np.ndarray | str, dataformats: str) -> None:  # type: ignore
        """
        Args:
            image (Union[th.Tensor, np.ndarray, str]): the image to log
            dataformats (str): the dataformats of the image, e.g. "CHW"
        """
        self.image = image
        self.dataformats = dataformats


class HParam:
    def __init__(
        self,
        hparam_dict: Mapping[str, bool | str | float | None],
        metric_dict: Mapping[str, float],
    ) -> None:
        self.hparam_dict = hparam_dict
        if not metric_dict:
            raise Exception(
                "`metric_dict` must not be empty to display hyperparameters to the HPARAMS tensorboard tab."
            )
        self.metric_dict = metric_dict


class FormatUnsupportedError(NotImplementedError):
    def __init__(self, unsupported_formats: Sequence[str], value_description: str) -> None:
        if len(unsupported_formats) > 1:
            format_str = f"formats {', '.join(unsupported_formats)} are"
        else:
            format_str = f"format {unsupported_formats[0]} is"
        super().__init__(
            f"The {format_str} not supported for the {value_description} value logged.\n"
            f"You can exclude formats via the `exclude` parameter of the logger's `record` function."
        )


class KVWriter:
    """
    Key Value writer
    """

    def write(
        self,
        key_values: dict[str, Any],
        key_excluded: dict[str, tuple[str, ...] | None],
        step: int = 0,
    ) -> None:
        """
        Write a dictionary to file
        """
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


class SeqWriter:
    """
    sequence writer
    """

    def write_sequence(self, sequence: list[str]) -> None:
        """
        write_sequence an array to file
        """
        raise NotImplementedError


class HumanOutputFormat(KVWriter, SeqWriter):
    """A human-readable output format producing ASCII tables of key-value pairs.

    Set attribute ``max_length`` to change the maximum length of keys and values
    to write to output (or specify it when calling ``__init__``).

    :param filename_or_file: the file to write the log to
    :param max_length: the maximum length of keys and values to write to output.
        Outputs longer than this will be truncated. An error will be raised
        if multiple keys are truncated to the same value. The maximum output
        width will be ``2*max_length + 7``. The default of 36 produces output
        no longer than 79 characters wide.
    """

    def __init__(self, filename_or_file: str | TextIO, max_length: int = 36) -> None:
        self.max_length = max_length
        if isinstance(filename_or_file, str):
            self.file = open(filename_or_file, "w")
            self.own_file = True
        elif isinstance(filename_or_file, TextIOBase) or hasattr(filename_or_file, "write"):
            # Note: in theory `TextIOBase` check should be sufficient,
            # in practice, libraries don't always inherit from it, see GH#1598
            self.file = filename_or_file  # type: ignore[assignment]
            self.own_file = False
        else:
            raise ValueError(f"Expected file or str, got {filename_or_file}")

    def write(
        self,
        key_values: dict[str, Any],
        key_excluded: dict[str, tuple[str, ...] | None],
        step: int = 0,
    ) -> None:
        # Create strings for printing
        key2str = {}
        tag = ""
        for (key, value), (_, excluded) in zip(
            sorted(key_values.items()), sorted(key_excluded.items()), strict=False
        ):
            if excluded is not None and ("stdout" in excluded or "log" in excluded):
                continue

            elif isinstance(value, Video):
                raise FormatUnsupportedError(["stdout", "log"], "video")

            elif isinstance(value, Figure):
                raise FormatUnsupportedError(["stdout", "log"], "figure")

            elif isinstance(value, Image):
                raise FormatUnsupportedError(["stdout", "log"], "image")

            elif isinstance(value, HParam):
                raise FormatUnsupportedError(["stdout", "log"], "hparam")

            elif isinstance(value, float):
                # Align left
                value_str = f"{value:<8.3g}"
            else:
                value_str = str(value)

            if key.find("/") > 0:  # Find tag and add it to the dict
                tag = key[: key.find("/") + 1]
                key2str[(tag, self._truncate(tag))] = ""
            # Remove tag from key and indent the key
            if len(tag) > 0 and tag in key:
                key = f"{'':3}{key[len(tag) :]}"

            truncated_key = self._truncate(key)
            if (tag, truncated_key) in key2str:
                raise ValueError(
                    f"Key '{key}' truncated to '{truncated_key}' that already exists. Consider increasing `max_length`."
                )
            key2str[(tag, truncated_key)] = self._truncate(value_str)

        # Find max widths
        if len(key2str) == 0:
            warnings.warn("Tried to write empty key-value dict")
            return
        else:
            tagless_keys = map(lambda x: x[1], key2str.keys())
            key_width = max(map(len, tagless_keys))
            val_width = max(map(len, key2str.values()))

        # Write out the data
        dashes = "-" * (key_width + val_width + 7)
        lines = [dashes]
        for (_, key), value in key2str.items():
            key_space = " " * (key_width - len(key))
            val_space = " " * (val_width - len(value))
            lines.append(f"| {key}{key_space} | {value}{val_space} |")
        lines.append(dashes)

        if tqdm is not None and hasattr(self.file, "name") and self.file.name == "<stdout>":
            # Do not mess up with progress bar
            tqdm.write("\n".join(lines) + "\n", file=sys.stdout, end="")
        else:
            self.file.write("\n".join(lines) + "\n")

        # Flush the output to the file
        self.file.flush()

    def _truncate(self, string: str) -> str:
        if len(string) > self.max_length:
            string = string[: self.max_length - 3] + "..."
        return string

    def write_sequence(self, sequence: list[str]) -> None:
        for i, elem in enumerate(sequence):
            self.file.write(elem)
            if i < len(sequence) - 1:  # add space unless this is the last one
                self.file.write(" ")
        self.file.write("\n")
        self.file.flush()

    def close(self) -> None:
        """
        closes the file
        """
        if self.own_file:
            self.file.close()


def filter_excluded_keys(
    key_values: dict[str, Any], key_excluded: dict[str, tuple[str, ...] | None], _format: str
) -> dict[str, Any]:
    """
    Filters the keys specified by ``key_exclude`` for the specified format

    :param key_values: log dictionary to be filtered
    :param key_excluded: keys to be excluded per format
    :param _format: format for which this filter is run
    :return: dict without the excluded keys
    """

    def is_excluded(key: str) -> bool:
        return (
            key in key_excluded and key_excluded[key] is not None and _format in key_excluded[key] # type: ignore
        )

    return {key: value for key, value in key_values.items() if not is_excluded(key)}


class JSONOutputFormat(KVWriter):
    """
    Log to a file, in the JSON format

    :param filename: the file to write the log to
    """

    def __init__(self, filename: str):
        self.file = open(filename, "w")

    def write(
        self,
        key_values: dict[str, Any],
        key_excluded: dict[str, tuple[str, ...] | None],
        step: int = 0,
    ) -> None:
        def cast_to_json_serializable(value: Any):
            if isinstance(value, Video):
                raise FormatUnsupportedError(["json"], "video")
            if isinstance(value, Figure):
                raise FormatUnsupportedError(["json"], "figure")
            if isinstance(value, Image):
                raise FormatUnsupportedError(["json"], "image")
            if isinstance(value, HParam):
                raise FormatUnsupportedError(["json"], "hparam")
            if hasattr(value, "dtype"):
                if value.shape == () or len(value) == 1:
                    # if value is a dimensionless numpy array or of length 1, serialize as a float
                    return float(value.item())
                else:
                    # otherwise, a value is a numpy array, serialize as a list or nested lists
                    return value.tolist()
            return value

        key_values = {
            key: cast_to_json_serializable(value)
            for key, value in filter_excluded_keys(key_values, key_excluded, "json").items()
        }
        self.file.write(json.dumps(key_values) + "\n")
        self.file.flush()

    def close(self) -> None:
        """
        closes the file
        """

        self.file.close()


class CSVOutputFormat(KVWriter):
    """
    Log to a file, in a CSV format

    :param filename: the file to write the log to
    """

    def __init__(self, filename: str):
        self.file = open(filename, "w+")
        self.keys: list[str] = []
        self.separator = ","
        self.quotechar = '"'

    def write(
        self,
        key_values: dict[str, Any],
        key_excluded: dict[str, tuple[str, ...] | None],
        step: int = 0,
    ) -> None:
        # Add our current row to the history
        key_values = filter_excluded_keys(key_values, key_excluded, "csv")
        extra_keys = key_values.keys() - self.keys
        if extra_keys:
            self.keys.extend(extra_keys)
            self.file.seek(0)
            lines = self.file.readlines()
            self.file.seek(0)
            for i, key in enumerate(self.keys):
                if i > 0:
                    self.file.write(",")
                self.file.write(key)
            self.file.write("\n")
            for line in lines[1:]:
                self.file.write(line[:-1])
                self.file.write(self.separator * len(extra_keys))
                self.file.write("\n")
        for i, key in enumerate(self.keys):
            if i > 0:
                self.file.write(",")
            value = key_values.get(key)

            if isinstance(value, Video):
                raise FormatUnsupportedError(["csv"], "video")

            elif isinstance(value, Figure):
                raise FormatUnsupportedError(["csv"], "figure")

            elif isinstance(value, Image):
                raise FormatUnsupportedError(["csv"], "image")

            elif isinstance(value, HParam):
                raise FormatUnsupportedError(["csv"], "hparam")

            elif isinstance(value, str):
                # escape quotechars by prepending them with another quotechar
                value = value.replace(self.quotechar, self.quotechar + self.quotechar)

                # additionally wrap text with quotechars so that any delimiters in the text are ignored by csv readers
                self.file.write(self.quotechar + value + self.quotechar)

            elif value is not None:
                self.file.write(str(value))
        self.file.write("\n")
        self.file.flush()

    def close(self) -> None:
        """
        closes the file
        """
        self.file.close()


class TensorBoardOutputFormat(KVWriter):
    """
    Dumps key/value pairs into TensorBoard's numeric format.

    :param folder: the folder to write the log to
    """

    def __init__(self, folder: str):
        assert (
            SummaryWriter is not None
        ), "tensorboard is not installed, you can use `pip install tensorboard` to do so"
        self.writer = SummaryWriter(log_dir=folder)
        self._is_closed = False

    def write(
        self,
        key_values: dict[str, Any],
        key_excluded: dict[str, tuple[str, ...] | None],
        step: int = 0,
    ) -> None:
        assert not self._is_closed, "The SummaryWriter was closed, please re-create one."
        for (key, value), (_, excluded) in zip(
            sorted(key_values.items()), sorted(key_excluded.items()), strict=False
        ):
            if excluded is not None and "tensorboard" in excluded:
                continue

            if isinstance(value, np.ScalarType):
                if isinstance(value, str):
                    # str is considered a np.ScalarType
                    self.writer.add_text(key, value, step)
                else:
                    self.writer.add_scalar(key, value, step)

            if isinstance(value, th.Tensor):
                self.writer.add_histogram(key, value, step)

            if isinstance(value, Video):
                self.writer.add_video(key, value.frames, step, int(value.fps))

            if isinstance(value, Figure):
                self.writer.add_figure(key, value.figure, step, close=value.close)

            if isinstance(value, Image):
                self.writer.add_image(key, value.image, step, dataformats=value.dataformats)

            if isinstance(value, HParam):
                # we don't use `self.writer.add_hparams` to have control over the log_dir
                experiment, session_start_info, session_end_info = hparams( # type: ignore
                    value.hparam_dict, metric_dict=value.metric_dict
                )
                self.writer.file_writer.add_summary(experiment) # type: ignore
                self.writer.file_writer.add_summary(session_start_info) # type: ignore
                self.writer.file_writer.add_summary(session_end_info) # type: ignore

        # Flush the output to the file
        self.writer.flush()

    def close(self) -> None:
        """
        closes the file
        """
        if self.writer:
            self.writer.close()
            self._is_closed = True


class WandbOutputFormat(KVWriter):
    """
    Log to Weights & Biases (wandb)

    :param project: wandb project name
    :param entity: wandb entity (your username or team name)
    :param config: dictionary of hyperparameters or configurations
    """

    def __init__(
        self,
        project: str,
        entity: str | None = None,
        config: dict[str, Any] | None = None,
        log_dir: str | None = None,
        mode: Literal["online", "offline", "disabled"] = "online",
        exp_name: str | None = None,
    ):
        if log_dir is not None:
            os.environ["WANDB_DIR"] = log_dir
        wandb.init(project=project, entity=entity, config=config, mode=mode, name=exp_name)
        self.run = wandb.run

    def write(
        self,
        key_values: dict[str, Any],
        key_excluded: dict[str, tuple[str, ...] | None],
        step: int = 0,
    ) -> None:
        log_dict = filter_excluded_keys(key_values, key_excluded, "wandb")

        for key, value in log_dict.items():
            if isinstance(value, float | int):
                wandb.log({key: value}, step=step)
            elif isinstance(value, Image):
                wandb.log({key: wandb.Image(value.image, caption=key)}, step=step)
            elif isinstance(value, Figure):
                wandb.log({key: wandb.Image(value.figure, caption=key)}, step=step)
            elif isinstance(value, Video):
                wandb.log({key: wandb.Video(value.frames, fps=int(value.fps), format="mp4")}, step=step) # type: ignore
            elif isinstance(value, HParam):
                wandb.config.update(value.hparam_dict)
                wandb.log(value.metric_dict, step=step) # type: ignore
            else:
                # For other types, log as text
                wandb.log({key: str(value)}, step=step)

    def close(self) -> None:
        wandb.finish()


def make_output_format(_format: str, log_dir: str, log_suffix: str = "", **kwargs) -> KVWriter:
    os.makedirs(log_dir, exist_ok=True)
    if _format == "stdout":
        return HumanOutputFormat(sys.stdout)
    elif _format == "log":
        return HumanOutputFormat(os.path.join(log_dir, f"log{log_suffix}.txt"))
    elif _format == "json":
        return JSONOutputFormat(os.path.join(log_dir, f"progress{log_suffix}.json"))
    elif _format == "csv":
        return CSVOutputFormat(os.path.join(log_dir, f"progress{log_suffix}.csv"))
    elif _format == "tensorboard":
        return TensorBoardOutputFormat(log_dir)
    elif _format == "wandb":
        return WandbOutputFormat(
            project=kwargs.get("project", "gsrl"),
            entity=kwargs.get("entity", None),
            config=kwargs.get("config", None),
            log_dir=log_dir,
            mode=kwargs.get("mode", "online"),
            exp_name=kwargs.get("exp_name", None)
        )
    else:
        raise ValueError(f"Unknown format specified: {_format}")


# ================================================================
# Backend
# ================================================================


class Logger:
    """
    The logger class.

    :param folder: the logging location
    :param output_formats: the list of output formats
    """

    def __init__(self, folder: str | None, output_formats: list[KVWriter]) -> None:
        self.name_to_value: dict[str, float] = defaultdict(float)  # values this iteration
        self.name_to_count: dict[str, int] = defaultdict(int)
        self.name_to_excluded: dict[str, tuple[str, ...]] = {}
        self.level = INFO
        self.dir = folder
        self.output_formats = output_formats

    @staticmethod
    def to_tuple(
        string_or_tuple: str | tuple[str, ...] | None,
    ) -> tuple[str, ...] | tuple[str]:
        """
        Helper function to convert str to tuple of str.
        """
        if string_or_tuple is None:
            return ("",)
        if isinstance(string_or_tuple, tuple):
            return string_or_tuple
        return (string_or_tuple,)

    def record(
        self,
        key: str,
        value: Any,
        exclude: str | tuple[str, ...] | None = None,
    ) -> None:
        """
        Log a value of some diagnostic
        Call this once for each diagnostic quantity, each iteration
        If called many times, last value will be used.

        :param key: save to log this key
        :param value: save to log this value
        :param exclude: outputs to be excluded
        """
        self.name_to_value[key] = value
        self.name_to_excluded[key] = self.to_tuple(exclude)

    def record_mean(
        self,
        key: str,
        value: float | None,
        exclude: str | tuple[str, ...] | None = None,
    ) -> None:
        """
        The same as record(), but if called many times, values averaged.

        :param key: save to log this key
        :param value: save to log this value
        :param exclude: outputs to be excluded
        """
        if value is None:
            return
        old_val, count = self.name_to_value[key], self.name_to_count[key]
        self.name_to_value[key] = old_val * count / (count + 1) + value / (count + 1)
        self.name_to_count[key] = count + 1
        self.name_to_excluded[key] = self.to_tuple(exclude)

    def dump(self, step: int = 0) -> None:
        """
        Write all of the diagnostics from the current iteration
        """
        if self.level == DISABLED:
            return
        for _format in self.output_formats:
            if isinstance(_format, KVWriter): # type: ignore
                _format.write(self.name_to_value, self.name_to_excluded, step) # type: ignore

        self.name_to_value.clear()
        self.name_to_count.clear()
        self.name_to_excluded.clear()

    def log(self, *args, level: int = INFO) -> None:
        """
        Write the sequence of args, with no separators,
        to the console and output files (if you've configured an output file).

        level: int. (see logger.py docs) If the global logger level is higher than
                    the level argument here, don't print to stdout.

        :param args: log the arguments
        :param level: the logging level (can be DEBUG=10, INFO=20, WARN=30, ERROR=40, DISABLED=50)
        """
        if self.level <= level:
            self._do_log(args)

    def debug(self, *args) -> None:
        """
        Write the sequence of args, with no separators,
        to the console and output files (if you've configured an output file).
        Using the DEBUG level.

        :param args: log the arguments
        """
        self.log(*args, level=DEBUG)

    def info(self, *args) -> None:
        """
        Write the sequence of args, with no separators,
        to the console and output files (if you've configured an output file).
        Using the INFO level.

        :param args: log the arguments
        """
        self.log(*args, level=INFO)

    def warn(self, *args) -> None:
        """
        Write the sequence of args, with no separators,
        to the console and output files (if you've configured an output file).
        Using the WARN level.

        :param args: log the arguments
        """
        self.log(*args, level=WARN)

    def error(self, *args) -> None:
        """
        Write the sequence of args, with no separators,
        to the console and output files (if you've configured an output file).
        Using the ERROR level.

        :param args: log the arguments
        """
        self.log(*args, level=ERROR)

    # Configuration
    # ----------------------------------------
    def set_level(self, level: int) -> None:
        """
        Set logging threshold on current logger.

        :param level: the logging level (can be DEBUG=10, INFO=20, WARN=30, ERROR=40, DISABLED=50)
        """
        self.level = level

    def get_dir(self) -> str | None:
        """
        Get directory that log files are being written to.
        will be None if there is no output directory (i.e., if you didn't call start)

        :return: the logging directory
        """
        return self.dir

    def close(self) -> None:
        """
        closes the file
        """
        for _format in self.output_formats:
            _format.close()

    # Misc
    # ----------------------------------------
    def _do_log(self, args: tuple[Any, ...]) -> None:
        """
        log to the requested format outputs

        :param args: the arguments to log
        """
        for _format in self.output_formats:
            if isinstance(_format, SeqWriter):
                _format.write_sequence(list(map(str, args)))


def configure(folder: str | None = None, format_strings: list[str] | None = None, **kwargs) -> Logger:
    """
    Configure the current logger.

    :param folder: the save location
        (if None, $GSRL_LOGDIR, if still None, tempdir/GSRL-[date & time])
    :param format_strings: the output logging format
        (if None, $GSRL_LOG_FORMAT, if still None, ['stdout', 'log', 'csv'])
    :return: The logger object.
    """
    if folder is None:
        folder = os.getenv("GSRL_LOGDIR")
    if folder is None:
        folder = os.path.join(
            tempfile.gettempdir(),
            datetime.datetime.now().strftime("GSRL-%Y-%m-%d-%H-%M-%S-%f"),
        )
    assert isinstance(folder, str)
    os.makedirs(folder, exist_ok=True)

    log_suffix = ""
    if format_strings is None:
        format_strings = os.getenv("GSRL_LOG_FORMAT", "stdout,log,csv").split(",")

    format_strings = list(filter(None, format_strings))
    output_formats = [make_output_format(f, folder, log_suffix, **kwargs) for f in format_strings]

    logger = Logger(folder=folder, output_formats=output_formats)
    # Only print when some files will be saved
    if len(format_strings) > 0 and format_strings != ["stdout"]:
        logger.log(f"Logging to {folder}")
    return logger


# ================================================================
# Readers
# ================================================================


def read_json(filename: str) -> pandas.DataFrame:
    """
    read a json file using pandas

    :param filename: the file path to read
    :return: the data in the json
    """
    data = []
    with open(filename) as file_handler:
        for line in file_handler:
            data.append(json.loads(line))
    return pandas.DataFrame(data)


def read_csv(filename: str) -> pandas.DataFrame:
    """
    read a csv file using pandas

    :param filename: the file path to read
    :return: the data in the csv
    """
    return pandas.read_csv(filename, index_col=None, comment="#")
