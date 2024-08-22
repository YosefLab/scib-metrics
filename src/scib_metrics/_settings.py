import logging
import os
from typing import Literal

from rich.console import Console
from rich.logging import RichHandler

scib_logger = logging.getLogger("scib_metrics")


class ScibConfig:
    """Config manager for scib-metrics.

    Examples
    --------
    To set the progress bar style, choose one of "rich", "tqdm"

    >>> scib_metrics.settings.progress_bar_style = "rich"

    To set the verbosity

    >>> import logging
    >>> scib_metrics.settings.verbosity = logging.INFO
    """

    def __init__(
        self,
        verbosity: int = logging.INFO,
        progress_bar_style: Literal["rich", "tqdm"] = "tqdm",
        jax_preallocate_gpu_memory: bool = False,
    ):
        if progress_bar_style not in ["rich", "tqdm"]:
            raise ValueError("Progress bar style must be in ['rich', 'tqdm']")
        self.progress_bar_style = progress_bar_style
        self.jax_preallocate_gpu_memory = jax_preallocate_gpu_memory
        self.verbosity = verbosity

    @property
    def progress_bar_style(self) -> str:
        """Library to use for progress bar."""
        return self._pbar_style

    @progress_bar_style.setter
    def progress_bar_style(self, pbar_style: Literal["tqdm", "rich"]):
        """Library to use for progress bar."""
        self._pbar_style = pbar_style

    @property
    def verbosity(self) -> int:
        """Verbosity level (default `logging.INFO`).

        Returns
        -------
        verbosity: int
        """
        return self._verbosity

    @verbosity.setter
    def verbosity(self, level: str | int):
        """Set verbosity level.

        If "scib_metrics" logger has no StreamHandler, add one.
        Else, set its level to `level`.

        Parameters
        ----------
        level
            Sets "scib_metrics" logging level to `level`
        force_terminal
            Rich logging option, set to False if piping to file output.
        """
        self._verbosity = level
        scib_logger.setLevel(level)
        if len(scib_logger.handlers) == 0:
            console = Console(force_terminal=True)
            if console.is_jupyter is True:
                console.is_jupyter = False
            ch = RichHandler(level=level, show_path=False, console=console, show_time=False)
            formatter = logging.Formatter("%(message)s")
            ch.setFormatter(formatter)
            scib_logger.addHandler(ch)
        else:
            scib_logger.setLevel(level)

    def reset_logging_handler(self) -> None:
        """Reset "scib_metrics" log handler to a basic RichHandler().

        This is useful if piping outputs to a file.

        Returns
        -------
        None
        """
        scib_logger.removeHandler(scib_logger.handlers[0])
        ch = RichHandler(level=self._verbosity, show_path=False, show_time=False)
        formatter = logging.Formatter("%(message)s")
        ch.setFormatter(formatter)
        scib_logger.addHandler(ch)

    def jax_fix_no_kernel_image(self) -> None:
        """Fix for JAX error "No kernel image is available for execution on the device"."""
        os.environ["XLA_FLAGS"] = "--xla_gpu_force_compilation_parallelism=1"

    @property
    def jax_preallocate_gpu_memory(self):
        """Jax GPU memory allocation settings.

        If False, Jax will ony preallocate GPU memory it needs.
        If float in (0, 1), Jax will preallocate GPU memory to that
        fraction of the GPU memory.

        Returns
        -------
        jax_preallocate_gpu_memory: bool or float
        """
        return self._jax_gpu

    @jax_preallocate_gpu_memory.setter
    def jax_preallocate_gpu_memory(self, value: float | bool):
        # see https://jax.readthedocs.io/en/latest/gpu_memory_allocation.html#gpu-memory-allocation
        if value is False:
            os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        elif isinstance(value, float):
            if value >= 1 or value <= 0:
                raise ValueError("Need to use a value between 0 and 1")
            # format is ".XX"
            os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(value)[1:4]
        else:
            raise ValueError("value not understood, need bool or float in (0, 1)")
        self._jax_gpu = value


settings = ScibConfig()
