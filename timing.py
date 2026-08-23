"""Generic timing and resource-usage instrumentation."""

import resource
import sys
import time
from typing import Any


class TimingContext:
    """Context manager for measuring wall clock time, CPU time, and resource usage.

    Captures timing and resource metrics on entry/exit, with optional auto-print.
    Reusable across different methods and scenarios.
    """

    def __init__(self, label: str = "", auto_print: bool = True) -> None:
        """Initialize timing context.

        Parameters
        ----------
        label : str
            Optional label for this timing block (used in output)
        auto_print : bool
            If True, print summary on context exit. Default: True
        """
        self.label = label
        self.auto_print = auto_print

        # Timing metrics
        self.wall_time: float = 0.0
        self.user_time: float = 0.0
        self.system_time: float = 0.0

        # Resource metrics from getrusage
        self.peak_memory_kb: float = 0.0  # ru_maxrss
        self.minor_page_faults: int = 0  # ru_minflt (memory not on disk)
        self.major_page_faults: int = 0  # ru_majflt (memory on disk, required I/O)
        self.voluntary_context_switches: int = 0  # ru_nvcsw (yield/blocking)
        self.involuntary_context_switches: int = 0  # ru_nivcsw (preemption)
        self.input_blocks: int = 0  # ru_inblock
        self.output_blocks: int = 0  # ru_oublock

        # Internal state
        self._start_wall: float = 0.0
        self._start_rusage: resource.struct_rusage | None = None

    def __enter__(self) -> "TimingContext":
        """Start timing."""
        self._start_wall = time.perf_counter()
        self._start_rusage = resource.getrusage(resource.RUSAGE_SELF)
        return self

    def __exit__(self, *args: Any) -> None:
        """Stop timing and optionally print summary."""
        end_wall = time.perf_counter()
        end_rusage = resource.getrusage(resource.RUSAGE_SELF)

        assert self._start_rusage is not None

        # Calculate deltas
        self.wall_time = end_wall - self._start_wall
        self.user_time = end_rusage.ru_utime - self._start_rusage.ru_utime
        self.system_time = end_rusage.ru_stime - self._start_rusage.ru_stime
        self.peak_memory_kb = float(end_rusage.ru_maxrss)
        self.minor_page_faults = end_rusage.ru_minflt - self._start_rusage.ru_minflt
        self.major_page_faults = end_rusage.ru_majflt - self._start_rusage.ru_majflt
        self.voluntary_context_switches = (
            end_rusage.ru_nvcsw - self._start_rusage.ru_nvcsw
        )
        self.involuntary_context_switches = (
            end_rusage.ru_nivcsw - self._start_rusage.ru_nivcsw
        )
        self.input_blocks = end_rusage.ru_inblock - self._start_rusage.ru_inblock
        self.output_blocks = end_rusage.ru_oublock - self._start_rusage.ru_oublock

        if self.auto_print:
            print(self.summary())

    def summary(self) -> str:
        """Return formatted timing and resource summary."""
        lines = []
        if self.label:
            lines.append(f"\n***** Timing: {self.label}")
        else:
            lines.append("\n***** Timing Summary")

        # CPU and wall clock timing
        cpu_total = self.user_time + self.system_time
        cpu_pct = (cpu_total / self.wall_time * 100) if self.wall_time > 0 else 0.0

        lines.append(f"Wall clock time:           {self.wall_time:8.3f} s")
        lines.append(f"User CPU time:             {self.user_time:8.3f} s")
        lines.append(f"System CPU time:           {self.system_time:8.3f} s")
        lines.append(f"Total CPU time:            {cpu_total:8.3f} s ({cpu_pct:5.1f}%)")

        # Memory and page faults
        # On macOS and BSD, ru_maxrss is in bytes; on Linux it's in KB
        if sys.platform == "darwin" or sys.platform.startswith("freebsd"):
            peak_memory_mb = self.peak_memory_kb / (1024 * 1024)  # bytes to MB
        else:  # Linux
            peak_memory_mb = self.peak_memory_kb / 1024  # KB to MB
        lines.append(f"Peak memory:               {peak_memory_mb:8.1f} MB")
        lines.append(f"Minor page faults:         {self.minor_page_faults:8d}")
        lines.append(f"Major page faults:         {self.major_page_faults:8d}")

        # Context switches
        lines.append(
            f"Voluntary context switches: {self.voluntary_context_switches:8d}"
        )
        lines.append(
            f"Involuntary context switches: {self.involuntary_context_switches:8d}"
        )

        # I/O
        lines.append(f"Input blocks (fsync):      {self.input_blocks:8d}")
        lines.append(f"Output blocks (fsync):     {self.output_blocks:8d}")

        return "\n".join(lines)
