# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared timing utilities for nvMolKit benchmarks."""

import statistics
import time
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class TimingResult:
    """Holds timing results from a benchmark run."""

    times_ms: list[float] = field(default_factory=list)

    @property
    def median_ms(self) -> float:
        """Median time in milliseconds."""
        return statistics.median(self.times_ms)

    @property
    def mean_ms(self) -> float:
        """Mean time in milliseconds."""
        return statistics.mean(self.times_ms)

    @property
    def std_ms(self) -> float:
        """Sample standard deviation in milliseconds."""
        if len(self.times_ms) < 2:
            return 0.0
        return statistics.stdev(self.times_ms)

    @property
    def median_s(self) -> float:
        """Median time in seconds."""
        return self.median_ms / 1000.0


def time_it(func: Callable, runs: int = 3, warmups: int = 1, gpu_sync: bool = False) -> TimingResult:
    """Time a callable with warmup iterations and optional CUDA synchronization.

    Args:
        func: Zero-argument callable to benchmark.
        runs: Number of timed iterations.
        warmups: Number of untimed warmup iterations.
        gpu_sync: If True, call torch.cuda.synchronize() before and after each
                  timed iteration to ensure GPU work is included in the measurement.

    Returns:
        A TimingResult with per-iteration times in milliseconds.
    """
    if gpu_sync:
        import torch

        sync = torch.cuda.synchronize
    else:

        def sync() -> None:
            pass

    for _ in range(warmups):
        func()
        sync()

    if runs <= 0:
        raise ValueError(f"runs must be positive, got {runs}")

    times_ms = []
    for _ in range(runs):
        sync()
        t0 = time.perf_counter()
        func()
        sync()
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)

    return TimingResult(times_ms=times_ms)
