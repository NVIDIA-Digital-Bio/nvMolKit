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

"""Autotune wrapper for substructure search.

Tunes :class:`SubstructSearchConfig` against one of the user-facing entry
points (:func:`hasSubstructMatch`, :func:`countSubstructMatches`, or
:func:`getSubstructMatches`). Throughput is reported in target-query pairs per
second.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Iterable, Optional, Sequence

from rdkit.Chem import Mol

from nvmolkit.autotune._calibration import normalize_calibration_set
from nvmolkit.autotune._core import (
    CalibrationState,
    TuneResult,
    _require_optuna,
    collect_int_from_space,
    resolve_search_space,
    run_study,
    suggest_from_space,
)
from nvmolkit.substructure import (
    SubstructSearchConfig,
    countSubstructMatches,
    getSubstructMatches,
    hasSubstructMatch,
)

_API_FUNCTIONS = {
    hasSubstructMatch,
    countSubstructMatches,
    getSubstructMatches,
}

_DEFAULT_SUBSTRUCT_SEARCH_SPACE = {
    "batchSize": (128, 8192, "log"),
    "workerThreads": (1, 8),
    "preprocessingThreads": (1, min(32, max(1, os.cpu_count() or 1))),
}


def tune_substructure(
    targets: Sequence[Mol],
    queries: Sequence[Mol],
    *,
    api: Callable = hasSubstructMatch,
    maxMatches: int = 0,
    uniquify: bool = False,
    gpuIds: Optional[Iterable[int]] = None,
    calibration_set: Optional[Iterable[int]] = None,
    calibration_fraction: float = 0.1,
    calibration_max_size: int = 2000,
    target_seconds_per_trial: float = 10.0,
    n_trials: int = 30,
    search_space_overrides: Optional[dict[str, Any]] = None,
    sampler: Any = None,
    seed: Optional[int] = None,
    verbose: bool = False,
) -> TuneResult:
    """Tune :class:`SubstructSearchConfig` for a substructure-search workflow.

    Args:
        targets: Library of target molecules. Calibration trials are run on a
            (possibly auto-subsampled) slice of these targets.
        queries: Query molecules. The same query set is used for every trial.
        api: Which substructure-search entry point to tune. One of
            :func:`hasSubstructMatch`, :func:`countSubstructMatches`, or
            :func:`getSubstructMatches`.
        maxMatches: ``maxMatches`` argument forwarded to the resulting config.
            Held constant across trials.
        uniquify: ``uniquify`` flag forwarded to the resulting config.
        gpuIds: GPU device IDs to use. Fixed across the study.
        calibration_set: Optional explicit indices into ``targets``.
        calibration_fraction: Fraction of the workload to auto-sample.
        calibration_max_size: Cap on the auto-sampled calibration size.
        target_seconds_per_trial: Target wall-clock budget for one trial.
        n_trials: Number of Optuna trials to run after warm-up.
        search_space_overrides: Optional overrides for ``batchSize``,
            ``workerThreads``, or ``preprocessingThreads`` ranges.
        sampler: Optional Optuna sampler.
        seed: Seed for the default sampler.
        verbose: Print warm-up and trial diagnostics.

    Returns:
        :class:`TuneResult` with ``best_config`` set to a fully-populated
        :class:`SubstructSearchConfig` instance.
    """
    optuna = _require_optuna()  # noqa: F841

    if api not in _API_FUNCTIONS:
        raise ValueError(
            "api must be one of nvmolkit.substructure.hasSubstructMatch, "
            "countSubstructMatches, or getSubstructMatches"
        )
    if not targets:
        raise ValueError("targets must be non-empty for autotuning")
    if not queries:
        raise ValueError("queries must be non-empty for autotuning")

    indices = normalize_calibration_set(
        calibration_set,
        len(targets),
        fraction=calibration_fraction,
        max_size=calibration_max_size,
    )
    space = resolve_search_space(_DEFAULT_SUBSTRUCT_SEARCH_SPACE, search_space_overrides)
    fixed_gpu_ids = list(gpuIds) if gpuIds is not None else []

    def _make_config(values: dict[str, Any]) -> SubstructSearchConfig:
        return SubstructSearchConfig(
            batchSize=int(values.get("batchSize", 1024)),
            workerThreads=int(values.get("workerThreads", -1)),
            preprocessingThreads=int(values.get("preprocessingThreads", -1)),
            maxMatches=int(maxMatches),
            uniquify=bool(uniquify),
            gpuIds=fixed_gpu_ids if fixed_gpu_ids else None,
        )

    queries_list = list(queries)

    def _run_once(config: SubstructSearchConfig, state: CalibrationState) -> int:
        target_slice = [targets[i] for i in state.indices]
        api(target_slice, queries_list, config)
        return len(target_slice) * len(queries_list)

    def default_runner(state: CalibrationState) -> int:
        return _run_once(
            SubstructSearchConfig(
                maxMatches=int(maxMatches),
                uniquify=bool(uniquify),
                gpuIds=fixed_gpu_ids if fixed_gpu_ids else None,
            ),
            state,
        )

    def trial_runner(trial, state: CalibrationState) -> int:
        values = {name: suggest_from_space(trial, name, spec) for name, spec in space.items()}
        return _run_once(_make_config(values), state)

    def build_config(params_dict: dict[str, Any]) -> SubstructSearchConfig:
        merged = {name: params_dict.get(name, collect_int_from_space(spec)) for name, spec in space.items()}
        return _make_config(merged)

    initial_state = CalibrationState(indices=list(indices), items_per_trial=len(indices) * len(queries_list))
    return run_study(
        default_runner=default_runner,
        trial_runner=trial_runner,
        build_config=build_config,
        initial_state=initial_state,
        n_trials=n_trials,
        target_seconds_per_trial=target_seconds_per_trial,
        sampler=sampler,
        seed=seed,
        verbose=verbose,
    )
