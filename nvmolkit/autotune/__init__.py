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

"""Autotuning support utilities for nvMolKit hardware options.

This subpackage exposes the persistence helpers (:func:`load`, :func:`save`)
and shared building blocks used by tuning workflows. The optional ``optuna``
dependency is only required by callers that drive a study directly via
:mod:`nvmolkit.autotune._core`; importing this package never requires it, and
:func:`is_available` reports whether ``optuna`` is importable.

"""

from nvmolkit.autotune._core import (
    OPTUNA_INSTALL_HINT,
    CalibrationState,
    TuneResult,
    is_optuna_available,
)
from nvmolkit.autotune._persistence import load, save
from nvmolkit.autotune.tune_substructure import tune_substructure


def is_available() -> bool:
    """Return ``True`` if autotune dependencies are present (currently optuna)."""
    return is_optuna_available()


__all__ = [
    "CalibrationState",
    "OPTUNA_INSTALL_HINT",
    "TuneResult",
    "is_available",
    "is_optuna_available",
    "load",
    "save",
    "tune_substructure",
]
