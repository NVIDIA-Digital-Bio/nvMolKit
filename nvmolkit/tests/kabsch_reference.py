# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Shared numpy Kabsch RMSD reference used by tests and benchmarks."""

import numpy as np


def numpy_kabsch_rmsd(p, q):
    """Independent Kabsch RMSD using numpy SVD (gold reference).

    Args:
        p: (N, 3) array of atom coordinates for the first conformer.
        q: (N, 3) array of atom coordinates for the second conformer.

    Returns:
        float: Kabsch-aligned RMSD between p and q.
    """
    p_c = p - p.mean(axis=0)
    q_c = q - q.mean(axis=0)
    H = p_c.T @ q_c
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(H))
    S[-1] *= d if d != 0.0 else 1.0
    Sp = np.sum(p_c ** 2)
    Sq = np.sum(q_c ** 2)
    return np.sqrt(max((Sp + Sq - 2.0 * np.sum(S)) / len(p), 0.0))
