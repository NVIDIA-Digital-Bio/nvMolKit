#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SMARTS_DIR="${SCRIPT_DIR}/test_data/SMARTS"
SMARTS_FILTER="${1:-${SCRIPT_DIR}/../build/tests/smarts_filter}"

if [[ ! -x "$SMARTS_FILTER" ]]; then
    echo "Error: smarts_filter executable not found at: $SMARTS_FILTER"
    echo "Usage: $0 [path_to_smarts_filter]"
    exit 1
fi

for file in "$SMARTS_DIR"/*.txt; do
    base=$(basename "$file" .txt)
    
    # Skip files that are already outputs
    [[ "$base" == *_supported ]] && continue
    [[ "$base" == *_unsupported ]] && continue
    [[ "$base" == *_invalid ]] && continue
    echo " ----------------------"    
    echo "Processing: $base.txt"
    "$SMARTS_FILTER" \
        "$file" \
        "${SMARTS_DIR}/${base}_supported.txt" \
        "${SMARTS_DIR}/${base}_unsupported.txt" \
        "${SMARTS_DIR}/${base}_invalid.txt"
done

