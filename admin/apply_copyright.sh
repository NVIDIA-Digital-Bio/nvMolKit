#!/bin/bash
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

set -ex
REPO_ROOT=$(git rev-parse --show-toplevel)

# Define the Python copyright header
PY_COPYRIGHT_HEADER='# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

'

# Find all Python files and add the header (excluding rdkit_extensions folder)
find "$REPO_ROOT" -name "*.py" -not -path "*/rdkit_extensions/*" | while read -r file; do
    # Check if the file already contains the header
    if ! grep -q "SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES" "$file"; then
        # Add the header to the file
        echo "Adding copyright header to Python file: $file"
        echo "$PY_COPYRIGHT_HEADER$(cat "$file")" > "$file"
    else
        echo "Python file already has copyright header: $file"
    fi
done

# Define the C++ copyright header
CPP_COPYRIGHT_HEADER='// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

'

# Find all C++ files and add the header (excluding rdkit_extensions folder)
find "$REPO_ROOT" -type f \( -name "*.h" -o -name "*.cpp" -o -name "*.cuh" -o -name "*.cu" \) -not -path "*/rdkit_extensions/*" | while read -r file; do
    # Check if the file already contains the header
    if ! grep -q "SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES" "$file"; then
        # Add the header to the file
        echo "Adding copyright header to C++ file: $file"
        echo "$CPP_COPYRIGHT_HEADER$(cat "$file")" > "$file"
    else
        echo "C++ file already has copyright header: $file"
    fi
done

# Define the shell script copyright header
SH_COPYRIGHT_HEADER='# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

'

# Find all shell script files and add the header
find "$REPO_ROOT" -name "*.sh" | while read -r file; do
    # Check if the file already contains the header
    if ! grep -q "SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES" "$file"; then
        # Check if file starts with shebang and preserve it
        if head -n1 "$file" | grep -q "^#!"; then
            # File has shebang, add header after it
            echo "Adding copyright header to shell script (after shebang): $file"
            shebang=$(head -n1 "$file")
            rest=$(tail -n +2 "$file")
            echo "$shebang
$SH_COPYRIGHT_HEADER$rest" > "$file"
        else
            # No shebang, add header at the beginning
            echo "Adding copyright header to shell script: $file"
            echo "$SH_COPYRIGHT_HEADER$(cat "$file")" > "$file"
        fi
    else
        echo "Shell script already has copyright header: $file"
    fi
done

# Define the CMake copyright header
CMAKE_COPYRIGHT_HEADER='# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

'

# Find all CMake files and add the header
find "$REPO_ROOT" -type f \( -name "CMakeLists.txt" -o -name "*.cmake" \) | while read -r file; do
    # Check if the file already contains the header
    if ! grep -q "SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES" "$file"; then
        # Add the header to the file
        echo "Adding copyright header to CMake file: $file"
        echo "$CMAKE_COPYRIGHT_HEADER$(cat "$file")" > "$file"
        # Format the CMake file after adding copyright header
        echo "Formatting CMake file: $file"
        cmake-format -i "$file"
    else
        echo "CMake file already has copyright header: $file"
        # Still format the file to ensure consistent formatting
        echo "Formatting CMake file: $file"
        cmake-format -i "$file"
    fi
done
