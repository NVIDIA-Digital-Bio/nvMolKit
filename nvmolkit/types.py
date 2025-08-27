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

"""Types facilitating GPU-accelerated operations."""
import torch

class AsyncGpuResult:
    """Handle to a GPU result.

    Populates the __cuda_array_interface__ attribute which can be consumed by other libraries. Note that
    this result is async, and the data cannot be accessed without a sync, such as torch.cuda.synchronize().

    # TODO: Handle devices and streams.
    """
    def __init__(self, obj):
        """Internal construction of the AsyncGpuResult object."""
        if not hasattr(obj, '__cuda_array_interface__'):
            raise TypeError(f"Object {obj} does not have a __cuda_array_interface__ attribute")
        self.arr = torch.as_tensor(obj, device='cuda')

    @property
    def __cuda_array_interface__(self):
        """Return the CUDA array interface for the underlying data."""
        return self.arr.__cuda_array_interface__

    @property
    def device(self):
        """Return the device of the underlying data."""
        return self.arr.device

    def torch(self):
        """Return the underlying data as a torch tensor. This is an asynchronous operation."""
        return self.arr

    def numpy(self):
        """Return the underlying data as a numpy array. This is a blocking operation."""
        torch.cuda.synchronize()
        return self.arr.cpu().numpy()
