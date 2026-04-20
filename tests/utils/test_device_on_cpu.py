# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for verl.utils.device — runs on CPU, no accelerator required."""

import unittest
from unittest import mock


class TestIsSupportIpcRocm(unittest.TestCase):
    """Tests for the is_support_ipc ROCm short-circuit added in this branch.

    ROCm emulates the CUDA API so is_cuda_available is True on ROCm, but ROCm
    does not support CUDA IPC.  is_support_ipc() must return False when
    is_rocm_available is True, regardless of is_cuda_available.
    """

    def test_returns_false_when_rocm_available(self):
        """is_support_ipc must be False on ROCm even though CUDA API is present."""
        import verl.utils.device as device_mod

        with mock.patch.object(device_mod, "is_rocm_available", True), \
             mock.patch.object(device_mod, "is_cuda_available", True):
            result = device_mod.is_support_ipc()

        self.assertFalse(result, "is_support_ipc should return False on ROCm")

    def test_returns_true_when_cuda_not_rocm(self):
        """is_support_ipc must be True when CUDA is available but ROCm is not."""
        import verl.utils.device as device_mod

        with mock.patch.object(device_mod, "is_rocm_available", False), \
             mock.patch.object(device_mod, "is_cuda_available", True):
            result = device_mod.is_support_ipc()

        self.assertTrue(result, "is_support_ipc should return True on plain CUDA")

    def test_returns_false_when_no_accelerator(self):
        """is_support_ipc must be False on CPU-only machines."""
        import verl.utils.device as device_mod

        with mock.patch.object(device_mod, "is_rocm_available", False), \
             mock.patch.object(device_mod, "is_cuda_available", False), \
             mock.patch.object(device_mod, "is_npu_available", False):
            result = device_mod.is_support_ipc()

        self.assertFalse(result, "is_support_ipc should return False with no accelerator")

    def test_rocm_takes_precedence_over_cuda(self):
        """The ROCm guard must execute before the CUDA guard, not after."""
        import verl.utils.device as device_mod

        # Both flags True — ROCm check must win and return False.
        with mock.patch.object(device_mod, "is_rocm_available", True), \
             mock.patch.object(device_mod, "is_cuda_available", True):
            result = device_mod.is_support_ipc()

        self.assertFalse(result, "ROCm guard must take precedence over CUDA guard")


if __name__ == "__main__":
    unittest.main()
