from pathlib import Path
import sys

import torch
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)
from torch.utils import _capture_launch as cl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "serving"))
import vllm_launches as launches


class TestVllmLaunches(TestCase):
    def test_scheduler_reset_order(self):
        metadata = torch.empty(4, dtype=torch.int32)
        kernel = [0, 123, 1, 1, 1, 32, 1, 1, 1, 1, 1, 0, 0, 8, 456]
        sink = torch.tensor([3, *kernel, 1, metadata.data_ptr() + 12, 4, *kernel])
        owner = object()
        got = launches._parse_sink(sink, owner, metadata)
        self.assertEqual(got[0].kernel, 123)
        self.assertIsInstance(got[1], cl.ZeroFill)
        self.assertEqual(got[1].dst, metadata.data_ptr() + 12)
        self.assertEqual(got[1].nbytes, 4)
        self.assertIs(got[1].owner[1], metadata)
        self.assertEqual(got[2].kernel, 123)
        self.assertIs(got[0].owner, owner)

    @parametrize("offset,nbytes", [(-4, 4), (2, 4), (16, 4), (0, 8)])
    def test_invalid_scheduler_reset(self, offset, nbytes):
        metadata = torch.empty(4, dtype=torch.int32)
        sink = torch.tensor([1, 1, metadata.data_ptr() + offset, nbytes])
        with self.assertRaises(cl.Mismatch):
            launches._parse_sink(sink, None, metadata)

    def test_missing_scheduler_metadata(self):
        sink = torch.tensor([1, 1, 123, 4])
        with self.assertRaises(cl.Mismatch):
            launches._parse_sink(sink, None)


instantiate_parametrized_tests(TestVllmLaunches)

if __name__ == "__main__":
    run_tests()
