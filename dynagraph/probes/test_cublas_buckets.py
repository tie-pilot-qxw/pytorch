#!/usr/bin/env python3
"""Correctness and capture-count regressions for per-GEMM M buckets.

Run with this fork's built torch, e.g. CUDA_VISIBLE_DEVICES=3 python
dynagraph/probes/test_cublas_buckets.py. No timing claims are made here.
"""

from unittest.mock import patch

import torch
from torch._inductor import config, dynagraph as dg
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase


class TestCublasBuckets(TestCase):
    def setUp(self):
        super().setUp()
        torch._dynamo.reset()
        torch.manual_seed(17)
        self.settings = config.patch(
            {
                "triton.dynagraph": True,
                "triton.dynagraph_extern_child": True,
                "triton.dynagraph_cublas_bucket_size": 128,
                "triton.dynagraph_clone_sites": False,
                "triton.dynagraph_update": "host",
                "compile_threads": 1,
                "force_disable_caches": True,
            }
        )
        self.settings.__enter__()
        self.addCleanup(self.settings.__exit__, None, None, None)
        self.addCleanup(torch._dynamo.reset)

    def run_shapes(
        self,
        *,
        device,
        dtype=torch.bfloat16,
        bias=None,
        transpose=False,
        noncontiguous=False,
        alpha=0.5,
        beta=-0.75,
        width=128,
        update="host",
        shapes=(127, 128, 129, 170, 256, 257, 130, 127),
        expect_bucket=True,
        expect_reuse=True,
    ):
        runners, refusals, served = [], [], []
        original_init = dg.DynaGraphRunner.__init__
        original_call = dg.DynaGraphRunner.__call__
        original_fallback = dg._fallback

        def init(runner, *a, **kw):
            original_init(runner, *a, **kw)
            runners.append(runner)

        def call(runner, inputs):
            out = original_call(runner, inputs)
            if out is not None and out is not dg.SKIP_SHAPE:
                served.append(runner)
            return out

        def fallback(*a, **kw):
            refusals.append(a)
            return original_fallback(*a, **kw)

        def fn(x, w, b):
            # An intermediate input checks that the lowering accepts computed
            # activations, and the epilogue checks the actual output extent.
            x = x + 0.25 if not noncontiguous else x
            y = x @ w if bias is None else torch.addmm(b, x, w, alpha=alpha, beta=beta)
            return y.relu()

        with (
            config.patch(
                {
                    "triton.dynagraph_cublas_bucket_size": width,
                    "triton.dynagraph_update": update,
                }
            ),
            patch.object(dg.DynaGraphRunner, "__init__", init),
            patch.object(dg.DynaGraphRunner, "__call__", call),
            patch.object(dg, "_fallback", fallback),
            torch.no_grad(),
        ):
            compiled = torch.compile(
                fn, dynamic=True, mode="reduce-overhead", fullgraph=True
            )
            for m in shapes:
                torch.compiler.cudagraph_mark_step_begin()
                x = torch.randn(
                    m, 128 if noncontiguous else 64, device=device, dtype=dtype
                )
                if noncontiguous:
                    x = x[:, ::2]
                w = torch.randn(
                    (96, 64) if transpose else (64, 96), device=device, dtype=dtype
                )
                if transpose:
                    w = w.t()
                bias_shape = {
                    None: (96,),
                    "scalar": (),
                    "vector": (96,),
                    "row": (1, 96),
                    "matrix": (m, 96),
                    "column": (m, 1),
                }[bias]
                b = torch.randn(bias_shape, device=device, dtype=dtype)
                if beta == 0:
                    b.fill_(float("nan"))
                torch._dynamo.mark_dynamic(x, 0)
                torch._dynamo.mark_static(x, 1)
                torch._dynamo.mark_static(w)
                if bias in ("matrix", "column"):
                    torch._dynamo.mark_dynamic(b, 0)
                    torch._dynamo.mark_static(b, 1)
                else:
                    torch._dynamo.mark_static(b)
                expected = fn(x, w, b)
                actual = compiled(x, w, b)
                self.assertEqual(actual, expected, atol=0.04, rtol=0.03)
                self.assertEqual(tuple(actual.shape), (m, 96))

        self.assertFalse(refusals, refusals)
        self.assertEqual(len(runners), 1, "unexpected compilation or runner rebuild")
        self.assertGreaterEqual(
            len(served), len(shapes), "ordinary execution hid a DG fallback"
        )
        runner = runners[0]
        sites = [s for s in runner.extern_sites if s.startswith("dynagraph_bucket_")]
        self.assertEqual(len(sites), int(expect_bucket))
        if expect_bucket:
            expected_captures = len({(m + width - 1) // width for m in shapes})
            stats = runner.cublas_bucket_stats
            if expect_reuse:
                self.assertEqual(stats["captures"], expected_captures)
                self.assertGreater(stats["clones"], 0)
                self.assertEqual(stats["fallbacks"], 0)
            else:
                self.assertGreater(stats["captures"], expected_captures)
                self.assertEqual(stats["clones"], 0)
                self.assertGreater(stats["fallbacks"], 0)
        return runner

    def test_mm_bf16(self, device):
        self.run_shapes(device=device)

    def test_mm_fp16_transposed_weight(self, device):
        self.run_shapes(device=device, dtype=torch.float16, transpose=True)

    @parametrize("bias", ["scalar", "vector", "row", "matrix", "column"])
    def test_addmm_bias(self, device, bias):
        self.run_shapes(device=device, bias=bias)

    def test_addmm_zero_beta(self, device):
        self.run_shapes(device=device, bias="matrix", beta=0)

    def test_addmm_zero_alpha(self, device):
        self.run_shapes(device=device, bias="vector", alpha=0)

    def test_custom_width(self, device):
        self.run_shapes(device=device, width=64, shapes=(65, 100, 128, 129, 160, 66))

    def test_disabled(self, device):
        self.run_shapes(device=device, width=0, expect_bucket=False)

    def test_unsupported_dtype(self, device):
        self.run_shapes(device=device, dtype=torch.float32, expect_bucket=False)

    def test_noncontiguous_input(self, device):
        # Inductor normalizes this view to contiguous storage before mm
        # lowering; bucketing must preserve the original strided values.
        self.run_shapes(device=device, noncontiguous=True)

    def test_device_update(self, device):
        self.run_shapes(device=device, update="device")

    def test_device_matrix_bias(self, device):
        self.run_shapes(device=device, bias="matrix", update="device")

    def test_existing_clone_sites(self, device):
        with config.patch({"triton.dynagraph_clone_sites": True}):
            self.run_shapes(device=device, width=0, expect_bucket=False)

    def test_uncloneable_capture(self, device):
        # An unknown node/pointer must cost another capture, never cause the
        # bucket to replay a child with stale operands.
        with patch.object(dg, "_clone_plan", return_value=None):
            self.run_shapes(device=device, shapes=(129, 170, 256, 130), expect_reuse=False)

    def test_driver_refuses_clone(self, device):
        with patch.object(dg, "_clone_child", side_effect=RuntimeError("refused")):
            self.run_shapes(device=device, shapes=(129, 170, 256, 130), expect_reuse=False)

    def test_backward(self, device):
        runners = []
        original_init = dg.DynaGraphRunner.__init__

        def init(runner, *a, **kw):
            original_init(runner, *a, **kw)
            runners.append(runner)

        def loss(x, w, b):
            return torch.addmm(b, x, w).float().square().mean()

        with (
            patch.object(dg.DynaGraphRunner, "__init__", init),
            patch.object(dg, "_fallback", wraps=dg._fallback) as fallback,
        ):
            compiled = torch.compile(
                loss, dynamic=True, mode="reduce-overhead", fullgraph=True
            )
            for m in (129, 170, 256, 257, 130):
                torch.compiler.cudagraph_mark_step_begin()
                x = torch.randn(
                    m, 64, device=device, dtype=torch.float16, requires_grad=True
                )
                w = torch.randn(
                    64, 96, device=device, dtype=torch.float16, requires_grad=True
                )
                b = torch.randn(
                    96, device=device, dtype=torch.float16, requires_grad=True
                )
                torch._dynamo.mark_dynamic(x, 0)
                torch._dynamo.mark_static(x, 1)
                torch._dynamo.mark_static(w)
                torch._dynamo.mark_static(b)
                reference = loss(x, w, b)
                expected = torch.autograd.grad(reference, (x, w, b))
                actual_loss = compiled(x, w, b)
                actual = torch.autograd.grad(actual_loss, (x, w, b))
                self.assertEqual(actual_loss, reference, atol=0.01, rtol=0.01)
                for got, ref in zip(actual, expected):
                    self.assertEqual(got, ref, atol=0.002, rtol=0.02)
        bucketed = [
            r
            for r in runners
            if any(s.startswith("dynagraph_bucket_") for s in r.extern_sites)
        ]
        self.assertTrue(bucketed, "training did not lower any bucketed GEMM")
        self.assertTrue(
            all(r.execs for r in bucketed), "training graph was not captured"
        )
        self.assertFalse(fallback.call_args_list)
        self.assertTrue(all(r.cublas_bucket_stats["clones"] for r in bucketed))


instantiate_device_type_tests(TestCublasBuckets, globals(), only_for=("cuda",))


if __name__ == "__main__":
    run_tests()
