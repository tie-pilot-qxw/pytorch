from unittest.mock import Mock, patch

import torch
from torch._inductor import cudagraph_trees as trees, dynagraph as dg
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


@instantiate_parametrized_tests
class TestFallbackPolicy(TestCase):
    @torch._inductor.config.patch({
        "triton.dynagraph": True,
        "triton.dynagraph_rebuilds": 0,
    })
    @parametrize("failure", ["build", "retire", "skip", "rebuild"])
    def test_refusal_raises(self, failure):
        model = Mock()
        result = {"retire": None, "skip": dg.SKIP_SHAPE, "rebuild": dg.REBUILD}
        runner = False if failure == "build" else Mock(return_value=result[failure])
        with (
            patch.object(trees, "_maybe_build_dynagraph", return_value=runner),
            patch.object(trees, "cudagraphify") as capture,
        ):
            wrapped = trees.cudagraphify_impl(model, [4], ())
            with self.assertRaisesRegex(RuntimeError, "fallback to Inductor CUDA graph trees is disabled"):
                wrapped([4])
            capture.assert_not_called()
            model.assert_not_called()

    @torch._inductor.config.patch({"triton.dynagraph": True})
    def test_served_call(self):
        model = Mock()
        runner = Mock(return_value=[5])
        with (
            patch.object(trees, "_maybe_build_dynagraph", return_value=runner),
            patch.object(trees, "cudagraphify") as capture,
        ):
            wrapped = trees.cudagraphify_impl(model, [4], ())
            self.assertEqual(wrapped([4]), [5])
            capture.assert_not_called()
            model.assert_not_called()

    @torch._inductor.config.patch({
        "triton.dynagraph": True,
        "triton.dynagraph_rebuilds": 1,
    })
    @parametrize("rebuilt", [False, True])
    def test_rebuild(self, rebuilt):
        model = Mock()
        first = Mock(return_value=dg.REBUILD)
        second = Mock(return_value=[5]) if rebuilt else False
        with (
            patch.object(trees, "_maybe_build_dynagraph", side_effect=[first, second]) as build,
            patch.object(trees, "cudagraphify") as capture,
        ):
            wrapped = trees.cudagraphify_impl(model, [4], ())
            if rebuilt:
                self.assertEqual(wrapped([4]), [5])
            else:
                with self.assertRaisesRegex(RuntimeError, "fallback to Inductor CUDA graph trees is disabled"):
                    wrapped([4])
            self.assertEqual(build.call_count, 2)
            capture.assert_not_called()
            model.assert_not_called()

    @torch._inductor.config.patch({"triton.dynagraph": False})
    def test_disabled(self):
        model = Mock(return_value=[5])
        with (
            patch.object(trees, "_maybe_build_dynagraph") as build,
            patch.object(trees, "cudagraphify", return_value=(model, [5])) as capture,
        ):
            wrapped = trees.cudagraphify_impl(model, [4], ())
            self.assertEqual(wrapped([4]), [5])
            build.assert_not_called()
            capture.assert_called_once()


if __name__ == "__main__":
    run_tests()
