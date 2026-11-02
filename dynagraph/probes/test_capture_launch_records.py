from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from cuda.bindings import runtime as cr
from torch._inductor import dynagraph as dg
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)
from torch.utils import _capture_launch as cl


class TestCaptureLaunchRecords(TestCase):
    @parametrize("recorded", [False, True])
    @parametrize("conditional", [False, True])
    def test_recording_branch(self, recorded, conditional):
        runner = dg.DynaGraphRunner(None, "def call(args):\n    return ()\n", "cuda:0")
        launches = Mock(return_value=[cl.Launch(123, (1, 1, 1), (1, 1, 1), 0, ())])
        predicate = Mock(side_effect=lambda value, *, profile: profile)
        launches.records = predicate if conditional else recorded
        runner.inline_sites = {0: cl.Declaration(launches, None, None)}
        runner.site_out = {0: None}
        runner.ex = SimpleNamespace(
            inline_nodes={}, inline_cluster={}, child_nodes=[],
            site_body_nodes=[], site_cond=[], site_held=[],
        )
        node = cl.Recorded(123, (1, 1, 1), (1, 1, 1), 0, None, [])
        success = cr.cudaError_t.cudaSuccess
        fn = Mock(return_value="result")
        with (
            patch.object(torch.cuda, "current_stream", return_value=SimpleNamespace(cuda_stream=1)),
            patch.object(cr, "cudaStreamGetCaptureInfo", side_effect=[
                (success, None, None, None, []),
                (success, None, None, None, [1]),
            ]),
            patch.object(cr, "cudaGraphNodeGetDependencies", return_value=(success, None, None, 0)),
            patch.object(cl, "_chain_order", return_value=[1]),
            patch.object(cl, "read_node", return_value=node),
            patch.object(cl, "check", return_value=[123]) as check,
        ):
            result = runner._capture_inline(0, fn, ("input",), {"profile": recorded})
        self.assertEqual(result, "result")
        fn.assert_called_once_with("input", profile=recorded)
        self.assertEqual(check.call_count, 0 if recorded else 1)
        self.assertEqual(launches.call_count, 0 if recorded else 1)
        if conditional:
            predicate.assert_called_once_with("input", profile=recorded)
        self.assertEqual(runner.ex.inline_nodes[0], [1])


instantiate_parametrized_tests(TestCaptureLaunchRecords)

if __name__ == "__main__":
    run_tests()
