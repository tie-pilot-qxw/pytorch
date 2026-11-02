from types import SimpleNamespace

import torch
from cuda.bindings import runtime as cr
from torch._inductor import dynagraph as dg
from torch.cuda._utils import _check_cuda_bindings as ck
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.utils import _capture_launch as cl


class TestZeroFillBindings(TestCase):
    def test_rebind(self):
        first, second = torch.empty(4), torch.empty(4)
        template = [cl.ZeroFill(first.data_ptr() + 4, 8)]
        plan = cl._bind_plan(template, [first])
        got = cl._rebind(template, plan, [second])
        self.assertEqual(got, [cl.ZeroFill(second.data_ptr() + 4, 8)])

    def test_check(self):
        first, second = cl.ZeroFill(123, 4), cl.ZeroFill(456, 4)
        self.assertEqual(cl.check([first], [second], exact=False), [0])
        with self.assertRaises(cl.Mismatch):
            cl.check([first], [second])
        with self.assertRaises(cl.Mismatch):
            cl.check([first], [cl.ZeroFill(123, 8)], exact=False)


class TestZeroFill(TestCase):
    def test_recorded_rebind(self, device):
        first = torch.empty(4, dtype=torch.int32, device=device)
        second = torch.empty_like(first)
        describe = cl.recorded(lambda output: output.zero_(), template_key=lambda output: None)
        a = describe(first)
        b = describe(second)
        self.assertEqual(len(a), 1)
        self.assertIsInstance(a[0], cl.ZeroFill)
        self.assertEqual(a[0].dst, first.data_ptr())
        self.assertEqual(b[0].dst, second.data_ptr())
        self.assertIs(a[0].owner, b[0].owner)

    def test_inline_update(self, device):
        first = torch.full((4,), 7, dtype=torch.int32, device=device)
        second = torch.full((8,), 9, dtype=torch.int32, device=device)
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph):
            first.zero_()
        graph.instantiate()
        nodes = ck(cr.cudaGraphGetNodes(graph.raw_cuda_graph(), 1))[0]
        node = nodes[0]
        recorded = cl.read_zero_fill(node)
        self.assertEqual(recorded, cl.ZeroFill(first.data_ptr(), first.nbytes))
        runner = dg.DynaGraphRunner(None, "def call(args):\n    return ()\n", device)
        runner.host_mode = False
        runner._inline_no_keys = False
        ex = SimpleNamespace(
            exec_h=graph.raw_cuda_graph_exec(), inline_applied={},
            inline_nodes={0: [node]}, inline_cluster={0: [(0, 0, 0)]},
            inline_off=set(), inline_state=None,
        )
        prepared = runner._node_params([cl.ZeroFill(second.data_ptr(), second.nbytes)])
        self.assertIs(runner._apply_wanted(ex, None, [(0, 1, prepared)], []), ex)
        graph.replay()
        torch.cuda.synchronize()
        self.assertEqual(first, torch.full_like(first, 7))
        self.assertEqual(second, torch.zeros_like(second))


instantiate_device_type_tests(TestZeroFill, globals(), only_for=("cuda",))

if __name__ == "__main__":
    run_tests()
