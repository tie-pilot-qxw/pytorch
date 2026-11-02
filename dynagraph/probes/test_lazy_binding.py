import weakref
from unittest.mock import Mock, patch

import torch
from torch._inductor import dynagraph as dg
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.utils import _capture_launch as cl

from test_inline_cache import CountingKey


class TestLazyBinding(TestCase):
    def test_first_use_and_reuse(self):
        first, second = torch.empty(8), torch.empty(8)
        launch = cl.Launch(1, (1, 1, 1), (1, 1, 1), 0,
                           ((first.data_ptr() + 4).to_bytes(8, "little"),))
        with patch.object(cl, "_bind_plan", wraps=cl._bind_plan) as bind:
            template = cl._LaunchTemplate([launch], [first])
            bind.assert_not_called()
            for _ in range(2):
                rebound = template.rebind([second])
                self.assertEqual(
                    int.from_bytes(rebound[0].args[0], "little"), second.data_ptr() + 4
                )
            bind.assert_called_once()

    def test_source_snapshot(self):
        first, replacement, second = torch.empty(8), torch.empty(8), torch.empty(8)
        template = cl._LaunchTemplate([cl.ZeroFill(first.data_ptr() + 4, 8)], [first])
        first.set_(replacement)
        self.assertEqual(template.rebind([second]), [cl.ZeroFill(second.data_ptr() + 4, 8)])

    def test_rejected_binding_is_not_retried(self):
        base = torch.empty(8)
        view = base[2:6]
        with patch.object(cl, "_bind_plan", wraps=cl._bind_plan) as bind:
            template = cl._LaunchTemplate([cl.ZeroFill(base.data_ptr(), 4)], [view])
            self.assertIsNone(template.rebind([torch.empty_like(view)]))
            self.assertIsNone(template.rebind([torch.empty_like(view)]))
            bind.assert_called_once()

    def test_empty_template(self):
        template = cl._LaunchTemplate([], [])
        self.assertEqual(template.rebind([]), [])
        self.assertEqual(template.plan, [])

    def test_launch_owner(self):
        first, second, owner = torch.empty(8), torch.empty(8), object()
        template = cl._LaunchTemplate([cl.ZeroFill(first.data_ptr(), 4, owner)], [first])
        self.assertIs(template.rebind([second])[0].owner, owner)

    def test_packed_launch_cache(self):
        runner = dg.DynaGraphRunner(None, "def call(args):\n    return ()\n", "cuda:0")
        runner.extern_sites = ["ops:lazy_test.launch.default"]
        first, second = torch.empty(8), torch.empty(8)
        describe = Mock(side_effect=lambda x: [cl.ZeroFill(x.data_ptr(), 4)])
        decl = cl.Declaration(describe, None, None, share=True)
        with patch.object(cl, "_bind_plan", wraps=cl._bind_plan) as bind:
            self.assertEqual(runner._packed_launches(0, decl, (first,), {}, 5),
                             [cl.ZeroFill(first.data_ptr(), 4)])
            bind.assert_not_called()
            self.assertEqual(runner._packed_launches(0, decl, (second,), {}, 5),
                             [cl.ZeroFill(second.data_ptr(), 4)])
            self.assertEqual(describe.call_count, 1)
            bind.assert_called_once()
            runner._packed_launches(0, decl, (second,), {}, 6)
            self.assertEqual(describe.call_count, 2)
            bind.assert_called_once()
            self.assertEqual({key[1] for key in runner._inline_templates}, {5, 6})

    def test_recorded_template_is_lazy(self):
        first, second, owner = torch.empty(8), torch.empty(8), object()
        describe = cl.recorded(lambda x: None, template_key=lambda x: 0)
        with (
            patch.object(cl, "_record_graph", return_value=(
                [cl.ZeroFill(first.data_ptr(), 4)], owner,
            )) as record,
            patch.object(cl, "_bind_plan", wraps=cl._bind_plan) as bind,
        ):
            describe(first)
            bind.assert_not_called()
            self.assertEqual(describe(second), [cl.ZeroFill(second.data_ptr(), 4, owner)])
            record.assert_called_once()
            bind.assert_called_once()

    def test_snapshot_does_not_retain_source(self):
        source = torch.empty(8)
        reference = weakref.ref(source)
        template = cl._LaunchTemplate([cl.ZeroFill(source.data_ptr(), 4)], [source])
        del source
        self.assertIsNone(reference())
        self.assertIsNotNone(template.rebind([torch.empty(8)]))

    def test_rejected_template_uses_fresh_description(self):
        runner = dg.DynaGraphRunner(None, "def call(args):\n    return ()\n", "cuda:0")
        runner.extern_sites = ["ops:lazy_test.launch.default"]
        describe = Mock(side_effect=lambda x: [cl.ZeroFill(x.untyped_storage().data_ptr(), 4)])
        decl = cl.Declaration(describe, None, None, share=True)
        first, second = torch.empty(8), torch.empty(8)
        with patch.object(cl, "_bind_plan", wraps=cl._bind_plan) as bind:
            runner._packed_launches(0, decl, (first[2:6],), {}, 5)
            for _ in range(2):
                result = runner._packed_launches(0, decl, (second[2:6],), {}, 5)
                self.assertEqual(result, [cl.ZeroFill(second.data_ptr(), 4)])
            bind.assert_called_once()
            self.assertEqual(describe.call_count, 3)

    def test_sharing_key_uses_region_id(self):
        runner = dg.DynaGraphRunner(None, "def call(args):\n    return ()\n", "cuda:0")
        state = [0]
        x = torch.empty(8)
        decl = cl.Declaration(lambda x: [cl.ZeroFill(x.data_ptr(), 4)],
                              lambda x: state[0], None,
                              template=lambda x: state[0], share=True)
        runner.extern_sites = ["ops:lazy_test.launch.default"]
        runner.inline_sites = {0: decl}
        runner._inline_no_keys = False
        hkey = CountingKey(((("tokens", 3),), 0, tuple(range(141)), ()))
        with (
            patch.object(runner, "_shape_inputs", return_value=[]),
            patch.object(runner, "_site_operands", return_value=((x,), {})),
            patch.object(runner, "_ensure_prepared"),
            patch.object(runner, "_node_params", side_effect=lambda launches: launches),
            patch.object(runner, "_apply_wanted", side_effect=lambda *args: args[2]),
        ):
            runner._apply_inline(None, {}, hkey, [], [])
            hkey.hash_calls = 0
            state[0] = 1
            runner._apply_inline(None, {}, hkey, [], [])
            self.assertEqual(hkey.hash_calls, 1)
            self.assertEqual(len(runner._inline_templates), 2)


if __name__ == "__main__":
    run_tests()
