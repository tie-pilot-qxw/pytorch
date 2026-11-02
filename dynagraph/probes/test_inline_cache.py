from unittest.mock import patch

from torch._inductor import dynagraph as dg
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)
from torch.utils._capture_launch import Declaration


class CountingKey(tuple):
    hash_calls: int

    def __new__(cls, value):
        return super().__new__(cls, value)

    def __init__(self, value):
        self.hash_calls = 0

    def __hash__(self):
        self.hash_calls += 1
        return super().__hash__()


class TestInlineCache(TestCase):
    def make_runner(self, sites=1):
        src = "def call(args):\n    arg0, = args\n    args.clear()\n    return (arg0,)\n"
        runner = dg.DynaGraphRunner(None, src, "cuda:0")
        operation_key = [0]
        decl = Declaration(lambda: [], key=lambda: operation_key[0], prepare=None)
        runner.inline_sites = dict.fromkeys(range(sites), decl)
        runner._inline_no_keys = False
        self.enterContext(patch.object(runner, "_shape_inputs", return_value=[]))
        self.enterContext(patch.object(runner, "_site_operands", return_value=((), {})))
        prepare = self.enterContext(
            patch.object(runner, "_prepare_inline", side_effect=lambda *a: object())
        )
        self.enterContext(
            patch.object(runner, "_apply_wanted", side_effect=lambda *args: args[2])
        )
        return runner, operation_key, prepare

    @parametrize("change", ["shape", "lane", "address", "dependency", "operation_key"])
    def test_configuration_changes(self, change):
        runner, operation_key, prepare = self.make_runner()
        hkey = ((("tokens", 3),), 0, (4096,), ())
        first = runner._apply_inline(None, {}, hkey, [], [])
        equal_key = ((("tokens", int("3")),), 0, (int("4096"),), ())
        self.assertEqual(runner._apply_inline(None, {}, equal_key, [], []), first)
        self.assertEqual(prepare.call_count, 1)
        changed = list(hkey)
        if change == "shape":
            changed[0] = (("tokens", 8),)
        elif change == "lane":
            changed[1] = 1
        elif change == "address":
            changed[2] = (8192,)
        elif change == "dependency":
            changed[3] = ("new communicator",)
        else:
            operation_key[0] = 1
        second = runner._apply_inline(None, {}, tuple(changed), [], [])
        self.assertNotEqual(first[0][1], second[0][1])
        self.assertEqual(prepare.call_count, 2)
        operation_key[0] = 0
        self.assertEqual(runner._apply_inline(None, {}, hkey, [], []), first)
        self.assertEqual(prepare.call_count, 2)

    def test_ids_survive_operand_cache_clear(self):
        runner, _, prepare = self.make_runner()
        hkey = ((("tokens", 3),), 0, (4096,), ())
        first = runner._apply_inline(None, {}, hkey, [], [])
        runner._inline_ops.clear()
        second = runner._apply_inline(None, {}, hkey, [], [])
        self.assertNotEqual(first[0][1], second[0][1])
        self.assertEqual(prepare.call_count, 2)

    def test_unkeyed_sites(self):
        runner, _, prepare = self.make_runner()
        runner.inline_sites = {0: Declaration(lambda: [], key=None, prepare=None)}
        runner._inline_no_keys = True
        self.enterContext(patch.object(runner, "_site_signature", return_value=None))
        hkey = ((("tokens", 3),), 0, (4096,), ())
        first = runner._apply_inline(None, {}, hkey, [], [])
        self.assertEqual(runner._apply_inline(None, {}, hkey, [], []), first)
        self.assertEqual(prepare.call_count, 1)

    @parametrize("sites", [1, 168])
    def test_region_key_hashed_once(self, sites):
        runner, _, prepare = self.make_runner(sites)
        hkey = CountingKey(((("tokens", 3),), 0, tuple(range(141)), ()))
        runner._apply_inline(None, {}, hkey, [], [])
        hkey.hash_calls = 0
        wanted = runner._apply_inline(None, {}, hkey, [], [])
        self.assertEqual(len(wanted), sites)
        self.assertEqual(hkey.hash_calls, 1)
        self.assertEqual(prepare.call_count, sites)


instantiate_parametrized_tests(TestInlineCache)

if __name__ == "__main__":
    run_tests()
