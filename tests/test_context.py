"""Context.set/get, including the auto-vivify fix for scratch namespaces.

Found by trying to actually run the since-removed fast_helical_native.yaml:
its first step writes to `scene.vertices`, but `scene` is never seeded in
the initial context (cli.py seeds only `{"dataset": dataset}`). Context.set
KeyError'd on that first write. test_workflows.py never caught this because
it only validates workflow structure statically, never actually runs one.
"""

from __future__ import annotations

import unittest

from pipeline.context import Context


class _WithExtras:
    def __init__(self):
        self.extras = {}


class TestContextSet(unittest.TestCase):
    def test_writes_a_fresh_top_level_namespace(self):
        ctx = Context({"dataset": _WithExtras()})
        ctx.set("scene.vertices", [1, 2, 3])
        self.assertEqual(ctx.get("scene.vertices"), [1, 2, 3])

    def test_writes_a_fresh_nested_namespace(self):
        ctx = Context({})
        ctx.set("scene.image_warp.camera", "cam")
        self.assertEqual(ctx.get("scene.image_warp.camera"), "cam")

    def test_second_write_to_same_namespace_does_not_clobber_siblings(self):
        ctx = Context({})
        ctx.set("scene.vertices", [1])
        ctx.set("scene.faces", [2])
        self.assertEqual(ctx.get("scene.vertices"), [1])
        self.assertEqual(ctx.get("scene.faces"), [2])

    def test_sets_a_dict_attribute_on_an_existing_object(self):
        ctx = Context({"dataset": _WithExtras()})
        ctx.set("dataset.extras.orbit_target", [0, 0, 0])
        self.assertEqual(ctx.get("dataset.extras.orbit_target"), [0, 0, 0])

    def test_overwrites_an_existing_value(self):
        ctx = Context({})
        ctx.set("scene.x", 1)
        ctx.set("scene.x", 2)
        self.assertEqual(ctx.get("scene.x"), 2)

    def test_top_level_set_replaces_the_whole_value(self):
        ctx = Context({})
        ctx.set("dataset", "first")
        ctx.set("dataset", "second")
        self.assertEqual(ctx.get("dataset"), "second")

    def test_get_missing_top_level_key_raises(self):
        ctx = Context({})
        with self.assertRaises(KeyError):
            ctx.get("nope")


if __name__ == "__main__":
    unittest.main()
