"""`${...}` resolution, including the inline form the output paths rely on."""

from __future__ import annotations

import unittest

from pipeline.templating import global_ref, resolve

SCOPE = {
    "params": {
        "output_root": "/data/output/run-1",
        "resolution": [720, 1280],
        "seed": 7,
        "flag": True,
        "prompt": "a person",
    }
}


class TestResolve(unittest.TestCase):
    def test_whole_string_reference_keeps_the_value_type(self):
        """`${params.resolution}` must stay a list; a workflow indexes it."""
        self.assertEqual(resolve("${params.resolution}", SCOPE), [720, 1280])
        self.assertIsInstance(resolve("${params.seed}", SCOPE), int)
        self.assertIs(resolve("${params.flag}", SCOPE), True)

    def test_indexing_into_a_list(self):
        self.assertEqual(resolve("${params.resolution.0}", SCOPE), 720)

    def test_inline_reference_produces_a_string(self):
        self.assertEqual(
            resolve("${params.output_root}/colmap", SCOPE), "/data/output/run-1/colmap"
        )

    def test_several_inline_references(self):
        self.assertEqual(
            resolve("seed-${params.seed}-w${params.resolution.0}", SCOPE), "seed-7-w720"
        )

    def test_plain_strings_pass_through(self):
        self.assertEqual(resolve("no templates here", SCOPE), "no templates here")
        self.assertEqual(resolve("costs $5 {maybe}", SCOPE), "costs $5 {maybe}")

    def test_nested_structures(self):
        resolved = resolve(
            {"a": ["${params.seed}", "${params.output_root}/x"], "b": {"c": "${params.flag}"}},
            SCOPE,
        )
        self.assertEqual(resolved, {"a": [7, "/data/output/run-1/x"], "b": {"c": True}})

    def test_unknown_param_raises(self):
        with self.assertRaises(KeyError):
            resolve("${params.nope}", SCOPE)
        with self.assertRaises(KeyError):
            resolve("prefix-${params.nope}", SCOPE)


class TestGlobalRef(unittest.TestCase):
    """`global_ref` is how the web UI spots a step param wired straight to a
    workflow global, so it can draw it read-only instead of giving one
    setting a second editable home."""

    def test_whole_value_globals_reference(self):
        self.assertEqual(global_ref("${globals.resolution}"), "resolution")
        self.assertEqual(global_ref("  ${globals.resolution}  "), "resolution")
        self.assertEqual(global_ref("${globals.resolution.0}"), "resolution.0")

    def test_inline_and_non_globals_forms_are_not_refs(self):
        self.assertIsNone(global_ref("${globals.output_root}/colmap"))
        self.assertIsNone(global_ref("${params.resolution}"))
        self.assertIsNone(global_ref("[720, 1280]"))
        self.assertIsNone(global_ref([720, 1280]))
        self.assertIsNone(global_ref(720))
        self.assertIsNone(global_ref(None))


if __name__ == "__main__":
    unittest.main()
