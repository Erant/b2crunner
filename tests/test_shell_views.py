"""inject_shell_views: which frames take the shell's render, and what the
VACE mask says about them (helical_shell.yaml, 2026-09-19).

Synthetic throughout, and it can be: the step is a batch swap by index
plus a mask built the way `render`'s `splat_inactive_mask` builds it. What
it cannot check is the thing that matters most — whether a shell render
ten degrees above the photograph is good enough to hold a denoise — and
that needs a pod.
"""

from __future__ import annotations

import unittest

import numpy as np

from tests.helpers import run_step

import pipeline.steps  # noqa: F401


HEIGHT, WIDTH = 8, 6


class _Camera:
    def __init__(self, position):
        self.position = np.asarray(position, dtype=np.float32)


def cameras(count: int, *, shift: float = 0.0):
    return [_Camera([float(i) + shift, 0.0, 2.0]) for i in range(count)]


def frames(count: int, value: int):
    """A batch whose every pixel carries the frame's index, offset by
    `value` — so a test can say which batch a returned frame came from."""
    return [np.full((HEIGHT, WIDTH, 3), value + i, dtype=np.uint8)
            for i in range(count)]


def shell_alpha(count: int):
    """Each shell render covers the left half of the frame fully, has a
    soft rim one column wide, and nothing on the right."""
    out = []
    for _ in range(count):
        alpha = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
        alpha[:, : WIDTH // 2] = 1.0
        alpha[:, WIDTH // 2] = 0.5
        out.append(alpha)
    return out


class ShellViewsCase(unittest.TestCase):
    N = 10

    def build(self, *, masks=None, anchor=0, shell_cameras=True, n=None):
        n = self.N if n is None else n
        self.drawings = frames(n, 0)
        self.shell = frames(n, 100)
        inputs = {
            "images": self.drawings,
            "shell_images": self.shell,
            "shell_masks": shell_alpha(n),
            "cameras": cameras(n),
            "anchor_frame_index": anchor,
        }
        if shell_cameras:
            inputs["shell_cameras"] = cameras(n)
        if masks is not None:
            inputs["masks"] = masks
        return inputs

    def sources(self, result):
        return ["shell" if int(img[0, 0, 0]) >= 100 else "drawing"
                for img in result["images"]]


class TestTheBands(ShellViewsCase):
    def test_the_default_takes_the_last_frame_only(self):
        result = run_step("inject_shell_views", self.build())
        self.assertEqual(self.sources(result),
                         ["drawing"] * (self.N - 1) + ["shell"])
        roles = [v["source"] for v in result["view_roles"]]
        self.assertEqual(roles[-1], "shell_tail")
        self.assertEqual(result["view_roles"][0]["role"], "anchor")

    def test_tail_and_head_frames_count_from_the_ends(self):
        result = run_step("inject_shell_views", self.build(),
                          {"tail_frames": 2, "head_frames": 1})
        self.assertEqual(
            self.sources(result),
            ["drawing", "shell"] + ["drawing"] * (self.N - 4) + ["shell", "shell"],
        )
        self.assertEqual(result["view_roles"][1]["source"], "shell_head")
        self.assertEqual(result["view_roles"][self.N - 2]["source"], "shell_tail")

    def test_the_anchor_frame_is_never_touched(self):
        """inject_anchor owns it, wherever it sits: a head band counts from
        the frame after it, and a tail band that reaches it skips it."""
        result = run_step("inject_shell_views", self.build(anchor=3),
                          {"tail_frames": 0, "head_frames": 2})
        self.assertEqual(self.sources(result)[3:6], ["drawing", "shell", "shell"])

        result = run_step("inject_shell_views", self.build(anchor=self.N - 1),
                          {"tail_frames": 3})
        sources = self.sources(result)
        self.assertEqual(sources[-1], "drawing")
        self.assertEqual(sources[-3:-1], ["shell", "shell"])

    def test_zero_frames_substitutes_nothing(self):
        result = run_step("inject_shell_views", self.build(),
                          {"tail_frames": 0, "head_frames": 0})
        self.assertEqual(self.sources(result), ["drawing"] * self.N)
        for mask in result["masks"]:
            self.assertTrue(np.all(mask == 1.0))

    def test_the_input_batch_is_not_mutated(self):
        inputs = self.build()
        before = [img.copy() for img in self.drawings]
        run_step("inject_shell_views", inputs, {"tail_frames": 2})
        for original, kept in zip(before, inputs["images"]):
            np.testing.assert_array_equal(original, kept)


class TestTheVaceMask(ShellViewsCase):
    def test_silhouette_keeps_the_covered_pixels_and_denoises_the_rest(self):
        """0.0 where the shell's alpha is at or above the threshold (the
        composited face's own rule, body2colmap's InactiveMaskOptions), 1.0
        over the soft rim, the holes and the grey around it."""
        result = run_step("inject_shell_views", self.build())
        mask = result["masks"][-1]
        self.assertEqual(mask.shape, (HEIGHT, WIDTH))
        self.assertEqual(mask.dtype, np.float32)
        self.assertTrue(np.all(mask[:, : WIDTH // 2] == 0.0))
        self.assertTrue(np.all(mask[:, WIDTH // 2:] == 1.0))
        # And the frames that kept their drawing are untouched.
        for mask in result["masks"][:-1]:
            self.assertTrue(np.all(mask == 1.0))

    def test_a_lower_threshold_keeps_the_rim(self):
        result = run_step("inject_shell_views", self.build(),
                          {"inactive_threshold": 0.5})
        mask = result["masks"][-1]
        self.assertTrue(np.all(mask[:, : WIDTH // 2 + 1] == 0.0))
        self.assertTrue(np.all(mask[:, WIDTH // 2 + 1:] == 1.0))

    def test_frame_keeps_the_whole_frame(self):
        result = run_step("inject_shell_views", self.build(),
                          {"reference": "frame"})
        self.assertTrue(np.all(result["masks"][-1] == 0.0))
        self.assertEqual(result["view_roles"][-1]["role"], "reference_frame")

    def test_none_substitutes_without_marking(self):
        result = run_step("inject_shell_views", self.build(),
                          {"reference": "none"})
        self.assertEqual(self.sources(result)[-1], "shell")
        self.assertTrue(np.all(result["masks"][-1] == 1.0))
        self.assertEqual(result["view_roles"][-1]["role"], "synthetic")

    def test_a_supplied_batch_is_written_into_not_over(self):
        """The photograph's 0.0 at the anchor (inject_anchor's) and the
        face composite's 0.0 elsewhere survive; only the substituted frames'
        masks change."""
        supplied = [np.ones((HEIGHT, WIDTH), dtype=np.float32) for _ in range(self.N)]
        supplied[0][:] = 0.0                # the injected photograph
        supplied[4][2:4, 2:4] = 0.0         # a composited face
        supplied[-1][:] = 0.0               # would be lost if manufactured over
        result = run_step("inject_shell_views", self.build(masks=supplied))
        self.assertTrue(np.all(result["masks"][0] == 0.0))
        self.assertEqual(float(result["masks"][4].sum()), HEIGHT * WIDTH - 4)
        self.assertTrue(np.all(result["masks"][-1][:, WIDTH // 2:] == 1.0))


class TestTheChecks(ShellViewsCase):
    def test_a_shell_batch_of_the_wrong_length_is_refused(self):
        inputs = self.build()
        inputs["shell_images"] = inputs["shell_images"][:-1]
        with self.assertRaises(ValueError):
            run_step("inject_shell_views", inputs)

    def test_a_shell_rendered_on_other_cameras_is_refused(self):
        inputs = self.build()
        inputs["shell_cameras"] = cameras(self.N, shift=0.01)
        with self.assertRaises(ValueError):
            run_step("inject_shell_views", inputs)

    def test_the_cameras_are_optional(self):
        result = run_step("inject_shell_views", self.build(shell_cameras=False))
        self.assertEqual(self.sources(result)[-1], "shell")

    def test_an_anchor_outside_the_batch_is_refused(self):
        with self.assertRaises(ValueError):
            run_step("inject_shell_views", self.build(anchor=self.N))

    def test_a_shell_frame_of_another_size_is_refused(self):
        inputs = self.build()
        inputs["shell_images"][-1] = np.zeros((HEIGHT + 1, WIDTH, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            run_step("inject_shell_views", inputs)


if __name__ == "__main__":
    unittest.main()
