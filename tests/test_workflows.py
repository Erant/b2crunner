"""Static validation of every workflow YAML in pipeline/workflows/.

None of these can be executed without a GPU, but most of the ways a
workflow file goes wrong are visible without running it: a step name that
isn't registered, a `${globals.x}` that doesn't resolve, an `env:` with no
entry in envs.yaml, a dispatch mode that doesn't exist. This catches those
so a pod run fails on the model, not on a typo.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import yaml

from pipeline.registry import STEP_REGISTRY
from pipeline.templating import resolve
from pipeline.workflow import WorkflowSpec

import pipeline.steps  # noqa: F401

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOW_DIR = REPO_ROOT / "pipeline" / "workflows"
ENVS_FILE = REPO_ROOT / "pipeline" / "envs" / "envs.yaml"
DOCKER_ENVS_FILE = REPO_ROOT / "docker" / "envs.docker.yaml"

VALID_DISPATCH = {"in_process", "subprocess", "service", "docker"}

# The denoise prompts, character for character, as
# ComfyUI-Body2COLMAP's workflows/api/denoise.json carries them (the
# positive template is node 188's string, before StringReplace fills in
# $SUBJECT_DESC$; the negative is node 7's text). Pinned here because the
# failure mode is silent: YAML's folded `>-` scalar turns each line break
# into a space, which inside the Chinese runs produces a prompt that still
# loads, still runs, and is not the one the graph sends.
DENOISE_PROMPT = (
    "时间静止，人物完全静止。身体如雕塑般僵硬，胸口没有起伏，头部"
    "保持固定角度，面部肌肉完全不动，维持单一的中性表情。双眼一眨"
    "不眨，眼睑保持张开且稳定，目光空洞而固定，锁定远处墙面上的一"
    "个点，对镜头毫无察觉、毫无反应。镜头以匀速、固定焦距围绕主体"
    "平滑移动。 Time is frozen and only the camera moves. The gaze "
    "stays anchored to that point on the wall as the camera passes, "
    "so the eyes slide across the frame, always aimed past the lens "
    "at the wall behind it. Eyelids stay open and steady, the expression "
    "holds without a single micro-movement, and the head keeps its "
    "exact angle throughout. The frozen subject is $SUBJECT_DESC$, "
    "lit by even soft studio light against a plain seamless backdrop, "
    "photorealistic, sharp focus, identical face and clothing in "
    "every frame."
)

DENOISE_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，"
    "画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的"
    "，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的"
    "，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背"
    "景，三条腿，背景人很多，倒着走, chewing, tattoos, 眼球运动，"
    "眨眼，眼动追踪，微表情，面部抽动，头部移动, 小头, 小脸"
)


def _workflows():
    return sorted(WORKFLOW_DIR.glob("*.yaml"))


class TestWorkflowFiles(unittest.TestCase):
    def test_there_are_workflows_to_check(self):
        self.assertTrue(_workflows())

    def test_each_workflow_loads(self):
        for path in _workflows():
            with self.subTest(workflow=path.name):
                spec = WorkflowSpec.from_yaml(str(path))
                self.assertTrue(spec.name)
                self.assertTrue(spec.steps)

    def test_step_names_are_registered(self):
        for path in _workflows():
            spec = WorkflowSpec.from_yaml(str(path))
            for step in spec.steps:
                with self.subTest(workflow=path.name, step=step.id):
                    self.assertIn(
                        step.step, STEP_REGISTRY,
                        f"{path.name}: step '{step.id}' references unregistered "
                        f"step '{step.step}'",
                    )

    def test_step_ids_are_unique(self):
        for path in _workflows():
            spec = WorkflowSpec.from_yaml(str(path))
            ids = [s.id for s in spec.steps]
            with self.subTest(workflow=path.name):
                self.assertEqual(len(ids), len(set(ids)))

    def test_dispatch_modes_are_valid(self):
        for path in _workflows():
            spec = WorkflowSpec.from_yaml(str(path))
            for step in spec.steps:
                with self.subTest(workflow=path.name, step=step.id):
                    self.assertIn(step.dispatch, VALID_DISPATCH)

    def test_envs_referenced_exist(self):
        envs = yaml.safe_load(ENVS_FILE.read_text())["envs"]
        for path in _workflows():
            spec = WorkflowSpec.from_yaml(str(path))
            for step in spec.steps:
                if step.dispatch == "in_process" or not step.env:
                    continue
                with self.subTest(workflow=path.name, step=step.id):
                    self.assertIn(
                        step.env, envs,
                        f"{path.name}: step '{step.id}' uses env '{step.env}' "
                        f"which envs.yaml does not define",
                    )

    def test_non_in_process_steps_name_an_env(self):
        for path in _workflows():
            spec = WorkflowSpec.from_yaml(str(path))
            for step in spec.steps:
                if step.dispatch in ("subprocess", "docker", "service"):
                    with self.subTest(workflow=path.name, step=step.id):
                        self.assertTrue(
                            step.env,
                            f"{path.name}: '{step.id}' dispatches "
                            f"{step.dispatch} but names no env",
                        )

    def test_docker_env_registry_matches_the_repo_one(self):
        """docker/envs.docker.yaml is copied over pipeline/envs/envs.yaml
        inside the image, so it has to define the same env names.

        These two drifted before: the Dockerfile built /workspace/venv_*
        while envs.yaml named pipeline/envs/*/venv/bin/python, and every
        subprocess step in the image would have failed to find an
        interpreter. Names here, not paths — the paths differ on purpose
        (bare pod vs container).
        """
        repo_envs = set(yaml.safe_load(ENVS_FILE.read_text())["envs"])
        docker_envs = set(yaml.safe_load(DOCKER_ENVS_FILE.read_text())["envs"])
        self.assertEqual(
            repo_envs, docker_envs,
            "pipeline/envs/envs.yaml and docker/envs.docker.yaml define "
            "different env names",
        )

    def test_docker_env_paths_are_absolute(self):
        """A relative python_bin resolves against the process working
        directory, which is what made the container paths wrong before."""
        envs = yaml.safe_load(DOCKER_ENVS_FILE.read_text())["envs"]
        for name, cfg in envs.items():
            with self.subTest(env=name):
                self.assertTrue(
                    cfg["python_bin"].startswith("/"),
                    f"{name}: python_bin must be absolute inside the image",
                )

    def test_all_param_templates_resolve(self):
        """A ${globals.x} naming a global that doesn't exist is the single
        easiest mistake to make in these files, and it would otherwise only
        surface once that step is reached on a GPU."""
        for path in _workflows():
            spec = WorkflowSpec.from_yaml(str(path))
            for step in spec.steps:
                with self.subTest(workflow=path.name, step=step.id):
                    try:
                        resolve(step.params, {"globals": spec.globals})
                    except Exception as exc:  # noqa: BLE001
                        self.fail(
                            f"{path.name}: step '{step.id}' has an unresolvable "
                            f"template: {exc}"
                        )

    def test_every_step_override_is_a_param_that_step_declares(self):
        """The other half of the templating check, and the one the split
        into namespaces made possible at all.

        A step's `params:` block is overrides on what its class declares
        (pipeline/step.py's `Step.PARAMS`), so a name that isn't declared is
        a value nothing will ever read — the pre-namespacing shape had no
        way to notice that, because a workflow-level param not consumed by
        anything is perfectly legal.
        """
        for path in _workflows():
            spec = WorkflowSpec.from_yaml(str(path))
            with self.subTest(workflow=path.name):
                spec.validate()

    def test_a_top_level_params_block_is_refused_by_name(self):
        """The pre-namespacing shape must not load as an empty workflow.

        Silently ignoring `params:` would run the whole pipeline at the step
        defaults and look like it had worked, which is the worst available
        outcome for a file somebody hasn't migrated yet.
        """
        import tempfile

        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
            handle.write("name: old\nparams:\n  seed: 0\nsteps: []\n")
            stale = handle.name
        with self.assertRaises(ValueError) as ctx:
            WorkflowSpec.from_yaml(stale)
        self.assertIn("globals:", str(ctx.exception))

    def test_when_conditions_resolve(self):
        """Same reasoning as the template check above, and more load-bearing:
        an unresolvable `when:` guards a step, and the whole point of
        resolving them up front in the runner is that this fails at the
        start of a run rather than an hour into one."""
        from pipeline.workflow import step_enabled

        for path in _workflows():
            spec = WorkflowSpec.from_yaml(str(path))
            for step in spec.steps:
                with self.subTest(workflow=path.name, step=step.id):
                    try:
                        step_enabled(step, spec.globals)
                    except Exception as exc:  # noqa: BLE001
                        self.fail(f"{path.name}: step '{step.id}': {exc}")

    def test_the_two_fast_helical_files_differ_only_by_the_upscale(self):
        """They are maintained as copies — there is no include mechanism —
        so the thing worth checking is that they have not drifted into two
        different pipelines that happen to share a name."""
        full = WorkflowSpec.from_yaml(str(WORKFLOW_DIR / "fast_helical_full.yaml"))
        short = WorkflowSpec.from_yaml(str(WORKFLOW_DIR / "fast_helical.yaml"))

        upscale_only = {"upscale", "rescale_cameras"}
        self.assertEqual(
            [s.id for s in full.steps if s.id not in upscale_only],
            [s.id for s in short.steps],
        )
        # Identical now, not merely close: everything the upscale had to be
        # told lives under the `upscale` step in the full file, so removing
        # that step removed its settings with it. Before the params were
        # namespaced this was `{"upscale_resolution", "upscale_batch_size"}`
        # — two workflow-level names that existed only because there was
        # nowhere else to put them.
        self.assertEqual(
            full.globals, short.globals,
            "the two files' globals have drifted apart",
        )
        for step in short.steps:
            twin = next(s for s in full.steps if s.id == step.id)
            with self.subTest(step=step.id):
                self.assertEqual(step.step, twin.step)
                self.assertEqual(step.dispatch, twin.dispatch)
                self.assertEqual(step.env, twin.env)
                self.assertEqual(step.inputs, twin.inputs)
                self.assertEqual(step.outputs, twin.outputs)
                self.assertEqual(step.when, twin.when)
                self.assertEqual(step.keep_loaded, twin.keep_loaded)

    def test_native_mirrors_full_from_denoise_pass1_on(self):
        """fast_helical_native is a bootstrap prologue (split_sheet ->
        reconstruct_body -> render_initial_views -> warp_reference_to_anchor
        -> reinject_anchor_initial) followed by a verbatim copy of
        fast_helical_full.yaml's steps. There is no include mechanism, so the
        thing worth checking is that the copy has not drifted.

        Everything but `output_root` (deliberately native-specific) is
        compared: step id, class, dispatch, env, inputs, outputs, when,
        keep_loaded, and the raw params block.
        """
        full = WorkflowSpec.from_yaml(str(WORKFLOW_DIR / "fast_helical_full.yaml"))
        native = WorkflowSpec.from_yaml(str(WORKFLOW_DIR / "fast_helical_native.yaml"))

        bootstrap = [
            "split_sheet", "reconstruct_body", "detect_face",
            "render_initial_views", "warp_reference_to_anchor",
            "reinject_anchor_initial",
        ]
        native_ids = [s.id for s in native.steps]
        self.assertEqual(
            native_ids[: len(bootstrap)], bootstrap,
            "fast_helical_native's bootstrap prologue has changed shape",
        )
        tail = native.steps[len(bootstrap):]
        self.assertEqual(
            [s.id for s in tail], [s.id for s in full.steps],
            "fast_helical_native's tail has drifted from fast_helical_full's steps",
        )
        for step in tail:
            twin = next(s for s in full.steps if s.id == step.id)
            with self.subTest(step=step.id):
                self.assertEqual(step.step, twin.step)
                self.assertEqual(step.dispatch, twin.dispatch)
                self.assertEqual(step.env, twin.env)
                self.assertEqual(step.inputs, twin.inputs)
                self.assertEqual(step.outputs, twin.outputs)
                self.assertEqual(step.when, twin.when)
                self.assertEqual(step.keep_loaded, twin.keep_loaded)
                if step.id == "rerender_splat":
                    # native threads a `framing` global through both its mesh
                    # render and this step; the other two files have no mesh
                    # render, so their rerender_splat frames at the default.
                    self.assertEqual(
                        step.params.get("framing"), "${globals.framing}",
                        "native's rerender_splat should read the framing global",
                    )
                    self.assertNotIn("framing", twin.params)
                    self.assertEqual(
                        {k: v for k, v in step.params.items() if k != "framing"},
                        twin.params,
                    )
                else:
                    self.assertEqual(step.params, twin.params)

        for key, value in full.globals.items():
            if key == "output_root":
                continue
            with self.subTest(glob=key):
                self.assertEqual(
                    native.globals.get(key), value,
                    f"global '{key}' differs between the two files",
                )

    def test_output_switches_and_the_steps_they_guard_agree(self):
        """A workflow's globals carry `export_colmap`/`export_ply` exactly
        when it has steps guarded by them.

        Both directions are a real failure. A global with no step behind it
        puts a checkbox in the UI that silently does nothing. A guarded step
        whose global is undeclared is caught by `test_when_conditions_resolve`
        above, but the pairing is what keeps the UI honest: the checkboxes
        are derived from the globals, so globals that do not match the steps
        mean the UI is offering the wrong choices.

        All three shipped workflows now declare both: fast_helical_native
        mirrors fast_helical_full after its bootstrap prologue, exports
        included. A workflow that declared a switch with no guarded step
        (or the reverse) is what this still guards against.
        """
        switches = ("export_colmap", "export_ply")
        for path in _workflows():
            spec = WorkflowSpec.from_yaml(str(path))
            for switch in switches:
                guarded = [s.id for s in spec.steps if f"globals.{switch}" in str(s.when)]
                with self.subTest(workflow=path.name, param=switch):
                    self.assertEqual(
                        switch in spec.globals, bool(guarded),
                        f"{path.name}: declares {switch}={switch in spec.globals} "
                        f"but the steps it guards are {guarded or 'none'}",
                    )

    def test_context_paths_look_like_paths(self):
        for path in _workflows():
            spec = WorkflowSpec.from_yaml(str(path))
            for step in spec.steps:
                for name, ctx_path in list(step.inputs.items()) + list(step.outputs.items()):
                    with self.subTest(workflow=path.name, step=step.id, name=name):
                        self.assertIsInstance(ctx_path, str)
                        self.assertTrue(ctx_path)
                        self.assertNotIn(" ", ctx_path)
                        self.assertFalse(ctx_path.startswith("."))
                        self.assertFalse(ctx_path.endswith("."))

    def test_nothing_clobbers_the_splat_alpha_before_mask_splat(self):
        """Between render_splat and mask_splat, dataset.masks is carrying
        the splat render's per-pixel alpha, and mask_splat exists to
        threshold it. Any step in that gap that writes the field without
        reading it destroys that alpha.

        This is the bug that cost a run's output quality: `inject_anchor`
        sat there and manufactured an all-1.0 batch, so mask_splat's
        keep-test passed on every pixel and the stage that blacks out the
        Gaussians' low-confidence fringes became a bare bilateral filter.
        Fixed by moving inject_anchor after mask_splat — the assertion
        below holds either by the gap being empty or by a step in it
        reading what it overwrites.
        """
        for path in _workflows():
            spec = WorkflowSpec.from_yaml(str(path))
            ids = [s.id for s in spec.steps]
            producers = [s for s in spec.steps if s.step == "render_splat"]
            consumers = [s for s in spec.steps if s.step == "mask_splat"]
            if not producers or not consumers:
                continue
            start = ids.index(producers[0].id)
            end = ids.index(consumers[0].id)
            self.assertLess(start, end, f"{path.name}: mask_splat precedes render_splat")
            self.assertIn(
                "dataset.masks", producers[0].outputs.values(),
                f"{path.name}: render_splat does not publish its alpha as dataset.masks",
            )
            for step in spec.steps[start + 1:end]:
                if "dataset.masks" not in step.outputs.values():
                    continue
                with self.subTest(workflow=path.name, step=step.id):
                    self.assertIn(
                        "dataset.masks", step.inputs.values(),
                        f"step '{step.id}' overwrites dataset.masks between "
                        f"render_splat and mask_splat without reading it, so the "
                        f"splat render's alpha never reaches mask_splat",
                    )

    def test_the_anchor_is_reinjected_after_masking(self):
        """Ordering, stated directly, because the pixel-level guard for it
        (tests/test_anchor.py) skips when the recorded run is absent.

        cyber2_6f/masked_splatted/frame_00038_.png is that stage's
        anchor.png byte for byte — the injected photo is not composited
        over black and not bilateral-filtered, which only happens if the
        stage-3 inject_anchor runs *after* mask_splat.

        Two distinct checks, because fast_helical_native legitimately has a
        SECOND inject_anchor: the pre-denoise one in its bootstrap, which
        builds the anchored dataset the fast_helical files are handed
        ready-made. That one belongs before everything here.

          (a) at least one inject_anchor runs after mask_splat — the
              stage-3 re-injection that feeds denoise_pass2;
          (b) none sits in the render_splat -> mask_splat gap, where
              dataset.masks is the splat's per-pixel alpha that mask_splat
              exists to threshold.
        """
        for path in _workflows():
            spec = WorkflowSpec.from_yaml(str(path))
            steps = spec.steps
            ids = [s.id for s in steps]
            if "mask_splat" not in [s.step for s in steps]:
                continue
            mask_at = min(i for i, s in enumerate(steps) if s.step == "mask_splat")
            inject_at = [i for i, s in enumerate(steps) if s.step == "inject_anchor"]
            resplat_at = [i for i, s in enumerate(steps) if s.step == "render_splat"]

            with self.subTest(workflow=path.name, check="reinjected after mask_splat"):
                self.assertTrue(
                    any(i > mask_at for i in inject_at),
                    f"{path.name}: nothing re-injects the anchor after mask_splat "
                    f"('{ids[mask_at]}'); denoise_pass2 would see a re-rendered "
                    f"anchor frame instead of the warped photo",
                )

            if resplat_at:
                lo = min(resplat_at)
                for i in inject_at:
                    if lo < i < mask_at:
                        with self.subTest(workflow=path.name, step=ids[i]):
                            self.fail(
                                f"'{ids[i]}' injects the anchor between render_splat "
                                f"('{ids[lo]}') and mask_splat ('{ids[mask_at]}'), "
                                f"where it manufactures an all-1.0 batch over the "
                                f"splat alpha mask_splat needs",
                            )


    def test_a_warped_anchor_is_bordered_in_grey_not_white(self):
        """generate_firstlast's border colour has no literal in the YAML —
        it arrives as `image_warp["bg_color"]`, which render.py copies from
        the render step's own `bg_color` param. So the workflow-level
        statement of "grey" is that param on whichever render step feeds
        the warp, and its default (white) is wrong here: the anchor frame
        reaches the diffusion pass among renders on mid grey.

        Asserted as "not white and mid-ish" rather than exactly 0.5, since
        the pixel-level value is pinned in test_anchor.py; this guards the
        wiring, which is the part that silently reverts.
        """
        for path in _workflows():
            spec = WorkflowSpec.from_yaml(str(path))
            warps = [s for s in spec.steps if s.step == "generate_firstlast"]
            if not warps:
                continue
            for warp in warps:
                source = warp.inputs.get("bg_color")
                with self.subTest(workflow=path.name, step=warp.id):
                    self.assertIsNotNone(
                        source, f"'{warp.id}' does not wire bg_color at all, so it "
                        f"falls back to the step's white default",
                    )
                    # e.g. "scene.image_warp.bg_color" -> the step that wrote it
                    producers = [
                        s for s in spec.steps
                        if source in s.outputs.values() or
                        any(source.startswith(v + ".") for v in s.outputs.values())
                    ]
                    self.assertTrue(
                        producers, f"nothing in {path.name} writes '{source}'",
                    )
                    bg = producers[0].params.get("bg_color")
                    self.assertIsNotNone(
                        bg, f"'{producers[0].id}' does not set bg_color, so "
                        f"'{warp.id}' borders the warped photo in white",
                    )
                    self.assertNotEqual(
                        [float(c) for c in bg], [1.0, 1.0, 1.0],
                        "white is the one value that cannot be right here",
                    )
                    for c in bg:
                        self.assertGreater(float(c), 0.0)
                        self.assertLess(float(c), 1.0)


    def test_denoise_prompts_match_the_graph_character_for_character(self):
        """See DENOISE_PROMPT's comment: the way this breaks is whitespace,
        so compare the whole string rather than eyeballing the YAML.

        The prompts are now a param of each wan22_vace_denoise step, not a
        workflow global — and each workflow has two denoise passes carrying
        their own copy, so this checks every one.
        """
        seen = 0
        for path in _workflows():
            spec = WorkflowSpec.from_yaml(str(path))
            for step in spec.steps:
                if step.step != "wan22_vace_denoise":
                    continue
                seen += 1
                with self.subTest(workflow=path.name, step=step.id):
                    self.assertEqual(step.params.get("prompt"), DENOISE_PROMPT)
                    self.assertEqual(
                        step.params.get("negative_prompt"), DENOISE_NEGATIVE_PROMPT
                    )
        self.assertGreaterEqual(seen, 6, "expected two denoise passes in each workflow")

    def test_a_mesh_is_reconstructed_from_the_sheet_s_front_half(self):
        """The from-an-image path is handed the two-panel front/back sheet,
        never a photo of the subject, so every consumer of a single view
        has to read one half or the other — and which half is not
        interchangeable:

          * sam3d_body and generate_firstlast want the FRONT. A mesh
            reconstructed from the sheet would be fitted to two people, and
            the anchor warp would paste both panels over the anchor frame.
          * wan22_vace_denoise wants the BACK, and gets it from
            dataset.reference_image, which the split overwrites. The front
            view already reaches that pass as the injected anchor frame, so
            the reference slot carries the one view nothing else supplies.

        Stated at the workflow level because nothing downstream can detect
        the mistake: a sheet is a valid image everywhere it would be used.
        """
        for path in _workflows():
            spec = WorkflowSpec.from_yaml(str(path))
            splits = [s for s in spec.steps if s.step == "split_reference_sheet"]
            reconstructions = [s for s in spec.steps if s.step == "sam3d_body"]
            if not reconstructions:
                self.assertFalse(
                    splits, f"{path.name}: splits a reference sheet but never "
                    f"reconstructs a body from it",
                )
                continue

            with self.subTest(workflow=path.name):
                self.assertEqual(
                    len(splits), 1,
                    f"{path.name}: sam3d_body runs on an image that was never "
                    f"split out of the front/back sheet",
                )
                split = splits[0]
                self.assertEqual(
                    spec.steps.index(split), 0,
                    f"{path.name}: '{split.id}' must run first — anything ahead "
                    f"of it reads the unsplit sheet",
                )
                front = split.outputs["front"]
                back = split.outputs["back"]
                self.assertEqual(
                    back, "dataset.reference_image",
                    "the back half is what wan22_vace_denoise conditions on, and "
                    "it reads dataset.reference_image",
                )
                self.assertNotEqual(front, back)

                wants_the_front = {"sam3d_body": "image", "generate_firstlast": "image"}
                for step in spec.steps:
                    key = wants_the_front.get(step.step)
                    if key is None:
                        continue
                    with self.subTest(step=step.id):
                        self.assertEqual(
                            step.inputs.get(key), front,
                            f"'{step.id}' reads '{step.inputs.get(key)}' where it "
                            f"needs the sheet's front half ('{front}')",
                        )
                for step in spec.steps:
                    if step.step != "wan22_vace_denoise":
                        continue
                    with self.subTest(step=step.id):
                        self.assertEqual(
                            step.inputs.get("reference_image"), back,
                            f"'{step.id}' conditions on "
                            f"'{step.inputs.get('reference_image')}' rather than the "
                            f"sheet's back half ('{back}')",
                        )


if __name__ == "__main__":
    unittest.main()
