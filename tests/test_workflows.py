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

# The denoise prompts, character for character, as every workflow's
# wan22_vace_denoise steps carry them. The negative is still
# ComfyUI-Body2COLMAP's workflows/api/denoise.json node 7; the positive
# started as node 188's string (before StringReplace fills in
# $SUBJECT_DESC$) and deliberately diverged on 2026-08-31, when its
# lighting was anchored to the room instead of to the camera. See the
# comment on denoise_pass1 in fast_helical_native.yaml for why.
#
# Pinned here because the failure mode is silent: YAML's folded `>-`
# scalar turns each line break into a space, and a continuation line
# starting with a space loses it to indentation stripping — either way
# the prompt still loads, still runs, and is not the one intended.
DENOISE_PROMPT = (
    "人工光、柔光、低对比度、环绕运镜。时间静止，人物完全静止。身体"
    "如雕塑般僵硬，胸口没有起伏，头部保持固定角度，面部肌肉完全不动"
    "，维持单一的中性表情。双眼一眨不眨，眼睑保持张开且稳定，目光空"
    "洞而固定，锁定远处墙面上的一个点，对镜头毫无察觉、毫无反应。镜"
    "头以匀速、固定焦距围绕主体平滑移动。灯光属于房间本身，位置固定"
    "：主光始终来自人物自身的左前方，右侧是柔和的补光。人物身上的明"
    "暗、阴影与高光始终停留在同一片皮肤和衣物上，每一帧完全相同——左"
    "侧始终是受光面，右侧始终是柔和的暗面；镜头绕到人物右侧时，看到"
    "的正是那一侧的暗面。 Time is frozen and only the camera moves. "
    "The gaze stays anchored to that point on the wall as the "
    "camera passes, so the eyes slide across the frame, always "
    "aimed past the lens at the wall behind it. Eyelids stay open "
    "and steady, the expression holds without a single "
    "micro-movement, and the head keeps its exact angle throughout. "
    "The frozen subject is $SUBJECT_DESC$, on a plain seamless "
    "backdrop. Soft, low-contrast studio light: the key sits high "
    "on the subject's own left, a soft fill on their right, both "
    "fixed to the room and holding still while the camera arcs "
    "around. Every highlight and shadow stays welded to the same "
    "patch of skin and cloth from the first frame to the last, so "
    "their left remains the lit side and their right remains the "
    "soft-shadowed side for the whole circle; as the camera arcs "
    "round to their right it sees that shadowed side. Matte skin, "
    "matte cloth, photorealistic, sharp focus, identical face, "
    "clothing and lighting in every frame."
)

DENOISE_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，"
    "画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的"
    "，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的"
    "，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背"
    "景，三条腿，背景人很多，倒着走, chewing, tattoos, 眼球运动，"
    "眨眼，眼动追踪，微表情，面部抽动，头部移动, 小头, 小脸"
)


# The from-a-sheet workflows, and the bootstrap prologue each one runs before
# it picks up the shared tail verbatim (see denoise_pass1 on). Pinned here
# rather than derived, because "the prologue changed shape" is exactly the
# edit that should make somebody look: this list is how the pipeline is
# allowed to manufacture the dataset that tail expects.
#
# 2026-08-29: it changed shape. `fix_head_angle` left (it cannot run beside
# `refine_pose_to_splat`), the matte/normals/shell/re-pose steps arrived, and
# `warp_reference_to_anchor` + `reinject_anchor_initial` left because
# `inject_shell_views` does that job — it puts photo-derived content on every
# frame near the source view, not just the one exactly on it, so the render
# no longer has to bend its path to land a camera there either.
#: The face branch — detect_face / locate_face / crop_face / face_seg /
#: face_mask / face_normals / face_splat, then render_face_support_views +
#: face_support_views — is common to both and gated on each file's
#: `face_splat` global. `render_initial_views` carries the rest of it and is
#: NOT gated: it composites the splat itself.
#:
#: 2026-08-30: `face_support_views` joined it. The face renders now reach
#: the trainings twice — composited into the drawings, and again as brush
#: supporting views, which is the only one of the two a diffusion pass
#: cannot rewrite.
#:
#: 2026-08-30 (later): `detect_face` came BACK. Not a reversal of the
#: 2026-08-29 note — the landmarks are geometry now, not an overlay, and
#: nothing draws them. Sapiens2's `parts: face` is Goliath class 3,
#: `Face_Neck`, so the neck can only be intersected out of the matte, not
#: deselected from it: `face_seg` produces it and `face_mask` (now
#: `face_landmark_mask`) intersects the landmark hull with it. `locate_face`
#: stays exactly where it was and still sizes the crop — a face-sized crop
#: was tried and flattens the face fourfold, see the step's docstring.
#: 2026-08-30 (latest): `fix_head_angle` left the native prologue for
#: `map_face_to_mesh` + `fit_head_to_face`, which turn the mesh head to the
#: photograph's face instead of to a fixed lean (steps/head_fit.py), and
#: `detect_face` moved ahead of them, ungated — the fit needs the landmarks
#: whether or not the face splat is drawn.
#: 2026-08-31: `render_face_views` and `composite_face` LEFT, both files.
#: body2colmap dropped gsplat for brush-splat-render, so its own
#: `skeleton+splat` composite mode is usable here and `render_initial_views`
#: draws the overlay itself — see docs/revert-when-body2colmap-drops-gsplat.md.
BOOTSTRAPS = {
    # The shipped default: the older, better-proven bootstrap — circular
    # orbit, anchor warp and injection — with the head re-fitted to the
    # photo and the face splat composited on. Everything else is as it was.
    "fast_helical_native": [
        "split_sheet", "reconstruct_body",
        "detect_face", "map_face_to_mesh", "fit_head_to_face",
        "locate_face", "crop_face", "face_seg", "face_mask",
        "face_normals", "face_splat",
        "render_initial_views", "render_face_support_views",
        "face_support_views",
        "warp_reference_to_anchor", "reinject_anchor_initial",
    ],
    # Parked: the photo-to-splat shell, which replaces the whole bootstrap.
    # Nothing selects it (webui.WORKFLOW_NATIVE names the file above); it is
    # kept in the tree so the shell wiring does not have to be reconstructed
    # from git history, and checked here so it cannot rot silently.
    "fast_helical_shell": [
        "split_sheet", "reconstruct_body",
        "front_matte", "front_normals", "shell_splat", "refine_pose",
        "detect_face", "locate_face", "crop_face", "face_seg", "face_mask",
        "face_normals", "face_splat",
        "render_initial_views", "render_face_support_views",
        "face_support_views",
        "render_shell_views", "inject_shell_band",
    ],
}


def _workflows():
    return sorted(WORKFLOW_DIR.glob("*.yaml"))


def _splat_alpha_producer(case, path, spec, mask_at: int) -> int:
    """Index of the `render_splat` whose per-pixel alpha `mask_splat` is
    about to threshold — i.e. the last one before it that publishes
    `dataset.masks`.

    Not simply "the first render_splat in the file": `fast_helical_native`
    renders a second splat in its bootstrap (the photo-derived shell, along
    the mesh render's own cameras) into a scratch namespace, publishing no
    masks at all. That render is not in this relationship with mask_splat,
    and treating it as the producer would put the entire bootstrap inside a
    gap that does not exist.
    """
    producers = [i for i, step in enumerate(spec.steps[:mask_at])
                 if step.step == "render_splat"
                 and "dataset.masks" in step.outputs.values()]
    case.assertTrue(
        producers,
        f"{path.name}: mask_splat runs with no render_splat publishing its "
        f"alpha as dataset.masks ahead of it, so it has nothing to threshold",
    )
    return producers[-1]


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

    def test_run_upscale_gates_the_upscale_stage(self):
        """fast_helical.yaml used to be a separate workflow minus the
        upscale; it is now the `run_upscale` global gating `upscale` on
        this one file (which also rescales the dataset's cameras as part
        of the same step — see steps/seedvr2.py). Off -> it drops out of
        enabled_steps(); on (the default) -> it runs."""
        spec = WorkflowSpec.from_yaml(str(WORKFLOW_DIR / "fast_helical_native.yaml"))
        gated = {"upscale"}
        self.assertTrue(gated <= {s.id for s in spec.enabled_steps()})
        spec.globals["run_upscale"] = False
        self.assertFalse(gated & {s.id for s in spec.enabled_steps()})

    def test_pre_upscale_colmap_is_off_by_default_and_gated_together(self):
        """The debug stage-4b export (masks + normals + colmap) is three
        steps, all guarded by `export_colmap_preupscale`, all skipped
        unless it is set."""
        preupscale = {"export_masks_preupscale", "export_normals_preupscale",
                      "export_colmap_preupscale"}
        spec = WorkflowSpec.from_yaml(str(WORKFLOW_DIR / "fast_helical_native.yaml"))
        self.assertFalse(spec.globals["export_colmap_preupscale"])
        self.assertFalse(preupscale & {s.id for s in spec.enabled_steps()})
        for step in spec.steps:
            if step.id in preupscale:
                self.assertEqual(step.when, "${globals.export_colmap_preupscale}")
        spec.globals["export_colmap_preupscale"] = True
        self.assertTrue(preupscale <= {s.id for s in spec.enabled_steps()})

    def test_the_intermediate_colmap_is_off_by_default_and_exports_brush_s_own_input(self):
        """The debug export of what the FIRST brush training is handed. One
        step, gated by `export_colmap_intermediate`, and — the part worth
        pinning — it must read the very same context paths `train_splat`
        does. Recomputing the mattes or the normals here would export
        something subtly different from what was trained on, which is the
        one thing this export exists not to do.
        """
        for path in _workflows():
            spec = WorkflowSpec.from_yaml(str(path))
            steps = {s.id: s for s in spec.steps}
            if "train_splat" not in steps:
                continue
            with self.subTest(workflow=path.name):
                self.assertIn("export_colmap_intermediate", steps)
                export = steps["export_colmap_intermediate"]
                self.assertFalse(spec.globals["export_colmap_intermediate"])
                self.assertNotIn("export_colmap_intermediate",
                                 {s.id for s in spec.enabled_steps()})
                self.assertEqual(export.when, "${globals.export_colmap_intermediate}")

                ids = [s.id for s in spec.steps]
                self.assertLess(ids.index("export_colmap_intermediate"),
                                ids.index("train_splat"),
                                "the export must run before the training it "
                                "describes, on the same context")
                brush = steps["train_splat"]
                for name in ("cameras", "image_names", "points_3d", "images",
                             "masks", "normal_maps"):
                    self.assertEqual(
                        export.inputs.get(name), brush.inputs.get(name),
                        f"{path.name}: the intermediate export reads "
                        f"'{export.inputs.get(name)}' for '{name}' where brush "
                        f"reads '{brush.inputs.get(name)}', so it would not be "
                        f"exporting what was trained on",
                    )

    def test_shell_mirrors_native_from_denoise_pass1_on(self):
        """fast_helical_shell.yaml is a bootstrap prologue followed by a
        verbatim copy of fast_helical_native.yaml's tail. There is no
        include mechanism, so the thing worth checking is that the copy has
        not drifted.

        Everything but `output_root` (deliberately per-file) is compared:
        step id, class, dispatch, env, inputs, outputs, when, keep_loaded,
        and the raw params block. A bootstrap file may declare globals of
        its own; what it may not do is disagree about one of native's.
        """
        native = WorkflowSpec.from_yaml(str(WORKFLOW_DIR / "fast_helical_native.yaml"))
        native_tail = native.steps[len(BOOTSTRAPS["fast_helical_native"]):]
        self._assert_mirrors_native(native, native_tail, "fast_helical_shell",
                                     BOOTSTRAPS["fast_helical_shell"])

    def _assert_mirrors_native(self, native, native_tail, name, bootstrap):
        shell = WorkflowSpec.from_yaml(str(WORKFLOW_DIR / f"{name}.yaml"))
        shell_ids = [s.id for s in shell.steps]
        self.assertEqual(
            shell_ids[: len(bootstrap)], bootstrap,
            f"{name}'s bootstrap prologue has changed shape",
        )
        tail = shell.steps[len(bootstrap):]
        self.assertEqual(
            [s.id for s in tail], [s.id for s in native_tail],
            f"{name}'s tail has drifted from fast_helical_native's steps",
        )
        for step in tail:
            twin = next(s for s in native_tail if s.id == step.id)
            with self.subTest(step=step.id):
                self.assertEqual(step.step, twin.step)
                self.assertEqual(step.dispatch, twin.dispatch)
                self.assertEqual(step.env, twin.env)
                self.assertEqual(step.inputs, twin.inputs)
                self.assertEqual(step.outputs, twin.outputs)
                self.assertEqual(step.when, twin.when)
                self.assertEqual(step.keep_loaded, twin.keep_loaded)
                self.assertEqual(step.params, twin.params)

        for key, value in native.globals.items():
            if key == "output_root":
                continue
            with self.subTest(glob=key):
                self.assertEqual(
                    shell.globals.get(key), value,
                    f"global '{key}' differs between {name} and fast_helical_native",
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

        The shipped workflow declares all of these, and the parked shell
        file mirrors it after its own bootstrap prologue, exports included.
        A workflow that declared a switch with no guarded step (or the
        reverse) is what this still guards against. `run_upscale` is in the
        list for the same reason — it is a global whose only job is to gate
        steps.
        """
        switches = ("export_colmap", "export_ply", "export_colmap_preupscale",
                    "export_colmap_intermediate", "run_upscale")
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

    def test_every_input_is_written_before_it_is_read(self):
        """A step reading a context path nothing upstream writes is the
        cheapest possible workflow bug and the most annoying one to hit on a
        pod: the runner raises at that step, forty minutes in, having done
        every expensive thing ahead of it.

        `dataset` and its fields are seeded by the runner before the first
        step. Everything else has to come from an earlier step's `outputs:`,
        matched three ways: the same path, a prefix of it (a step writing
        `scene.image_warp` satisfies a read of `scene.image_warp.camera`),
        or a path under it (`mesh_output: scene` reads the namespace
        `scene.vertices` and friends were assembled into).

        Gating is deliberately ignored — this is a statement about the
        file, and a `when:`-gated step in the shipped workflows either
        overwrites a path that already exists or is read only through an
        optional `?`.

        Optional reads are skipped, because "nothing writes this" is the
        case they exist for: `train_splat` takes the face splat's
        supporting views when the face branch built them, and trains
        without them when `face_splat: false` turns the branch off. See
        pipeline/workflow.py.
        """
        seeded = {"dataset"}
        for path in _workflows():
            spec = WorkflowSpec.from_yaml(str(path))
            written = set(seeded)
            for step in spec.steps:
                for name, ctx_path in step.inputs.items():
                    if ctx_path.endswith("?"):
                        continue
                    satisfied = any(
                        ctx_path == w
                        or ctx_path.startswith(w + ".")
                        or w.startswith(ctx_path + ".")
                        for w in written
                    )
                    with self.subTest(workflow=path.name, step=step.id, input=name):
                        self.assertTrue(
                            satisfied,
                            f"{path.name}: step '{step.id}' reads context path "
                            f"'{ctx_path}', which no earlier step writes",
                        )
                written.update(step.outputs.values())

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
            if not any(s.step == "mask_splat" for s in spec.steps):
                continue
            end = min(i for i, s in enumerate(spec.steps) if s.step == "mask_splat")
            start = _splat_alpha_producer(self, path, spec, end)
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
                lo = _splat_alpha_producer(self, path, spec, mask_at)
                for i in inject_at:
                    if lo < i < mask_at:
                        with self.subTest(workflow=path.name, step=ids[i]):
                            self.fail(
                                f"'{ids[i]}' injects the anchor between render_splat "
                                f"('{ids[lo]}') and mask_splat ('{ids[mask_at]}'), "
                                f"where it manufactures an all-1.0 batch over the "
                                f"splat alpha mask_splat needs",
                            )


    def test_a_confidence_render_is_paired_with_a_passthrough_mask_splat(self):
        """The two halves of one decision, and running both is worse than
        running either.

        `render_splat`'s `confidence` gates on per-Gaussian multi-view
        evidence and hands back the gate as the frame's alpha, composited
        over the cull colour. `mask_splat`'s threshold path then thresholds
        that alpha, composites the grey frames over BLACK and bilateral-
        filters the gate's soft edge — the old cut applied to output that
        already made the decision. Conversely a passthrough mask_splat with
        no confidence render above it drops the fringe stage altogether and
        hands denoise_pass2 the raw splat alpha's mush.
        """
        for path in _workflows():
            spec = WorkflowSpec.from_yaml(str(path))
            if not any(s.step == "mask_splat" for s in spec.steps):
                continue
            mask_at = min(i for i, s in enumerate(spec.steps) if s.step == "mask_splat")
            producer = spec.steps[_splat_alpha_producer(self, path, spec, mask_at)]
            gated = bool(producer.params.get("confidence"))
            mode = spec.steps[mask_at].params.get("mode", "threshold")
            with self.subTest(workflow=path.name):
                self.assertEqual(
                    mode, "passthrough" if gated else "threshold",
                    f"{path.name}: '{producer.id}' renders with confidence="
                    f"{gated} but '{spec.steps[mask_at].id}' runs in "
                    f"'{mode}' — the fringe decision is made twice, or not "
                    f"at all",
                )

    def test_a_confidence_render_has_evidence_to_read(self):
        """`--confidence` needs per-Gaussian evidence. It comes from the
        .ply (brush's `export_evidence`, on by default) or from an
        `evidence_dataset` measured now. With neither, brush-splat-render
        warns, trusts every splat, and the gate degenerates to the same
        alpha `mask_splat` used to threshold — with `mask_splat` now in
        passthrough, i.e. to nothing at all.
        """
        for path in _workflows():
            spec = WorkflowSpec.from_yaml(str(path))
            trainings = [s for s in spec.steps if s.step == "brush"]
            for step in spec.steps:
                if step.step != "render_splat" or not step.params.get("confidence"):
                    continue
                with self.subTest(workflow=path.name, step=step.id):
                    if step.params.get("evidence_dataset"):
                        continue
                    self.assertTrue(
                        trainings,
                        f"{path.name}: '{step.id}' renders with confidence but "
                        f"nothing trained the splat in this workflow and no "
                        f"evidence_dataset is set",
                    )
                    for training in trainings:
                        self.assertNotEqual(
                            training.params.get("export_evidence"), False,
                            f"{path.name}: '{training.id}' exports no evidence, "
                            f"so '{step.id}' has none to gate on",
                        )

    def test_no_confidence_render_feeds_a_step_that_needs_black(self):
        """`select_support_views` requires the render it is handed to be
        premultiplied over black, refusing one that is more than 8/255
        bright where it is fully transparent: it divides the colour back out
        (rgb / a) to get the straight-alpha frame brush's masked mode
        expects, and that only recovers anything if the background was 0. A
        confidence render is the cull colour there — 0.5 grey by default —
        so turning the mode on for a face view is a hard failure at runtime.
        Caught here instead, where the wiring is visible.

        There used to be a second such step, `composite_splat_views`. The
        compositing is `render`'s `...+splat` mode now, which passes the
        background itself (steps/splat.py's `render_splat_layers`), so there
        is no workflow-visible way to get it wrong there any more.
        """
        for path in _workflows():
            spec = WorkflowSpec.from_yaml(str(path))
            by_id = {s.id: s for s in spec.steps}
            producers = {}
            for step in spec.steps:
                if step.step == "render_splat":
                    for ctx in step.outputs.values():
                        producers[ctx] = step.id
            needs_black = {"select_support_views": ("images", "masks")}
            for step in spec.steps:
                if step.step not in needs_black:
                    continue
                for field in needs_black[step.step]:
                    source = producers.get(step.inputs.get(field))
                    if source is None:
                        continue
                    with self.subTest(workflow=path.name, step=step.id, input=field):
                        self.assertFalse(
                            by_id[source].params.get("confidence"),
                            f"{path.name}: '{step.id}' reads '{source}', "
                            f"which renders with confidence — its transparent "
                            f"pixels are the cull colour, not black, and "
                            f"{step.step} refuses that",
                        )

    def test_the_supporting_views_reach_the_stage_2_training_only(self):
        """The face splat's renders go into ONE brush training, the same
        way in every file.

        Three things, all of them wiring rather than code. The stage-2
        training's reads are optional (`?`), which is what lets
        `face_splat: false` turn the whole branch off without touching the
        training. The paths on both sides have to be the same ones, or the
        training silently gets nothing: an optional read of a path nothing
        writes is exactly the failure this feature is built out of. And the
        FINAL training must read none of them — `scene.support_views.*` is
        still populated at that point, so an optional read there would
        quietly succeed and fit the deliverable to renders taken along the
        bootstrap's circular orbit, two denoise passes and an upscale ago.
        """
        # brush's input name -> the output name select_support_views
        # publishes it under.
        support = {"support_images": "images", "support_masks": "masks",
                   "support_cameras": "cameras"}
        for path in _workflows():
            spec = WorkflowSpec.from_yaml(str(path))
            by_id = {s.id: s for s in spec.steps}
            trainings = [s for s in spec.steps if s.step == "brush"]
            if not trainings:
                continue
            with self.subTest(workflow=path.name):
                for step in trainings:
                    final = step.id == "train_final_splat"
                    for name in support:
                        wired = step.inputs.get(name, "")
                        if final:
                            self.assertEqual(
                                wired, "",
                                f"{path.name}: '{step.id}' must not read "
                                f"'{name}' — by then the dataset is the "
                                f"helical re-render and these are the "
                                f"circular bootstrap's renders",
                            )
                        else:
                            self.assertTrue(
                                wired.endswith("?"),
                                f"{path.name}: '{step.id}' must read '{name}' "
                                f"optionally — the branch that writes it is gated",
                            )
                if "face_support_views" not in by_id:
                    continue
                selector = by_id["face_support_views"]
                self.assertEqual(selector.step, "select_support_views")
                self.assertEqual(selector.when, "${globals.face_splat}")
                # It reads the CAP render. The face splat's other route
                # into the run is `render_initial_views` compositing it onto
                # the drawings along the dataset's own cameras; every one of
                # those is a view the training already has a denoised frame
                # for, which is why the supporting views get a render of
                # their own rather than a cull of that batch.
                cap = by_id["render_face_support_views"]
                self.assertEqual(cap.step, "render_splat")
                self.assertEqual(cap.params.get("pattern"), "cap")
                self.assertEqual(cap.params.get("bounds_source"), "splat")
                for name in ("images", "masks", "cameras"):
                    self.assertEqual(
                        selector.inputs[name], cap.outputs[name],
                        f"{path.name}: face_support_views must read the cap "
                        f"render's {name}, not the composite batch's",
                    )
                # The cap owns the outer edge, and it is TIGHTER than the
                # angle the drawings are composited at: nothing rewrites a
                # supporting view, while a composited frame is an input to
                # two denoise passes that can rewrite a flared rim.
                drawing = by_id["render_initial_views"]
                # The splat's FIRST route into the run: the drawing render
                # composites it itself, through body2colmap's own
                # `skeleton+splat` mode. The read is optional because the
                # branch that builds the .ply is gated and this step is not
                # — with `face_splat: false` there is nothing to resolve.
                self.assertTrue(
                    drawing.params["render_mode"].endswith("+splat"),
                    f"{path.name}: render_initial_views must draw the face splat "
                    f"itself, got render_mode "
                    f"{drawing.params['render_mode']!r}",
                )
                self.assertEqual(
                    drawing.inputs.get("splat_path"),
                    by_id["face_splat"].outputs["splat_path"] + "?",
                    f"{path.name}: render_initial_views must read the face "
                    f"splat's .ply, and read it optionally",
                )
                self.assertLess(
                    float(cap.params["cap_radius_deg"]),
                    float(drawing.params["splat_max_angle_deg"]),
                    f"{path.name}: the supporting views must be sampled from a "
                    f"tighter band than the drawings are composited over",
                )
                # The inner edge of the band is the denoising path, so the
                # step has to be handed the cameras the training's own
                # frames were rendered and denoised along — and a pivot to
                # measure the angle to them about. Wire neither and the
                # edge silently does not apply, which is the failure mode
                # worth a test: nothing raises, and the deliverable is
                # fitted to renders of views it already has photographs of.
                training = by_id["train_splat"]
                self.assertEqual(
                    selector.inputs.get("path_cameras"), training.inputs["cameras"],
                    f"{path.name}: face_support_views must measure against the "
                    f"same cameras train_splat trains on",
                )
                self.assertEqual(
                    selector.inputs.get("splat_center"),
                    "scene.face_splat_stats.world_center",
                    f"{path.name}: face_support_views must measure about the "
                    f"splat's own centre, which is the pivot render_initial_views "
                    f"culls about too — for a head on a full-body orbit the "
                    f"splat's centre and the orbit target differ",
                )
                # The training reads ONE support_* triple and there are two
                # producers of one, so the path between them is the merge —
                # which runs ungated precisely so that path has a single
                # writer whatever the branches are doing.
                merge = by_id["merge_support_views"]
                self.assertEqual(merge.step, "merge_support_views")
                self.assertIs(
                    merge.when, True,
                    f"{path.name}: merge_support_views must run ungated, or "
                    f"scene.support_views.* has no writer with the branches off",
                )
                for name, field in support.items():
                    self.assertEqual(
                        merge.inputs[f"a_{field}"].rstrip("?"),
                        selector.outputs[field],
                        f"{path.name}: the merge reads 'a_{field}' from a path "
                        f"face_support_views does not write",
                    )

                ids = [s.id for s in spec.steps]
                for step in trainings:
                    if step.id == "train_final_splat":
                        continue
                    self.assertLess(
                        ids.index("face_support_views"), ids.index("merge_support_views"),
                        f"{path.name}: the supporting views must be selected "
                        f"before they are merged",
                    )
                    self.assertLess(
                        ids.index("merge_support_views"), ids.index(step.id),
                        f"{path.name}: the supporting views must be merged "
                        f"before '{step.id}' trains on them",
                    )
                    for name, field in support.items():
                        self.assertEqual(
                            step.inputs[name].rstrip("?"), merge.outputs[field],
                            f"{path.name}: '{step.id}' reads '{name}' from a path "
                            f"merge_support_views does not write",
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


    def test_denoise_prompts_are_identical_in_every_pass(self):
        """See DENOISE_PROMPT's comment: the way this breaks is whitespace,
        so compare the whole string rather than eyeballing the YAML. (It
        no longer checks the ComfyUI graph — the positive prompt diverged
        from it deliberately — but all four copies must still agree.)

        The prompts are now a param of each wan22_vace_denoise step, not a
        workflow global — and each workflow has two denoise passes carrying
        their own copy, so this checks every one.
        """
        workflows = _workflows()
        seen = 0
        for path in workflows:
            spec = WorkflowSpec.from_yaml(str(path))
            passes = 0
            for step in spec.steps:
                if step.step != "wan22_vace_denoise":
                    continue
                seen += 1
                passes += 1
                with self.subTest(workflow=path.name, step=step.id):
                    self.assertEqual(step.params.get("prompt"), DENOISE_PROMPT)
                    self.assertEqual(
                        step.params.get("negative_prompt"), DENOISE_NEGATIVE_PROMPT
                    )
            self.assertEqual(passes, 2, f"{path.name}: expected two denoise passes")
        self.assertEqual(seen, 2 * len(workflows))

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


class TestIncompatibleSteps(unittest.TestCase):
    """Step pairs that each work and are wrong together.

    `head_angle_fix` deforms the mesh directly without touching the MHR pose
    parameters; `refine_pose_to_splat` regenerates the mesh *from* those
    parameters. Run together, one silently discards the other — and they are
    corrections in opposite directions besides (the head fix is an
    anatomical prior; the pose fit chases what the shell observed, and the
    shell inherited the same craned head from the same photo).
    """

    def _spec(self, *step_names, globals_=None):
        from pipeline.workflow import StepSpec, WorkflowSpec

        return WorkflowSpec(
            name="synthetic",
            globals=dict(globals_ or {}),
            steps=[StepSpec(id=f"s{i}", step=name)
                   for i, name in enumerate(step_names)],
        )

    def test_the_shipped_workflows_do_not_mix_them(self):
        for path in sorted(WORKFLOW_DIR.glob("*.yaml")):
            spec = WorkflowSpec.from_yaml(path)
            spec.validate()          # raises if any pair is enabled together

    def test_enabling_both_is_refused(self):
        spec = self._spec("head_angle_fix", "refine_pose_to_splat")
        with self.assertRaises(ValueError) as caught:
            spec.validate()
        message = str(caught.exception)
        self.assertIn("head_angle_fix", message)
        self.assertIn("refine_pose_to_splat", message)
        # The refusal has to say why, not just that.
        self.assertIn("pose parameters", message)

    def test_either_one_alone_is_fine(self):
        self._spec("head_angle_fix").validate()
        self._spec("refine_pose_to_splat").validate()

    def test_a_when_gated_step_that_is_off_does_not_count(self):
        """`when:` is how this pipeline makes a step optional, so a pair is
        only a conflict when both actually run."""
        from pipeline.workflow import StepSpec, WorkflowSpec

        spec = WorkflowSpec(
            name="synthetic",
            globals={"fix_head": False},
            steps=[StepSpec(id="a", step="head_angle_fix",
                            when="${globals.fix_head}"),
                   StepSpec(id="b", step="refine_pose_to_splat")],
        )
        spec.validate()

        spec.globals["fix_head"] = True
        with self.assertRaises(ValueError):
            spec.validate()


class TestDeclaredSettings(unittest.TestCase):
    """The `settings:` and `outputs:` blocks, which are the whole UI.

    The web UI holds no table of a workflow's knobs any more: it draws what
    these declare. So a mistake here is a mistake on the form, and the point
    of every check below is that it fires at load rather than at submit or
    forty minutes into a pod run.
    """

    def _write(self, body: str) -> str:
        import os
        import tempfile

        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
            handle.write(body)
            path = handle.name
        self.addCleanup(os.unlink, path)
        return path

    def test_every_shipped_workflow_declares_its_form(self):
        for path in _workflows():
            spec = WorkflowSpec.from_yaml(str(path))
            with self.subTest(workflow=path.name):
                self.assertTrue(spec.settings, "no settings: block")
                self.assertTrue(spec.outputs, "no outputs: block")
                for param in spec.settings:
                    self.assertTrue(param.help, f"{param.name} has no help")
                for output in spec.outputs:
                    self.assertTrue(output.label and output.help and output.directory)

    def test_resolution_and_framing_are_pipeline_settings(self):
        """The two knobs the pipeline exists to let somebody turn. Both were
        bare globals the UI had to carry a hardcoded choice list for."""
        for path in _workflows():
            spec = WorkflowSpec.from_yaml(str(path))
            by_name = {p.name: p for p in spec.settings}
            with self.subTest(workflow=path.name):
                self.assertEqual(by_name["resolution"].type, list)
                self.assertIn([720, 1280], by_name["resolution"].choices)
                self.assertEqual(
                    tuple(by_name["framing"].choices),
                    ("full", "torso", "bust", "head"),
                )

    def test_a_settings_choice_set_matches_the_step_param_it_feeds(self):
        """`framing` reaches `render` as `${globals.framing}`, so the two
        choice lists have to be the same list. They used to be two — one in
        the step class, one in a GLOBAL_CHOICES dict in webui.py with a
        comment asking the next person to keep them in step."""
        from pipeline.registry import get_step_class

        render_choices = get_step_class("render").declared_params()["framing"].choices
        for path in _workflows():
            spec = WorkflowSpec.from_yaml(str(path))
            framing = next(p for p in spec.settings if p.name == "framing")
            with self.subTest(workflow=path.name):
                self.assertEqual(tuple(framing.choices), tuple(render_choices))

    def test_the_outputs_dirs_are_the_ones_the_export_steps_write(self):
        """`dir:` is what packages a finished run, so it has to be the
        directory the step gated by that output actually writes under
        output_root."""
        for path in _workflows():
            spec = WorkflowSpec.from_yaml(str(path))
            written = resolve(
                [step.params for step in spec.steps], {"globals": spec.globals}
            )
            blob = repr(written)
            root = spec.globals["output_root"]
            for output in spec.outputs:
                with self.subTest(workflow=path.name, output=output.name):
                    self.assertIn(f"{root}/{output.directory}", blob)

    def test_a_setting_nothing_reads_is_refused(self):
        path = self._write(
            "name: orphan\n"
            "settings:\n"
            "  - name: unused\n    default: 3\n    help: nothing reads me\n"
            "steps:\n"
            "  - id: a\n    step: rmbg\n    dispatch: in_process\n"
        )
        with self.assertRaises(ValueError) as caught:
            WorkflowSpec.from_yaml(path).validate()
        self.assertIn("unused", str(caught.exception))

    def test_a_name_declared_twice_is_refused(self):
        path = self._write(
            "name: clash\n"
            "settings:\n"
            "  - name: run_it\n    default: true\n    help: x\n"
            "globals:\n  run_it: false\n"
            "steps:\n"
            "  - id: a\n    step: rmbg\n    dispatch: in_process\n"
            "    when: ${globals.run_it}\n"
        )
        with self.assertRaises(ValueError) as caught:
            WorkflowSpec.from_yaml(path)
        self.assertIn("one home", str(caught.exception))

    def test_a_default_outside_its_own_choices_is_refused(self):
        path = self._write(
            "name: badchoice\n"
            "settings:\n"
            "  - name: mode\n    default: sideways\n    help: x\n"
            "    choices: [up, down]\n"
            "steps:\n"
            "  - id: a\n    step: rmbg\n    dispatch: in_process\n"
            "    params:\n      device: ${globals.mode}\n"
        )
        with self.assertRaises(ValueError) as caught:
            WorkflowSpec.from_yaml(path).validate()
        self.assertIn("choices", str(caught.exception))

    def test_a_default_that_does_not_fit_its_type_is_refused(self):
        path = self._write(
            "name: badtype\n"
            "settings:\n"
            "  - name: count\n    type: int\n    default: many\n    help: x\n"
            "steps:\n"
            "  - id: a\n    step: rmbg\n    dispatch: in_process\n"
            "    params:\n      batch_size: ${globals.count}\n"
        )
        with self.assertRaises(ValueError) as caught:
            WorkflowSpec.from_yaml(path).validate()
        self.assertIn("count", str(caught.exception))

    def test_an_output_requiring_something_undeclared_is_refused(self):
        path = self._write(
            "name: badreq\n"
            "outputs:\n"
            "  - name: export_thing\n    dir: thing\n    label: Thing\n"
            "    requires: no_such_setting\n"
            "steps:\n"
            "  - id: a\n    step: rmbg\n    dispatch: in_process\n"
            "    when: ${globals.export_thing}\n"
        )
        with self.assertRaises(ValueError) as caught:
            WorkflowSpec.from_yaml(path).validate()
        self.assertIn("no_such_setting", str(caught.exception))

    def test_an_output_with_no_dir_is_refused(self):
        path = self._write(
            "name: nodir\n"
            "outputs:\n  - name: export_thing\n    label: Thing\n"
            "steps:\n  - id: a\n    step: rmbg\n    dispatch: in_process\n"
        )
        with self.assertRaises(ValueError) as caught:
            WorkflowSpec.from_yaml(path)
        self.assertIn("dir", str(caught.exception))

    def test_a_settings_value_is_coerced_the_way_a_step_param_is(self):
        """`--param run_upscale=no` and a text box both hand over strings,
        and `bool("no")` is True."""
        spec = WorkflowSpec.from_yaml(str(WORKFLOW_DIR / "fast_helical_native.yaml"))
        self.assertIs(spec.coerce_global("run_upscale", "no"), False)
        self.assertEqual(spec.coerce_global("seed", "7"), 7)
        # An undeclared global has no type to be brought to.
        self.assertEqual(spec.coerce_global("output_root", 3), 3)

    def test_one_seed_reaches_every_stochastic_step(self):
        """It was three step params holding 0, 0 and 42 — one run drawing
        three unrelated samples."""
        for path in _workflows():
            spec = WorkflowSpec.from_yaml(str(path))
            readers = [step.id for step in spec.steps
                       if step.params.get("seed") == "${globals.seed}"]
            with self.subTest(workflow=path.name):
                self.assertEqual(len(readers), 3, readers)
                self.assertTrue(any("upscale" in r for r in readers))
