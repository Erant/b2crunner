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
from pipeline.workflow import WorkflowSpec, truthy

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
# $SUBJECT_DESC$) and deliberately diverged twice: on 2026-08-31 its
# lighting was anchored to the room instead of to the camera, and on
# 2026-09-01 the room itself replaced the seamless backdrop, to match the
# `background` render setting. See the comment on denoise_pass1 in
# helical.yaml for why.
#
# Pinned here because the failure mode is silent: YAML's folded `>-`
# scalar turns each line break into a space, and a continuation line
# starting with a space loses it to indentation stripping — either way
# the prompt still loads, still runs, and is not the one intended.
DENOISE_PROMPT = (
    "人工光、柔光、低对比度、环绕运镜。时间静止，人物完全静止。身体如雕塑般"
    "僵硬，胸口没有起伏，头部保持固定角度，面部肌肉完全不动，维持单一的中性"
    "表情。双眼一眨不眨，眼睑保持张开且稳定，目光空洞而固定，锁定远处墙面上"
    "的一个点，对镜头毫无察觉、毫无反应。人物站在一间空房间里：灰色墙面上有"
    "均匀的网格线，深色地板，浅色天花板，墙面相交处有清晰的墙角。房间本身固"
    "定不动，镜头以匀速、固定焦距围绕主体平滑移动，墙角依次从画面中掠过。灯"
    "光属于房间本身，位置固定：主光始终来自人物自身的左前方，右侧是柔和的补"
    "光。人物身上的明暗、阴影与高光始终停留在同一片皮肤和衣物上，每一帧完全"
    "相同——左侧始终是受光面，右侧始终是柔和的暗面；镜头绕到人物右侧时，看到"
    "的正是那一侧的暗面。房间里的阴影同样固定：墙面上的明暗、以及人物投在地"
    "面上的影子，始终落在房间中的同一位置，不随镜头移动。 Time is frozen "
    "and only the camera moves. The gaze stays anchored to that point on "
    "the wall as the camera passes, so the eyes slide across the frame, "
    "always aimed past the lens at the wall behind it. Eyelids stay open "
    "and steady, the expression holds without a single micro-movement, and "
    "the head keeps its exact angle throughout. The frozen subject is "
    "$SUBJECT_DESC$, standing in an empty room: grey walls ruled with an "
    "even grid, a dark floor and a lighter ceiling, meeting at clear "
    "corners. The room is fixed in place and the camera travels through "
    "it, so the walls and their corners sweep past the frame while the "
    "room itself never turns. Soft, low-contrast studio light: the key "
    "sits high on the subject's own left, a soft fill on their right, both "
    "fixed to the room and holding still while the camera arcs around. "
    "Every highlight and shadow stays welded to the same patch of skin and "
    "cloth from the first frame to the last, so their left remains the lit "
    "side and their right remains the soft-shadowed side for the whole "
    "circle; as the camera arcs round to their right it sees that shadowed "
    "side. The room's shadows are just as fixed: the shading on the walls "
    "and the shadow the subject casts on the floor stay in the same place "
    "in the room in every frame, and do not swing round with the camera. "
    "Matte skin, matte cloth, photorealistic, sharp focus, identical face, "
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


# The from-a-sheet workflow, and the bootstrap prologue it runs before it
# picks up the tail (see denoise_pass1 on). Pinned here
# rather than derived, because "the prologue changed shape" is exactly the
# edit that should make somebody look: this list is how the pipeline is
# allowed to manufacture the dataset that tail expects.
#: The face branch — detect_face / locate_face / crop_face / face_seg /
#: face_mask / face_normals / face_splat, then render_face_support_views +
#: face_support_views — is gated on the `face_splat` global. `render_initial_views` carries the rest of it and is
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
#: 2026-08-31: `render_face_views` and `composite_face` LEFT.
#: body2colmap dropped gsplat for brush-splat-render, so its own
#: `skeleton+splat` composite mode is usable here and `render_initial_views`
#: draws the overlay itself — see docs/revert-when-body2colmap-drops-gsplat.md.
#: 2026-09-02: `render_face_support_views` and `face_support_views` LEFT the
#: bootstrap for the tail, where they follow `face_splat_refined` —
#: the face splat built again through the REFINED anchor camera after
#: `refine_cameras`. The bootstrap `face_splat` stays: render_initial_views
#: still composites it onto the drawings before the denoise.
BOOTSTRAPS = {
    # The shipped default: the older, better-proven bootstrap — circular
    # orbit, anchor warp and injection — with the head re-fitted to the
    # photo and the face splat composited on. Everything else is as it was.
    "helical": [
        "split_sheet", "reconstruct_body",
        "detect_face", "map_face_to_mesh", "fit_head_to_face",
        "locate_face", "crop_face", "face_seg", "face_mask",
        "face_normals", "face_splat",
        "render_initial_views",
        "warp_reference_to_anchor", "reinject_anchor_initial",
    ],
    # The EXPERIMENT (2026-09-19): the same bootstrap with a whole-body
    # pointmap shell built before the render (front_matte / front_normals /
    # shell_splat) and the render drawing a helix that starts on the
    # photograph. The shell's two remaining steps (render_shell_views,
    # inject_shell_views) sit after the gated re-outline block, outside
    # this prologue — see SHELL_ONLY_STEPS and the mirror test.
    "helical_shell": [
        "split_sheet", "reconstruct_body",
        "detect_face", "map_face_to_mesh", "fit_head_to_face",
        "locate_face", "crop_face", "face_seg", "face_mask",
        "face_normals", "face_splat",
        "front_matte", "front_normals", "shell_splat",
        "render_initial_views",
        "warp_reference_to_anchor", "reinject_anchor_initial",
    ],
}

#: The steps helical_shell.yaml has and helical.yaml does not. Take them
#: out and the two files must agree step for step (test_helical_shell_
#: mirrors_helical), with exactly the differences that test lists.
SHELL_ONLY_STEPS = (
    "front_matte", "front_normals", "shell_splat",
    "render_shell_views", "inject_shell_views",
)

#: The one sentence pass 1 and the re-outline pass carry that pass 2 does
#: not, since 2026-09-20: their control is a drawing, whose silhouette and
#: skeleton read the same from the front and from behind, and this says the
#: orbit is whole. It names no side and describes no rear view — the route
#: text that was tried and reverted on 2026-09-13 (see the comment above
#: denoise_pass1's prompt in helical.yaml). Pass 2's control is a render.
ORBIT_SENTENCE = (
    "The camera pans slowly around the subject in a 360 degree orbit, "
    "moving smoothly around the subject in a continuous arc. "
)
_ROOM = "meeting at clear corners. "
assert DENOISE_PROMPT.count(_ROOM) == 1
DRAWING_PROMPT = DENOISE_PROMPT.replace(_ROOM, _ROOM + ORBIT_SENTENCE)

#: helical_shell.yaml's first denoise (and its re-outline pass, which must
#: read as pass 1 does) carries the `elevation_hint` setting appended to the
#: drawing prompt — one sentence per language about the camera climbing,
#: blank for the arm without. Pass 2's prompt is the pinned one.
HINTED_DENOISE_PROMPT = DRAWING_PROMPT + " ${globals.elevation_hint}"


def _workflows():
    return sorted(WORKFLOW_DIR.glob("*.yaml"))


#: The steps that rasterise the splat along a path. `render_subject` was a
#: second one from 2026-09-17 to 2026-09-20 (render_splat with the removed
#: mesh path's mesh as an alternative source); it is a `render_splat` step
#: with that id now.
_SPLAT_RENDERS = ("render_splat",)


def _splat_alpha_producer(case, path, spec, mask_at: int) -> int:
    """Index of the `render_splat` whose per-pixel alpha `mask_splat` is
    about to threshold — i.e. the last one before it that publishes
    `dataset.masks`.

    Not simply "the first render_splat in the file": a bootstrap may render
    a splat of its own without publishing masks at all (the retired shell
    bootstrap did). Such a render is not in this relationship with
    mask_splat, and treating it as the producer would put the entire
    bootstrap inside a gap that does not exist.
    """
    producers = [i for i, step in enumerate(spec.steps[:mask_at])
                 if step.step in _SPLAT_RENDERS
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

    def test_every_debug_dir_points_under_output_root_debug(self):
        """`build_result_zip` packages `<output_root>/debug/` (and the face
        .ply directories) beside the deliverables; a `debug_dir` aimed
        anywhere else is written and never downloaded. Both refinements and
        both face splats have one — the four things run 5e2817's face-cap
        diagnosis (2026-09-04) needed and did not have."""
        for path in _workflows():
            spec = WorkflowSpec.from_yaml(str(path))
            wired = {}
            for step in spec.steps:
                if "debug_dir" in step.params:
                    wired[step.id] = step.params["debug_dir"]
            with self.subTest(workflow=path.name):
                # Every refinement and every face splat, however many a
                # file has.
                for step in spec.steps:
                    if step.step in ("refine_cameras", "face_pointmap_splat"):
                        self.assertIn(step.id, wired)
                for step_id, value in wired.items():
                    self.assertTrue(
                        value.startswith("${globals.output_root}/debug/"),
                        f"{path.name}: {step_id}.debug_dir = {value!r}")
                # One directory per step: two steps sharing one would
                # overwrite each other's dumps.
                self.assertEqual(len(set(wired.values())), len(wired))

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
        spec = WorkflowSpec.from_yaml(str(WORKFLOW_DIR / "helical.yaml"))
        gated = {"upscale"}
        self.assertTrue(gated <= {s.id for s in spec.enabled_steps()})
        spec.globals["run_upscale"] = False
        self.assertFalse(gated & {s.id for s in spec.enabled_steps()})

    def test_pre_upscale_colmap_wants_the_debug_bundle_and_the_upscale(self):
        """The debug stage-4b export (masks + colmap) is two steps, both
        guarded by the same conjunction, both skipped unless it holds. It
        was three until 2026-09-06: the normals were estimated for a brush
        training that has stopped supervising on them
        (docs/final-splat-alignment-guide.md §1).

        `export_debug` because this is a member of the debug bundle, and
        `run_upscale` because with the upscale off these are colmap/'s own
        frames — which is what `requires: run_upscale` said while it was an
        output of its own, up to 2026-09-08.
        """
        preupscale = {"export_masks_preupscale", "export_colmap_preupscale"}
        spec = WorkflowSpec.from_yaml(str(WORKFLOW_DIR / "helical.yaml"))
        for step in spec.steps:
            if step.id in preupscale:
                self.assertEqual(
                    step.when,
                    ["${globals.export_debug}", "${globals.run_upscale}"],
                )
        # Both on (the defaults) is the only combination that runs them.
        self.assertTrue(preupscale <= {s.id for s in spec.enabled_steps()})
        for off in ("export_debug", "run_upscale"):
            with self.subTest(off=off):
                spec.globals.update(export_debug=True, run_upscale=True)
                spec.globals[off] = False
                self.assertFalse(preupscale & {s.id for s in spec.enabled_steps()})

    def test_the_intermediate_colmap_rides_the_debug_bundle_and_exports_brush_s_own_input(self):
        """The debug export of what the FIRST brush training is handed. One
        step, gated by `export_debug` since 2026-09-08 — it is archived
        under `debug/`, and it is the one member of that bundle that is not
        a free side effect of work the run does anyway, so the switch skips
        it rather than only dropping it from the .zip. The part worth
        pinning is the wiring: it must read the very same context paths
        `train_splat` does. Recomputing the mattes or the normals here
        would export something subtly different from what was trained on,
        which is the one thing this export exists not to do.
        """
        for path in _workflows():
            spec = WorkflowSpec.from_yaml(str(path))
            steps = {s.id: s for s in spec.steps}
            if "train_splat" not in steps:
                continue
            with self.subTest(workflow=path.name):
                self.assertIn("export_colmap_intermediate", steps)
                export = steps["export_colmap_intermediate"]
                self.assertEqual(export.when, "${globals.export_debug}")
                self.assertTrue(spec.globals["export_debug"])
                self.assertIn("export_colmap_intermediate",
                              {s.id for s in spec.enabled_steps()})
                spec.globals["export_debug"] = False
                self.assertNotIn("export_colmap_intermediate",
                                 {s.id for s in spec.enabled_steps()})
                spec.globals["export_debug"] = True

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

    def test_output_switches_and_the_steps_they_guard_agree(self):
        """A workflow's globals carry `export_colmap`/`export_ply` exactly
        when it has steps guarded by them.

        Both directions are a real failure. A global with no step behind it
        puts a checkbox in the UI that silently does nothing. A guarded step
        whose global is undeclared is caught by `test_when_conditions_resolve`
        above, but the pairing is what keeps the UI honest: the checkboxes
        are derived from the globals, so globals that do not match the steps
        mean the UI is offering the wrong choices.

        The shipped workflow declares all of these. A workflow that declared a switch with no guarded step (or the
        reverse) is what this still guards against. `run_upscale` is in the
        list for the same reason — it is a global whose only job is to gate
        steps.
        """
        switches = ("export_colmap", "export_ply", "export_debug", "run_upscale")
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

        `rmbg` is the exception, and it is the reason the guard is worded as
        "nothing CLOBBERS it" rather than "nothing writes it": since
        2026-09-05 the matte mask_splat composites with is deliberately not
        the render's alpha but one measured against the frames themselves.
        Replacing it is that step's whole job, so a step that has no other,
        and it cannot read what it replaces.

        (`render_subject`, 2026-09-17, is not in the gap: it IS the producer,
        drawing either the splat or the textured mesh, so its alpha is always
        the frames' own.)
        """
        deliberate = {"rmbg"}
        for path in _workflows():
            spec = WorkflowSpec.from_yaml(str(path))
            if not any(s.step == "mask_splat" for s in spec.steps):
                continue
            end = min(i for i, s in enumerate(spec.steps) if s.step == "mask_splat")
            start = _splat_alpha_producer(self, path, spec, end)
            for step in spec.steps[start + 1:end]:
                if "dataset.masks" not in step.outputs.values():
                    continue
                if step.step in deliberate:
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

        Two distinct checks, because helical legitimately has a
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
            resplat_at = [i for i, s in enumerate(steps) if s.step in _SPLAT_RENDERS]

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


    def test_an_inject_anchor_has_a_path_that_reaches_the_anchor(self):
        """An `inject_anchor` that can actually match something.

        The step matches the anchor by camera POSITION, so it is only ever
        as good as the path the batch was rendered along. A `render_splat`
        that builds a fresh orbit without anchoring it puts no camera on the
        anchor at all — measured on cyber_6f with helical's own
        params, the nearest unanchored helical camera is 0.1408 from it
        against a 0.00593 tolerance, 24x — so the injection matches zero
        frames, returns the batch untouched and the run carries on. That was
        this file's state until 2026-09-04: denoise_pass2 conditioned on 81
        synthetic frames and an all-1.0 VACE mask, with the reference
        photograph nowhere in it.

        So: whenever a workflow injects an anchor into a batch that a
        `render_splat` re-rendered along a NEW path, that render must be
        anchored, and must publish where it put the anchor rather than
        leaving the extras describing the old path. A file with no anchor
        image to inject has nothing to check.
        """
        for path in _workflows():
            spec = WorkflowSpec.from_yaml(str(path))
            steps = spec.steps
            by_index = {i: s for i, s in enumerate(steps)}
            # An anchor image is what makes the injection live at all: with
            # none, inject_anchor passes through by design.
            writes_anchor_image = any(
                "dataset.anchor_image" in s.outputs.values() for s in steps
            )
            if not writes_anchor_image:
                continue

            for i, step in by_index.items():
                if step.step != "inject_anchor":
                    continue
                # The last render_splat before this injection that rebuilt
                # the batch — i.e. one with a `pattern`. Without one the
                # batch is still on the cameras the anchor was recorded
                # against and there is nothing to re-anchor.
                repaths = [
                    j for j, s in by_index.items()
                    if j < i and s.step in _SPLAT_RENDERS and s.params.get("pattern")
                    and "dataset.cameras" in s.outputs.values()
                ]
                if not repaths:
                    continue
                render = by_index[max(repaths)]
                with self.subTest(workflow=path.name, step=step.id):
                    anchored = render.params.get("override_cam_from_mesh")
                    self.assertTrue(
                        truthy(resolve(anchored, {"globals": spec.globals}))
                        if isinstance(anchored, str) else truthy(anchored),
                        f"{path.name}: '{step.id}' injects the anchor by position "
                        f"into the batch '{render.id}' re-rendered along a fresh "
                        f"'{render.params.get('pattern')}' path, but that render is "
                        f"not anchored (override_cam_from_mesh={anchored!r}), so no "
                        f"camera in it sits on the anchor and the injection is a "
                        f"silent no-op",
                    )
                    self.assertEqual(
                        render.outputs.get("anchor_position"),
                        "dataset.extras.anchor_position",
                        f"{path.name}: '{render.id}' anchors its path but does not "
                        f"publish anchor_position, so '{step.id}' would match "
                        f"against whichever anchor an earlier step left in the "
                        f"extras — refine_cameras republishes it ~62mm off the "
                        f"origin, ~10x the match tolerance",
                    )
                    # And if the cameras were refined between the source
                    # render and this one, the splat was trained with the
                    # photograph's camera OFF the pose the new path anchors
                    # on: the render must carry that delta onto its path
                    # (`given_anchor_camera`, splat.py's
                    # _carry_anchor_refinement) or the injected photo and the
                    # renders beside it disagree by it — 8-10 px at the
                    # subject in every run measured on 2026-09-08.
                    refined_before = [
                        j for j, s in by_index.items()
                        if j < max(repaths) and s.step == "refine_cameras"
                        and "dataset.cameras" in s.outputs.values()
                    ]
                    if refined_before:
                        self.assertTrue(
                            render.inputs.get("given_anchor_camera"),
                            f"{path.name}: '{by_index[refined_before[-1]].id}' "
                            f"rewrote dataset.cameras before '{render.id}' "
                            f"re-anchored its path, but the render wires no "
                            f"`given_anchor_camera`, so its anchor frame sits on "
                            f"the GIVEN anchor pose while the splat it renders was "
                            f"trained with the photograph's camera at the refined "
                            f"one — '{step.id}' then injects the photo where the "
                            f"renders beside it disagree with it",
                        )

    def test_a_confidence_render_is_not_thresholded_again(self):
        """The two halves of one decision, and running both is worse than
        running either.

        `render_splat`'s `confidence` gates on per-Gaussian multi-view
        evidence and hands back the gate as the frame's alpha, already
        applied to the RGB. `mask_splat`'s threshold path then thresholds
        that alpha, composites over BLACK and bilateral-filters the gate's
        soft edge — the old cut applied to output that already made the
        decision. Conversely a mask_splat that does NOT threshold with no
        confidence render above it drops the fringe stage altogether and
        hands denoise_pass2 the raw splat alpha's mush.

        Which of the two non-thresholding modes a gated render is paired
        with is a separate question — `passthrough` leaves the frames on the
        cull colour, `composite` lays them over `bg_color` with whatever
        matte reached `dataset.masks` — and not one this can settle from the
        wiring alone.
        """
        for path in _workflows():
            spec = WorkflowSpec.from_yaml(str(path))
            if not any(s.step == "mask_splat" for s in spec.steps):
                continue
            mask_at = min(i for i, s in enumerate(spec.steps) if s.step == "mask_splat")
            producer = spec.steps[_splat_alpha_producer(self, path, spec, mask_at)]
            gated = bool(producer.params.get("confidence"))
            mode = spec.steps[mask_at].params.get("mode", "threshold")
            thresholds = mode == "threshold"
            with self.subTest(workflow=path.name):
                self.assertEqual(
                    thresholds, not gated,
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
                if step.step not in _SPLAT_RENDERS or not step.params.get("confidence"):
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
                if step.step in _SPLAT_RENDERS:
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

    def test_the_supporting_views_reach_a_training_they_are_fresh_for(self):
        """The face splat's renders go into a brush training the same way
        in every file, and only into one they describe.

        Three things, all of them wiring rather than code. A training's
        reads are optional (`?`), which is what lets `face_splat: false`
        turn the whole branch off without touching the training. The paths
        on both sides have to be the same ones, or the training silently
        gets nothing: an optional read of a path nothing writes is exactly
        the failure this feature is built out of. And a training may read
        them only if nothing REPLACED the dataset between the cap render
        and the training — a `render_splat` that rebuilt the path, a
        `seedvr2` that resized the frames and rescaled the cameras. The
        final training is the case: `scene.support_views.*`
        is still populated by then, so an optional read would quietly
        succeed and fit the deliverable to renders taken along the
        bootstrap's circular orbit, two denoise passes and an upscale ago.
        The direct file's one training reads them, and may: its cap is
        rendered after its upscale, along the cameras it trains on.
        """
        # brush's input name -> the output name select_support_views
        # publishes it under.
        support = {"support_images": "images", "support_masks": "masks",
                   "support_cameras": "cameras"}
        for path in _workflows():
            spec = WorkflowSpec.from_yaml(str(path))
            by_id = {s.id: s for s in spec.steps}
            order = {s.id: i for i, s in enumerate(spec.steps)}
            trainings = [s for s in spec.steps if s.step == "brush"]
            if not trainings:
                continue
            with self.subTest(workflow=path.name):
                readers = []
                photo_steps = {s.id: s for s in spec.steps if s.step == "photo_priority_weights"}
                photo_outputs = {v + "?" for ps in photo_steps.values()
                                 for k, v in ps.outputs.items() if k.startswith("support_")}
                for step in trainings:
                    wired = [step.inputs.get(name, "") for name in support]
                    if not any(wired):
                        continue
                    # The photograph's masked copies (photo_priority_weights, 2026-09-19)
                    # are made by a step of this training's own dataset, after every
                    # replacer; a training that reads ONLY those is not a cap reader.
                    if all(v in photo_outputs for v in wired if v):
                        producers = [ps for ps in photo_steps.values()
                                     if any(v + "?" in wired for v in ps.outputs.values())]
                        for ps in producers:
                            self.assertLess(order[ps.id], order[step.id])
                            replaced = [s.id for s in spec.steps if order[ps.id] < order[s.id] < order[step.id]
                                        and ("dataset.cameras" in s.outputs.values() or "dataset.images" in s.outputs.values())]
                            self.assertFalse(replaced, f"{path.name}: {replaced} replaced the dataset after '{ps.id}' made the photograph's copies")
                        continue
                    readers.append(step)
                    for name, value in zip(support, wired):
                        self.assertTrue(
                            value.endswith("?"),
                            f"{path.name}: '{step.id}' must read '{name}' "
                            f"optionally — the branch that writes it is gated",
                        )
                    # Fresh: nothing between the cap render and this
                    # training rebuilt the path or resized the frames.
                    cap_at = order.get("render_face_support_views")
                    self.assertIsNotNone(cap_at, f"{path.name}: no cap render")
                    stale = [
                        s.id for s in spec.steps
                        if cap_at < order[s.id] < order[step.id] and (
                            (s.step in _SPLAT_RENDERS and s.params.get("pattern")
                             and "dataset.cameras" in s.outputs.values())
                            or (s.step in ("seedvr2", "resize_batch")
                                and "dataset.images" in s.outputs.values()))
                    ]
                    self.assertFalse(
                        stale,
                        f"{path.name}: '{step.id}' reads the face cap, but "
                        f"{stale} replaced the dataset after the cap was "
                        f"rendered — the views are renders along a path and "
                        f"at a size the training no longer uses",
                    )
                self.assertLessEqual(
                    len(readers), 1,
                    f"{path.name}: the face cap is one training's evidence",
                )
                if "face_support_views" not in by_id:
                    continue
                self.assertEqual(len(readers), 1)
                training = readers[0]
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
                # 60 = 60 since 2026-09-16: the refined cap sits on the body
                # model's surface (no open rim to flare) and each support
                # view is rendered with the head's occlusion (`cull_mesh`),
                # which is what lets the disc reach the composite's angle.
                self.assertLessEqual(
                    float(cap.params["cap_radius_deg"]),
                    float(drawing.params["splat_max_angle_deg"]),
                    f"{path.name}: the supporting views must not be sampled from a "
                    f"wider band than the drawings are composited over",
                )
                self.assertEqual(cap.inputs.get("cull_mesh"), "scene.mesh_world?",
                                 f"{path.name}: a 60-degree cap of a surface cap has to be "
                                 f"rendered with the body's occlusion")
                # The inner edge of the band is the denoising path, so the
                # step has to be handed the cameras the training's own
                # frames were rendered and denoised along — and a pivot to
                # measure the angle to them about. Wire neither and the
                # edge silently does not apply, which is the failure mode
                # worth a test: nothing raises, and the deliverable is
                # fitted to renders of views it already has photographs of.
                self.assertEqual(
                    selector.inputs.get("path_cameras"), training.inputs["cameras"],
                    f"{path.name}: face_support_views must measure against the "
                    f"same cameras '{training.id}' trains on",
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
                for step in readers:
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

    def test_the_frames_yield_to_the_face_cap_in_the_training_that_reads_it(self):
        """The face cap wins: `face_priority` turns the refined face splat's
        coverage into per-pixel loss weights for the training views. All
        wiring, and all of it silent if wrong: an optional read of a path
        nothing writes trains at full weight, the old behaviour, with
        nothing in the log to say so.

        The weights and the cap travel together: the training that reads
        the supporting views reads the weights (or it hears the cap and the
        frames at the same volume over the face), and one that reads no
        cap reads no weights (there is nothing for its frames to yield to).
        The stage-2 training is that training.

        (The stage-1 shells had a `face_priority_shells` twin folding the
        same yield into their masks; both went with the shells on
        2026-09-06.)
        """
        for path in _workflows():
            spec = WorkflowSpec.from_yaml(str(path))
            by_id = {s.id: s for s in spec.steps}
            if "face_priority" not in by_id:
                continue
            with self.subTest(workflow=path.name):
                priority = by_id["face_priority"]
                self.assertEqual(priority.step, "face_priority_weights")
                self.assertEqual(priority.when, "${globals.face_splat}")
                # The training views' own cameras, and the REFINED splat —
                # the one the cap was rendered from, rebuilt through the
                # refined anchor camera. Weighting the frames by the
                # bootstrap splat's coverage would put the yield where the
                # face was before refine_cameras moved the anchor.
                self.assertEqual(priority.inputs["cameras"], "dataset.cameras")
                refined = by_id["face_splat_refined"]
                self.assertEqual(priority.inputs["splat_path"],
                                 refined.outputs["splat_path"])
                # The rebuild hangs the photograph's rays on the anchor's
                # refinement DELTA, which needs the anchor as the path built
                # it: render's image_warp camera, the one the frame was
                # warped for. Hung on the refined camera alone the cap turns
                # by the path's look_at tilt (run 5e2817, 28 mm).
                self.assertEqual(refined.inputs.get("given_camera"),
                                 "scene.image_warp.camera")
                self.assertEqual(refined.inputs.get("cameras"), "dataset.cameras")
                # The refined cap's shape is the body model's head
                # (FACE_GUIDE.md): the one the training reproduces and the
                # mesh fuses. The bootstrap's cap keeps the pointmap.
                self.assertEqual(refined.params.get("depth_prior"), "mesh_surface")
                self.assertIsNone(by_id["face_splat"].params.get("depth_prior"))
                self.assertEqual(priority.inputs["anchor_cameras"], "dataset.cameras")
                # Its coverage is the cap as the head lets each frame see
                # it, the same cull the support renders get.
                self.assertEqual(priority.inputs.get("mesh_world"), "scene.mesh_world?")
                selector = by_id["face_support_views"]
                self.assertEqual(priority.inputs["splat_center"],
                                 selector.inputs["splat_center"])
                # With the frames silenced over the face, the cap views on
                # the denoising path are no longer redundant with them.
                self.assertEqual(selector.params.get("min_path_angle_deg"), 0.0)

                # The cap's weights do not reach the trainings directly:
                # `photo_priority` (2026-09-19) multiplies its own yield in
                # — the frames fade wherever the photograph sees the body —
                # and publishes the stacked map. It is ungated (strength 0
                # passes the cap's weights through), reads the cap's output
                # and the dataset's own cameras, mattes and anchor, and its
                # strength is the `photo_priority` setting.
                photo = by_id["photo_priority"]
                self.assertEqual(photo.step, "photo_priority_weights")
                self.assertIs(photo.when, True)  # ungated: strength 0 is the off switch
                self.assertEqual(photo.inputs["weights"], priority.outputs["weights"] + "?")
                self.assertEqual(photo.inputs["cameras"], "dataset.cameras")
                self.assertEqual(photo.inputs["anchor_cameras"], "dataset.cameras")
                self.assertEqual(photo.inputs["alphas"], "dataset.masks")
                self.assertEqual(photo.inputs.get("mesh_world"), "scene.mesh_world?")
                self.assertEqual(photo.params.get("strength"), "${globals.photo_priority}")
                self.assertGreater(spec.steps.index(photo), spec.steps.index(priority))
                # Its masked copies of the photograph reach the training through
                # merge_support_views' second triple, and it runs before that merge.
                self.assertEqual(photo.inputs["images"], "dataset.images")
                merge = by_id["merge_support_views"]
                for name in ("images", "masks", "cameras"):
                    self.assertEqual(merge.inputs[f"b_{name}"], photo.outputs[f"support_{name}"] + "?")
                self.assertGreater(spec.steps.index(merge), spec.steps.index(photo))

                weights = photo.outputs["weights"] + "?"
                photo_steps = [s for s in spec.steps if s.step == "photo_priority_weights"]
                for step in spec.steps:
                    if step.step != "brush":
                        continue
                    if step.inputs.get("support_images"):
                        # The nearest photo_priority step before this training is the
                        # one whose stacked weights (and copies) it must read.
                        before = [ps for ps in photo_steps if spec.steps.index(ps) < spec.steps.index(step)]
                        self.assertTrue(before, f"'{step.id}' reads supporting views with no photo_priority before it")
                        expected = before[-1].outputs["weights"] + "?"
                        if step.id == "train_splat":
                            self.assertEqual(expected, weights)
                        else:
                            for name in ("images", "masks", "cameras"):
                                self.assertEqual(step.inputs.get(f"support_{name}"), before[-1].outputs[f"support_{name}"] + "?")
                        self.assertEqual(
                            step.inputs.get("weights"), expected,
                            f"'{step.id}' reads supporting views and must read "
                            f"the weights that make the frames yield to them",
                        )
                    else:
                        self.assertNotIn(
                            "weights", step.inputs,
                            f"'{step.id}' takes no supporting views, so there "
                            f"is nothing for its frames to yield to",
                        )
                if "export_colmap_intermediate" in by_id:
                    self.assertEqual(
                        by_id["export_colmap_intermediate"].inputs.get("weights"),
                        weights,
                        "the debug export must train at the weighting train_splat did",
                    )

    def test_the_cameras_are_refined_before_anything_reads_them(self):
        """Camera refinement lands ahead of every consumer of a pose.

        Three orderings, and each one is a silent failure rather than a
        loud one if it slips.

        `pointmap_elevation_views` is the sharp one, and the check stays
        for it though no workflow places one today: it puts a Gaussian
        shell on each frame's own camera ray and renders it from a camera
        derived from that pose, so a refinement running after it leaves the
        supporting views built on poses the training no longer uses — they
        argue with the frames instead of supporting them, and nothing
        raises.

        Both `brush` trainings come next: each one is roughly an hour of
        GPU fitted to whatever poses reach it, and the second trains on a
        dataset `render_subject` replaced wholesale, so one refinement
        cannot cover both.

        And the refinement never reads a supporting view. Those are renders
        made FROM the poses being corrected; feeding them back in would
        vote for the answer already in hand.
        """
        refine_inputs = ("cameras", "image_names", "images", "masks")
        # The re-outline branch's training is the one brush that takes no
        # solve of its own, on purpose: it is fitted on the orbit's ideal
        # cameras and its coverage rendered straight back onto them
        # (docs/re-outline.md). A refinement for it could not publish to
        # dataset.cameras — those stay the ideal orbit the re-render and
        # the main solve read — and what it would correct is
        # per-frame jitter with the common mode removed, a sharpness lever
        # for a splat kept only for its coverage. Unmeasured either way, so
        # left out rather than added on principle.
        unrefined = {"reoutline_train_splat"}
        for path in _workflows():
            spec = WorkflowSpec.from_yaml(str(path))
            ids = [s.id for s in spec.steps]
            refiners = [s for s in spec.steps if s.step == "refine_cameras"]
            trainings = [s for s in spec.steps
                         if s.step == "brush" and s.id not in unrefined]
            if not trainings:
                continue
            with self.subTest(workflow=path.name):
                self.assertEqual(
                    len(refiners), len(trainings),
                    f"{path.name}: {len(trainings)} brush training(s) but "
                    f"{len(refiners)} refinement(s) — each training fits a "
                    f"different dataset, so each needs its own solve",
                )
                for step in refiners:
                    self.assertEqual(step.outputs.get("cameras"), "dataset.cameras")
                    # Trap 6: the gauge is chosen from the subject, which
                    # needs the body mesh the cameras were drawn around.
                    # Without it the step falls back to the orbit's gauge
                    # and a subject that slid along the anchor's ray stays
                    # slid — 3 cm on helical-splat_00307.
                    self.assertEqual(
                        step.inputs.get("mesh_world"), "scene.mesh_world?",
                        f"{path.name}: '{step.id}' must wire the body mesh so the "
                        f"refined cameras carry the subject's gauge",
                    )
                    for name in refine_inputs:
                        wired = step.inputs.get(name, "")
                        self.assertTrue(
                            wired.startswith("dataset."),
                            f"{path.name}: '{step.id}' must refine against the "
                            f"dataset's own training views, not '{wired}'",
                        )
                    self.assertNotIn(
                        "support", " ".join(step.inputs.values()),
                        f"{path.name}: '{step.id}' must not read a supporting "
                        f"view — they are renders made from the poses it is "
                        f"correcting",
                    )

                # Every step that reads dataset.cameras for geometry has a
                # refinement in front of it.
                for consumer in [s for s in spec.steps
                                 if s.step in ("brush", "pointmap_elevation_views")
                                 and s.id not in unrefined]:
                    earlier = [s for s in refiners
                               if ids.index(s.id) < ids.index(consumer.id)]
                    self.assertTrue(
                        earlier,
                        f"{path.name}: '{consumer.id}' reads camera poses with "
                        f"no refinement ahead of it",
                    )

                # ...and each refinement reads the mattes computed for the
                # frames it is about, not a stale batch. The failure this
                # rules out is masking a solve to the all-1.0 VACE control
                # batch, which is the unmasked variant under another name.
                producers = {}
                for step in spec.steps:
                    for field, ctx in step.outputs.items():
                        producers[ctx] = step
                for step in refiners:
                    source = producers.get(step.inputs["masks"])
                    self.assertIsNotNone(source)
                    self.assertEqual(
                        source.step, "rmbg",
                        f"{path.name}: '{step.id}' masks its features with "
                        f"'{source.id}' ({source.step}), which is not a "
                        f"foreground matte",
                    )

    def test_supporting_views_are_built_after_the_refinement_that_moves_their_poses(self):
        """A render made from a pose the refinement then moves is stale.

        The face cap is the case: its splat is unprojected from the
        reference photograph through the ANCHOR camera, so `refine_cameras`
        moving that pose leaves anything built on it describing a world
        nothing else believes in — `brush` is then handed a face off the
        head, weighted by the face's own alpha, and nothing raises. From
        2026-08-31 the cap's cameras were carried across by the anchor's own
        pose delta (`rebase_cameras`); measured on 2026-09-02 that put the
        face 50 mm inside the head, because the delta is mostly a slide
        along the anchor's viewing ray and the splat's depth was the mesh's,
        not the camera's, to move. So the rule is now about ORDER, with no
        carry: every supporting-view camera path `merge_support_views` reads
        must be WRITTEN after the last refinement ahead of it, and the splat
        the face cap renders must itself be built after that refinement,
        through the refined poses (`cameras: dataset.cameras`). The stage-1
        body shells have always had this wiring.
        """
        for path in _workflows():
            spec = WorkflowSpec.from_yaml(str(path))
            order = {step.id: i for i, step in enumerate(spec.steps)}
            by_id = {step.id: step for step in spec.steps}
            merges = [s for s in spec.steps if s.step == "merge_support_views"]
            refiners = [s for s in spec.steps if s.step == "refine_cameras"]
            if not merges or not refiners:
                continue
            # LAST writer: the step whose output the reader actually sees.
            producers: dict = {}
            for step in spec.steps:
                for ctx in step.outputs.values():
                    producers[ctx] = step
            with self.subTest(workflow=path.name):
                self.assertFalse(
                    [s for s in spec.steps if s.step == "rebase_cameras"],
                    f"{path.name}: a rebase_cameras step is back; the carry "
                    f"was measured wrong (2026-09-02) and the step deleted",
                )
                for merge in merges:
                    refine = max(
                        (s for s in refiners if order[s.id] < order[merge.id]),
                        key=lambda s: order[s.id], default=None)
                    self.assertIsNotNone(
                        refine,
                        f"{path.name}: '{merge.id}' gathers supporting views "
                        f"with no refinement ahead of it",
                    )
                    for name in ("a_cameras", "b_cameras"):
                        wired = merge.inputs.get(name, "").rstrip("?")
                        source = producers.get(wired)
                        if source is None:
                            continue
                        self.assertGreater(
                            order[source.id], order[refine.id],
                            f"{path.name}: '{wired}' is written by "
                            f"'{source.id}' before '{refine.id}' moves the "
                            f"poses it was rendered from, and nothing rebuilds "
                            f"it before '{merge.id}' reads it",
                        )
                        # Walk the views back to the splat they render, and
                        # check IT was built on the refined poses too.
                        step = source
                        seen = set()
                        while step is not None and step.id not in seen:
                            seen.add(step.id)
                            if step.step in ("face_pointmap_splat", "pointmap_splat"):
                                self.assertGreater(
                                    order[step.id], order[refine.id],
                                    f"{path.name}: '{step.id}' builds the splat "
                                    f"'{source.id}' renders BEFORE '{refine.id}' "
                                    f"moves the camera it is unprojected through",
                                )
                                self.assertEqual(
                                    step.inputs.get("cameras"), "dataset.cameras",
                                    f"{path.name}: '{step.id}' must unproject "
                                    f"through the REFINED anchor camera",
                                )
                                break
                            upstream = None
                            for field in ("splat_path", "images", "cameras"):
                                ctx = step.inputs.get(field, "").rstrip("?")
                                if ctx in producers and producers[ctx].id != step.id:
                                    upstream = producers[ctx]
                                    break
                            step = upstream

    def test_the_first_refinement_keeps_the_recorded_anchor_in_step(self):
        """`render_initial_views` records the GIVEN anchor's position in the
        extras; `refine_cameras` moves that camera. The first refinement
        must republish the refined position over the record, reading the
        anchor's index from the same extras — otherwise `inject_anchor` and
        `render_splat`'s cap fallback match against a pose nobody holds."""
        for path in _workflows():
            spec = WorkflowSpec.from_yaml(str(path))
            refiners = [s for s in spec.steps if s.step == "refine_cameras"]
            if not refiners:
                continue
            first = refiners[0]
            with self.subTest(workflow=path.name):
                self.assertEqual(first.inputs.get("anchor_frame_index"),
                                 "dataset.extras.anchor_frame_index?")
                self.assertEqual(first.outputs.get("anchor_position"),
                                 "dataset.extras.anchor_position")

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


    def test_a_splat_render_marks_the_splat_as_real_content(self):
        """`splat_inactive_mask` is ON in the workflow, though the step
        declares it off.

        The step's default has to stay off — nothing downstream is obliged
        to consume the batch, and a `+splat` render that produced it
        unasked would be publishing an output for nobody. In THIS workflow
        it has a reader (`reinject_anchor_initial` passes it through to the
        denoise) and 2026-09-08's sweep measured it worth having, so every
        reference run in docs/vace-denoise-findings-2026-09-07.md carries
        it. It spent a week as a per-run `--param`, which is the wrong
        place for something that works: forgetting it costs a whole pod
        run, and the run looks fine.

        Asserted on every render that composites a splat, so a new one
        cannot be added without the batch — that is the shape that
        silently reverts.
        """
        for path in _workflows():
            spec = WorkflowSpec.from_yaml(str(path))
            for step in spec.steps:
                mode = str(step.params.get("render_mode", ""))
                if step.step != "render" or "splat" not in mode:
                    continue
                with self.subTest(workflow=path.name, step=step.id):
                    self.assertIs(
                        step.params.get("splat_inactive_mask"), True,
                        f"'{step.id}' composites a splat ({mode}) but does not "
                        f"set splat_inactive_mask, so the denoise is not told "
                        f"which pixels are already a real photograph",
                    )

    def test_denoise_prompts_are_the_pinned_ones_in_every_pass(self):
        """See DENOISE_PROMPT's comment: the way this breaks is whitespace,
        so compare the whole string rather than eyeballing the YAML. (It
        no longer checks the ComfyUI graph — the positive prompt diverged
        from it deliberately — but every copy must still be one of the two
        pinned strings.)

        The prompts are now a param of each wan22_vace_denoise step, not a
        workflow global — and each workflow has THREE denoise passes carrying
        their own copy since 2026-09-08: the two full-resolution passes and
        the gated 480p re-outline pass between the bootstrap and pass 1,
        which must read as pass 1 does or its silhouette is of a different
        subject. Since 2026-09-20 the two passes on a DRAWING (pass 1 and
        the re-outline pass) carry ORBIT_SENTENCE and pass 2 does not. So
        this checks every one.
        """
        # How many wan22_vace_denoise steps each file carries: the two
        # full-resolution passes plus the gated 480p re-outline pass.
        expected = {"helical.yaml": 3, "helical_shell.yaml": 3}
        drawn = {"denoise_pass1", "reoutline_denoise"}
        # The experiment's pass 1 and re-outline pass carry the hint
        # appended (HINTED_DENOISE_PROMPT); its pass 2 does not.
        hinted = {("helical_shell.yaml", "denoise_pass1"),
                  ("helical_shell.yaml", "reoutline_denoise")}
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
                    want = (HINTED_DENOISE_PROMPT if (path.name, step.id) in hinted
                            else DRAWING_PROMPT if step.id in drawn
                            else DENOISE_PROMPT)
                    self.assertEqual(step.params.get("prompt"), want)
                    self.assertEqual(
                        step.params.get("negative_prompt"), DENOISE_NEGATIVE_PROMPT
                    )
            self.assertEqual(
                passes, expected[path.name],
                f"{path.name}: expected {expected[path.name]} denoise passes")
        self.assertEqual(seen, sum(expected[p.name] for p in workflows))

    def test_bootstrap_prologues_have_the_pinned_shape(self):
        """BOOTSTRAPS pins how each file manufactures the dataset the tail
        expects; a prologue that changed shape is the edit somebody should
        look at. Every shipped file is pinned, and every pinned file
        ships."""
        names = {p.stem for p in _workflows()}
        self.assertEqual(names, set(BOOTSTRAPS))
        for name, prologue in BOOTSTRAPS.items():
            spec = WorkflowSpec.from_yaml(str(WORKFLOW_DIR / f"{name}.yaml"))
            with self.subTest(workflow=name):
                self.assertEqual([s.id for s in spec.steps[: len(prologue)]], prologue)

    def test_helical_shell_mirrors_helical(self):
        """helical_shell.yaml is helical.yaml with five shell steps added
        and a short list of deliberate differences. There is no include
        mechanism, so the thing worth checking is that nothing ELSE has
        drifted between the two: an A/B against helical.yaml is only an A/B
        if the tail is the same tail.

        Compared, step by step after SHELL_ONLY_STEPS are removed: id,
        class, dispatch, env, inputs, outputs, when, keep_loaded and the raw
        params block — except where this test says the file differs:

          * both mesh renders draw the helix: `pattern`, `n_loops`,
            `amplitude_deg`, `lead_in_deg`, `lead_out_deg`, `helix_anchor`
            replace helical's `pattern: circular`; every other param on
            those steps is the same;
          * pass 1 and the re-outline pass carry the prompt hint
            (HINTED_DENOISE_PROMPT, pinned above); nothing else on them
            differs.

        Globals: helical's, all of them, at the same values, plus the
        experiment's own settings and a per-file output_root.
        """
        base = WorkflowSpec.from_yaml(str(WORKFLOW_DIR / "helical.yaml"))
        shell = WorkflowSpec.from_yaml(str(WORKFLOW_DIR / "helical_shell.yaml"))
        for name in SHELL_ONLY_STEPS:
            self.assertIn(name, [s.id for s in shell.steps])
            self.assertNotIn(name, [s.id for s in base.steps])
        mirrored = [s for s in shell.steps if s.id not in SHELL_ONLY_STEPS]
        self.assertEqual([s.id for s in mirrored], [s.id for s in base.steps],
                         "helical_shell's steps have drifted from helical's")

        helix_keys = {"pattern", "n_loops", "amplitude_deg", "lead_in_deg",
                      "lead_out_deg", "helix_anchor"}
        renders = {"render_initial_views", "render_reoutlined_views"}
        hinted = {"denoise_pass1", "reoutline_denoise"}
        for step, twin in zip(mirrored, base.steps):
            with self.subTest(step=step.id):
                self.assertEqual(step.step, twin.step)
                self.assertEqual(step.dispatch, twin.dispatch)
                self.assertEqual(step.env, twin.env)
                self.assertEqual(step.inputs, twin.inputs)
                self.assertEqual(step.outputs, twin.outputs)
                self.assertEqual(step.when, twin.when)
                self.assertEqual(step.keep_loaded, twin.keep_loaded)
                params, base_params = dict(step.params), dict(twin.params)
                if step.id in renders:
                    self.assertEqual(base_params.pop("pattern"), "circular")
                    self.assertEqual(
                        {k: params.pop(k) for k in helix_keys},
                        {"pattern": "helical", "n_loops": 1,
                         "amplitude_deg": "${globals.helix_amplitude_deg}",
                         "lead_in_deg": 0.0, "lead_out_deg": 4.5,
                         "helix_anchor": "start"},
                    )
                if step.id in hinted:
                    self.assertEqual(params.pop("prompt"), HINTED_DENOISE_PROMPT)
                    self.assertEqual(base_params.pop("prompt"), DRAWING_PROMPT)
                self.assertEqual(params, base_params)

        own = {"helix_amplitude_deg", "shell_tail_frames", "shell_head_frames",
               "shell_reference", "elevation_hint"}
        self.assertEqual(set(shell.globals) - set(base.globals), own)
        for key, value in base.globals.items():
            with self.subTest(glob=key):
                if key == "output_root":
                    self.assertNotEqual(shell.globals[key], value)
                    continue
                self.assertEqual(shell.globals.get(key), value,
                                 f"global '{key}' differs between the two files")
        # The hint is what makes the two prompts differ; blank, they are
        # the same string — which is the arm without.
        self.assertTrue(shell.globals["elevation_hint"].strip())
        self.assertEqual(
            resolve(HINTED_DENOISE_PROMPT, {"globals": {"elevation_hint": ""}}),
            DRAWING_PROMPT + " ",
        )

    def test_every_render_with_a_backdrop_draws_the_same_room(self):
        """The backdrop was a pipeline SETTING until 2026-09-01, which made
        this true by construction; it is now written per step, so it is a
        thing to keep by hand and worth pinning.

        Why it has to hold: the renders feeding the two denoise passes share
        a batch, so a subject whose room changes between
        them is a subject that has been teleported. A render that must NOT
        have a backdrop states that with `background: ""` and is exempt here
        — `select_support_views` divides the alpha back out of its frames
        and would recover the room as the face's own colour.

        No render drawing one at all satisfies this the other way, and is
        what helical has done since 2026-09-05: the grid came
        off the first denoise's frames on 09-04 and off the re-render the
        day after, so the passes agree on emptiness. The invariant is that
        the rooms agree, not that there is a room.
        """
        for path in _workflows():
            spec = WorkflowSpec.from_yaml(str(path))
            keys = ("background", "background_base_color",
                    "background_line_color", "background_params")
            rooms = {
                step.id: tuple(step.params.get(key) for key in keys)
                for step in spec.steps
                if step.params.get("background")
            }
            distinct = {repr(room) for room in rooms.values()}
            self.assertLessEqual(
                len(distinct), 1,
                f"{path.name}: renders disagree about the room — {rooms}",
            )

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


    def test_the_whole_face_branch_reads_the_unresized_front_half(self):
        """Every full-frame step in the face branch must read the SAME array.

        The face splat is placed by re-expressing SAM-3D-Body's camera on
        the crop's pixel grid, and that camera's focal length is in the
        pixels of the image `sam3d_body` was handed. Nothing downstream can
        notice if some of those steps are given a resized copy instead: a
        crop of a half-size photograph is still a crop, still native
        resolution, still self-consistent — and every Gaussian lands on a
        ray computed from a focal belonging to the other grid, which puts
        the whole face off the head by a scale factor.

        There is a runtime guard for it now
        (`FacePointmapSplatStep._source_intrinsics` against `image_size`),
        but the wiring is where the mistake would be made, so it is checked
        here too — including the pair nothing else could relate: the frame
        `crop_to_box` cuts and the frame the box it is cutting was found in.
        """
        full_frame_readers = {
            "detect_face_landmarks": "image",   # landmarks, normalized to it
            "map_face_to_mesh": "image",        # the mesh's head, rendered on it
            "fit_head_to_face": "image",        # landmarks projected into it
            "crop_to_box": "image",             # the frame the crop is cut from
        }
        for path in _workflows():
            spec = WorkflowSpec.from_yaml(str(path))
            splits = [s for s in spec.steps if s.step == "split_reference_sheet"]
            if not splits:
                continue
            front = splits[0].outputs["front"]

            with self.subTest(workflow=path.name):
                for step in spec.steps:
                    key = full_frame_readers.get(step.step)
                    if key is None:
                        continue
                    self.assertEqual(
                        step.inputs.get(key), front,
                        f"{path.name}: '{step.id}' reads "
                        f"'{step.inputs.get(key)}' where it needs the one "
                        f"un-resized front half ('{front}')",
                    )

                # The crop and its box have to be in the same pixels.
                produces = {out: s for s in spec.steps
                            for out in s.outputs.values()}
                for step in spec.steps:
                    if step.step != "crop_to_box":
                        continue
                    locator = produces.get(step.inputs["box"])
                    self.assertIsNotNone(
                        locator,
                        f"{path.name}: '{step.id}' crops to a box no step "
                        f"produces ({step.inputs['box']})",
                    )
                    self.assertEqual(
                        locator.inputs.get("image"), step.inputs["image"],
                        f"{path.name}: '{step.id}' cuts "
                        f"'{step.inputs['image']}' with a box '{locator.id}' "
                        f"found in '{locator.inputs.get('image')}'",
                    )

                # And the guard has something to compare against: the size
                # sam3d_body fitted on has to reach the splat's mesh_output.
                splats = [s for s in spec.steps
                          if s.step == "face_pointmap_splat"]
                if not splats:
                    continue
                mesh = splats[0].inputs["mesh_output"]
                sizes = [s.outputs.get("image_size")
                         for s in spec.steps if s.step == "sam3d_body"]
                self.assertTrue(
                    any(size and size.startswith(f"{mesh}.") for size in sizes),
                    f"{path.name}: sam3d_body does not publish image_size "
                    f"into '{mesh}', so face_pointmap_splat cannot tell the "
                    f"photograph from a resized copy of it",
                )


if __name__ == "__main__":
    unittest.main()


class TestTheIntermediateSplatIsKept(unittest.TestCase):
    """The first brush training exports somewhere the result .zip carries.

    That splat is what the helical re-render is built from, so it is the
    first thing to look at when the re-render is wrong — and it is only
    reachable afterwards if it lands under the run's `debug/`, which is
    what `runs.DEBUG_SUBDIRS` packages. An `output_dir` here instead would
    put it in `brush/training_<ms>/`: on the volume, and gone with the pod.
    """

    def test_the_first_training_exports_into_debug(self):
        from pipeline.cli import resolve_workflow
        from pipeline.templating import resolve
        from pipeline.workflow import WorkflowSpec

        spec = WorkflowSpec.from_yaml(resolve_workflow("helical"))
        scope = {"globals": dict(spec.globals, output_root="/out")}
        trainings = {
            step.id: resolve(step.params, scope)
            for step in spec.steps if step.step == "brush"
        }
        self.assertIn("train_splat", trainings)
        intermediate = trainings["train_splat"]
        self.assertEqual(intermediate.get("export_dir"), "/out/debug")
        self.assertTrue(str(intermediate.get("export_name", "")).endswith(".ply"))
        # `output_dir` would win nothing here (export_dir takes precedence),
        # but leaving both set would be a contradiction to read later.
        self.assertIsNone(intermediate.get("output_dir"))

    def test_the_two_trainings_cannot_collide(self):
        from pipeline.cli import resolve_workflow
        from pipeline.templating import resolve
        from pipeline.workflow import WorkflowSpec

        spec = WorkflowSpec.from_yaml(resolve_workflow("helical"))
        scope = {"globals": dict(spec.globals, output_root="/out")}
        exports = [
            (resolve(step.params, scope).get("export_dir"),
             resolve(step.params, scope).get("export_name"))
            for step in spec.steps if step.step == "brush"
        ]
        self.assertEqual(len(exports), len(set(exports)), f"two trainings share a path: {exports}")


class TestTheFirstDenoiseInputIsKept(unittest.TestCase):
    """The control video pass 1 is handed lands in the debug bundle.

    Everything else a run exports describes frames from AFTER a denoise —
    `colmap_intermediate/` is pass 1's output, `colmap/` is pass 2's. The
    drawings that caused them lived in memory only, so "was the skeleton
    too much ink", "did the face splat land on the face" and "does the
    anchor frame carry the photograph" could be asked of a finished run
    only by re-running the first fourteen steps from the same upload — and
    not at all once the upload was gone.
    """

    def _spec(self):
        from pipeline.cli import resolve_workflow
        from pipeline.workflow import WorkflowSpec

        return WorkflowSpec.from_yaml(resolve_workflow("helical"))

    def test_the_dataset_is_saved_under_debug(self):
        from pipeline.templating import resolve

        spec = self._spec()
        scope = {"globals": dict(spec.globals, output_root="/out")}
        saves = {
            step.id: resolve(step.params, scope)["directory"]
            for step in spec.steps if step.step == "save_dataset"
        }
        self.assertIn("dump_denoise_input", saves)
        self.assertEqual(saves["dump_denoise_input"], "/out/debug/denoise_pass1_input")
        for step_id, directory in saves.items():
            self.assertTrue(
                directory.startswith("/out/debug/"),
                f"{step_id} writes to {directory!r}, which the result .zip never sees")
        self.assertEqual(len(set(saves.values())), len(saves))

    def test_it_saves_what_the_denoise_reads(self):
        """After the anchor injection, before pass 1 — so the frames on disk
        carry the warped photograph and the 0.0 VACE mask at the anchor,
        which is the half of the input a mesh render cannot be re-derived
        into."""
        spec = self._spec()
        order = [step.id for step in spec.steps]
        self.assertLess(order.index("reinject_anchor_initial"), order.index("dump_denoise_input"))
        self.assertLess(order.index("dump_denoise_input"), order.index("denoise_pass1"))
        dump = next(s for s in spec.steps if s.id == "dump_denoise_input")
        denoise = next(s for s in spec.steps if s.id == "denoise_pass1")
        self.assertEqual(dump.inputs["dataset"], "dataset")
        # The frames and their masks are what the dump is for; both reach
        # the denoise from the same dataset the dump wrote.
        self.assertEqual(denoise.inputs["control_video"], "dataset.images")
        self.assertEqual(denoise.inputs["control_masks"], "dataset.masks")


class TestOnlyTheHelicalRerenderCapsTheShBands(unittest.TestCase):
    """`render_subject` is the one render_splat that sets `sh_degree`, and
    it sets 2. Every other instance leaves the step's default (3, every
    band the splat carries) alone — pinned as a workflow decision rather
    than as a step default, because a `sh_degree:` line quietly added to
    the face cap's render would change what the final training is
    supervised by."""

    def test_render_subject_says_two_and_nothing_else_says_anything(self):
        from pipeline.cli import resolve_workflow
        from pipeline.workflow import WorkflowSpec

        spec = WorkflowSpec.from_yaml(resolve_workflow("helical"))
        renders = [s for s in spec.steps if s.step in _SPLAT_RENDERS]
        self.assertGreater(len(renders), 1)
        setting = {s.id: s.params.get("sh_degree") for s in renders}
        self.assertEqual(setting.pop("render_subject"), 2)
        self.assertEqual(set(setting.values()), {None}, setting)


class TestTheSingleViewInput(unittest.TestCase):
    """A single frontal photo rides the same workflow as a sheet: the split
    step decides which it is (`input_layout`, auto by default) and
    publishes the verdict, and `pick_rear_view` reads that verdict after
    pass 1 to fill the reference slot the photo could not. Neither is
    gated — a `when:` resolves against globals before the run, so a
    runtime verdict cannot switch a step off — and the wiring below is what
    makes the two modes share one file (steps/reference_view.py)."""

    def _spec(self):
        from pipeline.cli import resolve_workflow
        from pipeline.workflow import WorkflowSpec

        return WorkflowSpec.from_yaml(resolve_workflow("helical"))

    def _step(self, spec, step_id):
        return next(s for s in spec.steps if s.id == step_id)

    def test_the_split_reads_the_setting_and_publishes_its_verdict(self):
        spec = self._spec()
        setting = next(p for p in spec.settings if p.name == "input_layout")
        self.assertEqual(setting.default, "auto")
        self.assertEqual(list(setting.choices), ["auto", "sheet", "single"])
        split = self._step(spec, "split_sheet")
        self.assertEqual(split.params.get("layout"), "${globals.input_layout}")
        self.assertEqual(split.outputs.get("layout"), "scene.input_layout")

    def test_pick_rear_view_sits_after_pass_1_and_before_the_training(self):
        spec = self._spec()
        order = [s.id for s in spec.steps]
        pick = order.index("pick_rear_view")
        self.assertGreater(pick, order.index("foreground_masks"), "it needs the mattes")
        self.assertGreater(pick, order.index("adjust_denoised"), "the treated frames")
        self.assertLess(pick, order.index("train_splat"))
        self.assertLess(pick, order.index("denoise_pass2"))
        step = self._step(spec, "pick_rear_view")
        self.assertTrue(step.when is True, "runs in both modes; see the class docstring")
        self.assertEqual(step.inputs.get("layout"), "scene.input_layout")
        self.assertEqual(step.inputs.get("reference_image"), "dataset.reference_image")
        self.assertEqual(step.outputs.get("reference_image"), "dataset.reference_image")
        self.assertEqual(step.inputs.get("masks"), "dataset.masks")

    def test_the_reference_has_exactly_two_writers(self):
        """The split (the back panel, or None) and the pick (the rear view,
        or the back panel again). A third would be a second opinion on
        what pass 2 conditions on."""
        spec = self._spec()
        writers = [
            s.id for s in spec.steps
            if "dataset.reference_image" in s.outputs.values()
        ]
        self.assertEqual(writers, ["split_sheet", "pick_rear_view"])

    def test_pass_2_reads_what_the_pick_wrote(self):
        spec = self._spec()
        order = [s.id for s in spec.steps]
        for step_id in ("denoise_pass2",):
            step = self._step(spec, step_id)
            self.assertGreater(order.index(step_id), order.index("pick_rear_view"))
            self.assertEqual(step.inputs.get("reference_image"), "dataset.reference_image")
        # and pass 1 reads the slot BEFORE the pick, when a photo leaves it empty
        self.assertLess(order.index("denoise_pass1"), order.index("pick_rear_view"))


class TestThePixelOpsShipOff(unittest.TestCase):
    """`adjust_denoised` (pixel_ops) runs in the workflow, but every knob on
    it is off unless a setting turns it on. The SH cap on `render_subject`
    is aimed at part of what the highlight suppression was for — the
    specular sparkle a re-render carries into pass 2 — so the two are kept
    separable: the pixel op stays a choice, not a default."""

    def test_specular_suppress_is_zero_and_is_what_the_step_reads(self):
        from pipeline.cli import resolve_workflow
        from pipeline.workflow import WorkflowSpec

        spec = WorkflowSpec.from_yaml(resolve_workflow("helical"))
        setting = next(s for s in spec.settings if s.name == "specular_suppress")
        self.assertEqual(setting.default, 0.0)
        step = next(s for s in spec.steps if s.id == "adjust_denoised")
        self.assertEqual(step.step, "pixel_ops")
        self.assertEqual(step.params["specular_suppress"],
                         "${globals.specular_suppress}")


class TestTheDenoiseSettingsAreTheMeasuredOnes(unittest.TestCase):
    """The two passes ship the settings the 2026-09-08 sweeps settled on
    (docs/vace-denoise-findings-2026-09-07.md, sections 5 and 6), applied
    2026-09-09. Pinned because they are decisions with numbers behind them,
    and a step default drifting under them — the step's own `sampler_shift`
    is 8, which cost pass 2 three points of head sharpness — would be
    silent.
    """

    def _step(self, step_id):
        from pipeline.cli import resolve_workflow
        from pipeline.workflow import WorkflowSpec

        spec = WorkflowSpec.from_yaml(resolve_workflow("helical"))
        return next(s for s in spec.steps if s.id == step_id)

    def test_pass_1_is_run_e4_on_euler(self):
        """The skeleton-leak sweep's E4 (shift 5 erases the ink, the
        structure steps stay at full scale and the four detail steps sit at
        a flat half rather than a taper spent on steps 5-6) — with both
        experts on euler since 2026-09-20 rather than E4's uni_pc on the
        high-noise one: at shift 5 the sampler no longer mattered to the
        leak (E2 = E4), and helical-20260920-220458 ran this way. The low
        sampler is written out because the step's default is uni_pc."""
        params = self._step("denoise_pass1").params
        self.assertEqual((params["sampler_high"], params["sampler_low"]),
                         ("euler", "euler"))
        self.assertEqual(params["sampler_shift"], 5)
        self.assertEqual(params["strength"], [1, 1, 0.5, 0.5, 0.5, 0.5])
        self.assertEqual((params["steps_high"], params["steps_low"]), (2, 4))

    def test_pass_2_is_the_quality_sweep_s_corner_at_0_8(self):
        """Euler on the opening steps at shift 2.5 (shift 8 and uni_pc each
        cost ~3 points of head sharpness on a splat render), a flat 0.8 on
        every step (strength is a fidelity-vs-texture dial; 0.8 is the
        texture compromise, and a taper to 0 is invention), 2/4 kept."""
        params = self._step("denoise_pass2").params
        self.assertEqual((params["sampler_high"], params["sampler_low"]),
                         ("euler", "euler"))
        self.assertEqual(params["sampler_shift"], 2.5)
        self.assertEqual(params["strength"], [0.8] * 6)
        self.assertEqual((params["steps_high"], params["steps_low"]), (2, 4))


class TestTheReoutlineBranch(unittest.TestCase):
    """The experimental branch that redraws the silhouette from a splat of
    the subject (docs/re-outline.md): eight gated steps between the anchor
    injection and the first denoise. What is pinned is what makes it a
    faithful copy of pass 1 and of the first render — a different denoise
    would matte a different subject, a splat trained or rendered on other
    cameras would put silhouette i on the wrong frame — and that with the
    setting off nothing of it runs.
    """

    BRANCH = [
        "reoutline_downscale", "reoutline_denoise", "reoutline_matte",
        "reoutline_upscale", "reoutline_train_splat", "render_reoutline_splat",
        "render_reoutlined_views", "reinject_anchor_reoutlined",
    ]

    def _spec(self):
        from pipeline.cli import resolve_workflow

        return WorkflowSpec.from_yaml(resolve_workflow("helical"))

    def _step(self, spec, step_id):
        return next(s for s in spec.steps if s.id == step_id)

    def test_it_sits_between_the_anchor_injection_and_the_dump(self):
        spec = self._spec()
        order = [s.id for s in spec.steps]
        first = order.index("reinject_anchor_initial") + 1
        self.assertEqual(order[first:first + len(self.BRANCH)], self.BRANCH)
        self.assertEqual(order[first + len(self.BRANCH)], "dump_denoise_input")

    def test_every_step_is_gated_on_the_setting_and_it_defaults_on(self):
        spec = self._spec()
        setting = next(s for s in spec.settings if s.name == "re_outline")
        self.assertIs(setting.default, True)
        for step_id in self.BRANCH:
            with self.subTest(step=step_id):
                self.assertEqual(self._step(spec, step_id).when, "${globals.re_outline}")

    def test_the_extra_denoise_is_pass_1_at_480p_and_four_steps(self):
        """Pass 1's block, apart from the size, the step count (2 high /
        2 low: the pass is kept for its shape, the low expert's extra steps
        are texture) and the per-step list that count reshapes — the
        strength schedule over four entries."""
        spec = self._spec()
        extra = self._step(spec, "reoutline_denoise")
        pass1 = self._step(spec, "denoise_pass1")
        self.assertEqual((pass1.params["steps_high"], pass1.params["steps_low"]), (2, 4))
        self.assertEqual(pass1.params["strength"], [1, 1, 0.5, 0.5, 0.5, 0.5])
        expected = dict(pass1.params, width=480, height=832, steps_low=2,
                        strength=[1, 1, 0.5, 0.5])
        self.assertEqual(extra.params, expected)
        self.assertEqual((extra.dispatch, extra.env, extra.keep_loaded),
                         (pass1.dispatch, pass1.env, pass1.keep_loaded))
        self.assertEqual(extra.inputs["reference_image"], pass1.inputs["reference_image"])
        self.assertEqual(extra.inputs["subject_desc"], pass1.inputs["subject_desc"])
        # It reads the downscaled batch, not the dataset, and writes beside it.
        self.assertEqual(extra.inputs["control_video"], "scene.reoutline.images")
        self.assertEqual(extra.inputs["control_masks"], "scene.reoutline.masks")
        self.assertEqual(extra.outputs, {"images": "scene.reoutline.denoised"})

    def test_the_batch_is_resized_here_not_by_diffusers(self):
        """diffusers fits a control video under the target AREA (464x832 for
        a 720x1280 batch asked for 480x832) — see steps/resize.py."""
        spec = self._spec()
        down = self._step(spec, "reoutline_downscale")
        self.assertEqual(down.params, {"width": 480, "height": 832})
        self.assertEqual(down.inputs, {"images": "dataset.images", "masks": "dataset.masks"})
        up = self._step(spec, "reoutline_upscale")
        self.assertEqual(up.params, {"width": "${globals.resolution.0}",
                                     "height": "${globals.resolution.1}"})
        self.assertEqual(up.inputs, {"images": "scene.reoutline.denoised",
                                     "masks": "scene.reoutline.mattes"})
        self.assertEqual(up.outputs, {"images": "scene.reoutline.frames",
                                      "masks": "scene.reoutline.frame_mattes"})

    def test_the_splat_is_trained_on_the_orbit_cameras_from_the_upscaled_pass(self):
        """The outline is the splat's coverage, so the splat has to be fitted
        on the very cameras the orbit is re-rendered from, to frames at
        those cameras' size; and only to the 480p pass — none of the
        supporting views, weights, rig or mesh the intermediate takes,
        which would pull it toward something other than that pass."""
        spec = self._spec()
        train = self._step(spec, "reoutline_train_splat")
        self.assertEqual(train.step, "brush")
        self.assertEqual(train.inputs, {
            "cameras": "dataset.cameras",
            "image_names": "dataset.image_names",
            "points_3d": "dataset.points_3d",
            "images": "scene.reoutline.frames",
            "masks": "scene.reoutline.frame_mattes",
        })
        self.assertEqual(train.outputs, {"splat_path": "scene.reoutline.splat_path"})
        # The intermediate's silhouette knobs, no alignment, no polish, half
        # the iterations (the texture is thrown away), and the .ply in the
        # debug bundle beside intermediate_splat.ply.
        intermediate = self._step(spec, "train_splat")
        for key in ("polish_steps", "align_iters", "match_alpha_weight"):
            with self.subTest(param=key):
                self.assertEqual(train.params[key], intermediate.params[key])
        self.assertEqual(train.params["total_steps"], intermediate.params["total_steps"] // 2)
        self.assertEqual(train.params["export_dir"], intermediate.params["export_dir"])
        self.assertEqual(train.params["export_name"], "reoutline_splat.ply")
        self.assertNotEqual(train.params["export_name"], intermediate.params["export_name"])
        self.assertNotIn("hollow_weight", train.params)

    def test_the_outline_is_the_splat_rendered_on_the_dataset_cameras(self):
        """No pattern: render_splat reuses the dataset's cameras verbatim,
        which is what puts silhouette i on frame i. Only the alpha is
        published, at the render's size, so `render` accepts it."""
        spec = self._spec()
        render = self._step(spec, "render_reoutline_splat")
        self.assertEqual(render.step, "render_splat")
        self.assertEqual(render.inputs, {"splat_path": "scene.reoutline.splat_path",
                                         "dataset": "dataset"})
        self.assertNotIn("pattern", render.params)
        self.assertNotIn("override_cam_from_mesh", render.params)
        self.assertFalse(render.params.get("confidence", False))
        self.assertEqual((render.params["width"], render.params["height"]),
                         ("${globals.resolution.0}", "${globals.resolution.1}"))
        self.assertEqual(render.outputs, {"masks": "scene.outline_masks"})

    def test_the_re_render_is_the_first_render_plus_the_mattes(self):
        """Same params, so the same cameras; and it republishes nothing about
        them, so nothing can drift. Two params differ: the fill's darkness,
        which is the setting for a re-outlined drawing, and the cleaning of
        the splat's coverage, which a mesh silhouette does not need."""
        spec = self._spec()
        first = self._step(spec, "render_initial_views")
        again = self._step(spec, "render_reoutlined_views")
        self.assertEqual(again.params, dict(first.params, outline_strength="${globals.reoutlined_strength}",
                                            outline_mask_clean_px=9))
        self.assertNotIn("outline_mask_clean_px", first.params)
        self.assertEqual(first.params["outline_strength"], "${globals.outline_strength}")
        self.assertEqual(again.inputs, dict(first.inputs, outline_masks="scene.outline_masks"))
        self.assertEqual(set(again.outputs), {"images", "masks", "inactive_masks"})
        self.assertEqual(again.outputs["images"], "dataset.images")

    def test_the_two_fill_strengths_are_settings_with_one_home_each(self):
        """`outline_strength` is the faint fill of the body model's
        silhouette — what pass 1 sees with the branch off, and what the 480p
        pass sees with it on; `reoutlined_strength` is the darker fill of the
        splat's, what pass 1 sees with the branch on, and is greyed out
        behind the switch. Each is read by exactly one render, as
        `${globals.<name>}`, which is what drops the render step's own
        `outline_strength` from the per-step panel."""
        from pipeline.templating import global_ref

        spec = self._spec()
        by_name = {p.name: p for p in spec.settings}
        plain, reoutlined = by_name["outline_strength"], by_name["reoutlined_strength"]
        self.assertEqual((plain.type, plain.default, plain.requires), (float, 6.25, ""))
        self.assertEqual((reoutlined.type, reoutlined.default, reoutlined.requires),
                         (float, 20.0, "re_outline"))
        readers = {}
        for step in spec.steps:
            ref = global_ref(step.params.get("outline_strength"))
            if ref is not None:
                readers.setdefault(ref, []).append(step.id)
        self.assertEqual(readers, {"outline_strength": ["render_initial_views"],
                                   "reoutlined_strength": ["render_reoutlined_views"]})
        # No render sets a literal darkness of its own.
        for step in spec.steps:
            if step.step == "render" and "outline_strength" in step.params:
                self.assertIsNotNone(global_ref(step.params["outline_strength"]), step.id)

    def test_the_anchor_goes_back_in_after_the_re_render(self):
        spec = self._spec()
        first = self._step(spec, "reinject_anchor_initial")
        again = self._step(spec, "reinject_anchor_reoutlined")
        self.assertEqual(again.inputs, first.inputs)
        self.assertEqual(again.outputs, first.outputs)

    def test_the_480p_pass_lands_in_the_debug_bundle(self):
        from pipeline.templating import resolve

        spec = self._spec()
        matte = self._step(spec, "reoutline_matte")
        scope = {"globals": dict(spec.globals, output_root="/out")}
        self.assertEqual(resolve(matte.params, scope)["debug_dir"], "/out/debug/reoutline")


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

    def test_a_setting_requiring_something_undeclared_or_itself_is_refused(self):
        """`requires:` on a setting greys its control out behind another
        setting, so the name has to be one — not an output, not itself."""
        def body(requires):
            return (
                "name: badreq\n"
                "settings:\n"
                "  - name: switch\n    type: bool\n    default: false\n    help: x\n"
                "  - name: knob\n    type: int\n    default: 1\n    help: x\n"
                f"    requires: {requires}\n"
                "outputs:\n"
                "  - name: export_thing\n    dir: thing\n    label: Thing\n    help: x\n"
                "steps:\n"
                "  - id: a\n    step: rmbg\n    dispatch: in_process\n"
                "    when: ${globals.switch}\n"
                "    params:\n      batch_size: ${globals.knob}\n"
                "  - id: b\n    step: rmbg\n    dispatch: in_process\n"
                "    when: ${globals.export_thing}\n"
            )
        for requires in ("no_such_setting", "knob", "export_thing"):
            with self.subTest(requires=requires):
                with self.assertRaises(ValueError) as caught:
                    WorkflowSpec.from_yaml(self._write(body(requires))).validate()
                self.assertIn(requires, str(caught.exception))
        spec = WorkflowSpec.from_yaml(self._write(body("switch")))
        spec.validate()
        self.assertEqual(next(p for p in spec.settings if p.name == "knob").requires, "switch")

    def test_a_setting_read_only_through_another_settings_requires_is_not_an_orphan(self):
        """The switch a `requires:` names is read, the way an output's is."""
        path = self._write(
            "name: reqread\n"
            "settings:\n"
            "  - name: switch\n    type: bool\n    default: false\n    help: x\n"
            "  - name: knob\n    type: int\n    default: 1\n    help: x\n"
            "    requires: switch\n"
            "steps:\n"
            "  - id: a\n    step: rmbg\n    dispatch: in_process\n"
            "    params:\n      batch_size: ${globals.knob}\n"
        )
        WorkflowSpec.from_yaml(path).validate()

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
        spec = WorkflowSpec.from_yaml(str(WORKFLOW_DIR / "helical.yaml"))
        self.assertIs(spec.coerce_global("run_upscale", "no"), False)
        self.assertEqual(spec.coerce_global("seed", "7"), 7)
        # An undeclared global has no type to be brought to.
        self.assertEqual(spec.coerce_global("output_root", 3), 3)

    def test_one_seed_reaches_every_stochastic_step(self):
        """It was three step params holding 0, 0 and 42 — one run drawing
        three unrelated samples. Four readers since 2026-09-08: the gated
        re-outline denoise draws the same seed as pass 1, so the silhouette
        it cuts is of the sample pass 1 would have drawn at 480p."""
        stochastic = {"wan22_vace_denoise", "seedvr2"}
        for path in _workflows():
            spec = WorkflowSpec.from_yaml(str(path))
            readers = [step.id for step in spec.steps
                       if step.params.get("seed") == "${globals.seed}"]
            expected = [step.id for step in spec.steps if step.step in stochastic]
            with self.subTest(workflow=path.name):
                # Every denoise and the upscale, and nothing else: four.
                self.assertEqual(readers, expected)
                self.assertGreaterEqual(len(readers), 3)
                self.assertTrue(any("upscale" in r for r in readers))

class TestPassTwoIsTheSplatsReRender(unittest.TestCase):
    """`render_subject` is `render_splat`, the splat's own re-render along the
    helix, and the frames pass 2 conditions on. From 2026-09-17 to 2026-09-20
    it was a subclass that could draw the textured mesh of the mesh path
    instead (`pass2_mesh`); the path — meshify, photo_texture, refine_texture
    and their globals — is removed, so nothing mesh-shaped may remain."""

    def _spec(self):
        return WorkflowSpec.from_yaml(str(next(p for p in _workflows() if p.name == "helical.yaml")))

    def test_the_subject_render_is_render_splat_and_sits_before_the_matte(self):
        spec = self._spec()
        ids = [s.id for s in spec.steps]
        at = ids.index("render_subject")
        step = spec.steps[at]
        self.assertEqual(step.step, "render_splat")
        self.assertEqual(step.params["pattern"], "helical", "the path is the same helical one")
        self.assertTrue(step.params["override_cam_from_mesh"])
        for key in ("from_mesh", "mesh_blur_px", "mesh_render_dir"):
            self.assertNotIn(key, step.params)
        for key in ("mesh_dir", "mesh_texture_path", "mesh_photo_texture_path"):
            self.assertNotIn(key, step.inputs)
        for field in ("images", "masks", "cameras", "anchor_position", "anchor_frame_index"):
            self.assertIn(field, step.outputs)
        self.assertLess(at, ids.index("resplat_foreground_masks"))
        self.assertIs(spec.steps[ids.index("resplat_foreground_masks")].when, True, "the rmbg matte always runs")
        self.assertLess(ids.index("resplat_foreground_masks"), ids.index("mask_splat_fringes"))
        self.assertLess(ids.index("mask_splat_fringes"), ids.index("reinject_anchor"), "the anchor is still re-injected after")
        self.assertLess(ids.index("reinject_anchor"), ids.index("denoise_pass2"))

    def test_nothing_of_the_mesh_path_remains(self):
        for path in _workflows():
            spec = WorkflowSpec.from_yaml(str(path))
            with self.subTest(workflow=path.name):
                steps = {s.step for s in spec.steps} | {s.id for s in spec.steps}
                self.assertFalse(steps & {"meshify", "photo_texture", "refine_texture",
                                          "render_subject_mesh", "render_mesh_views"})
                self.assertNotIn("render_subject", {s.step for s in spec.steps})
                names = set(spec.globals) | {p.name for p in spec.settings}
                self.assertFalse(names & {"export_mesh", "pass2_mesh", "photo_texture", "refine_texture",
                                          "texture_mode", "face_policy", "mesh_blur"})

    def test_when_forms(self):
        from pipeline.workflow import when_truthy

        self.assertTrue(when_truthy({"any": [False, "true"]}))
        self.assertFalse(when_truthy({"any": [False, "false"]}))
        self.assertFalse(when_truthy([{"any": [True]}, False]))
        self.assertTrue(when_truthy([{"any": [False, True]}, True]))
        self.assertTrue(when_truthy({"not": "false"}))
        self.assertFalse(when_truthy({"not": True}))
        with self.assertRaises(ValueError):
            when_truthy({"all": [True]})


if __name__ == "__main__":
    unittest.main()


class TestTheIntermediateSplatIsKept(unittest.TestCase):
    """The first brush training exports somewhere the result .zip carries.

    That splat is what the helical re-render is built from, so it is the
    first thing to look at when the re-render is wrong — and it is only
    reachable afterwards if it lands under the run's `debug/`, which is
    what `runs.DEBUG_SUBDIRS` packages. An `output_dir` here instead would
    put it in `brush/training_<ms>/`: on the volume, and gone with the pod.
    """

    def test_the_first_training_exports_into_debug(self):
        from pipeline.cli import resolve_workflow
        from pipeline.templating import resolve
        from pipeline.workflow import WorkflowSpec

        spec = WorkflowSpec.from_yaml(resolve_workflow("helical"))
        scope = {"globals": dict(spec.globals, output_root="/out")}
        trainings = {
            step.id: resolve(step.params, scope)
            for step in spec.steps if step.step == "brush"
        }
        self.assertIn("train_splat", trainings)
        intermediate = trainings["train_splat"]
        self.assertEqual(intermediate.get("export_dir"), "/out/debug")
        self.assertTrue(str(intermediate.get("export_name", "")).endswith(".ply"))
        # `output_dir` would win nothing here (export_dir takes precedence),
        # but leaving both set would be a contradiction to read later.
        self.assertIsNone(intermediate.get("output_dir"))

    def test_the_two_trainings_cannot_collide(self):
        from pipeline.cli import resolve_workflow
        from pipeline.templating import resolve
        from pipeline.workflow import WorkflowSpec

        spec = WorkflowSpec.from_yaml(resolve_workflow("helical"))
        scope = {"globals": dict(spec.globals, output_root="/out")}
        exports = [
            (resolve(step.params, scope).get("export_dir"),
             resolve(step.params, scope).get("export_name"))
            for step in spec.steps if step.step == "brush"
        ]
        self.assertEqual(len(exports), len(set(exports)), f"two trainings share a path: {exports}")


class TestTheFirstDenoiseInputIsKept(unittest.TestCase):
    """The control video pass 1 is handed lands in the debug bundle.

    Everything else a run exports describes frames from AFTER a denoise —
    `colmap_intermediate/` is pass 1's output, `colmap/` is pass 2's. The
    drawings that caused them lived in memory only, so "was the skeleton
    too much ink", "did the face splat land on the face" and "does the
    anchor frame carry the photograph" could be asked of a finished run
    only by re-running the first fourteen steps from the same upload — and
    not at all once the upload was gone.
    """

    def _spec(self):
        from pipeline.cli import resolve_workflow
        from pipeline.workflow import WorkflowSpec

        return WorkflowSpec.from_yaml(resolve_workflow("helical"))

    def test_the_dataset_is_saved_under_debug(self):
        from pipeline.templating import resolve

        spec = self._spec()
        scope = {"globals": dict(spec.globals, output_root="/out")}
        saves = {
            step.id: resolve(step.params, scope)["directory"]
            for step in spec.steps if step.step == "save_dataset"
        }
        self.assertIn("dump_denoise_input", saves)
        self.assertEqual(saves["dump_denoise_input"], "/out/debug/denoise_pass1_input")
        for step_id, directory in saves.items():
            self.assertTrue(
                directory.startswith("/out/debug/"),
                f"{step_id} writes to {directory!r}, which the result .zip never sees")
        self.assertEqual(len(set(saves.values())), len(saves))

    def test_it_saves_what_the_denoise_reads(self):
        """After the anchor injection, before pass 1 — so the frames on disk
        carry the warped photograph and the 0.0 VACE mask at the anchor,
        which is the half of the input a mesh render cannot be re-derived
        into."""
        spec = self._spec()
        order = [step.id for step in spec.steps]
        self.assertLess(order.index("reinject_anchor_initial"), order.index("dump_denoise_input"))
        self.assertLess(order.index("dump_denoise_input"), order.index("denoise_pass1"))
        dump = next(s for s in spec.steps if s.id == "dump_denoise_input")
        denoise = next(s for s in spec.steps if s.id == "denoise_pass1")
        self.assertEqual(dump.inputs["dataset"], "dataset")
        # The frames and their masks are what the dump is for; both reach
        # the denoise from the same dataset the dump wrote.
        self.assertEqual(denoise.inputs["control_video"], "dataset.images")
        self.assertEqual(denoise.inputs["control_masks"], "dataset.masks")


class TestOnlyTheHelicalRerenderCapsTheShBands(unittest.TestCase):
    """`render_subject` is the one render_splat that sets `sh_degree`, and
    it sets 2. Every other instance leaves the step's default (3, every
    band the splat carries) alone — pinned as a workflow decision rather
    than as a step default, because a `sh_degree:` line quietly added to
    the face cap's render would change what the final training is
    supervised by."""

    def test_render_subject_says_two_and_nothing_else_says_anything(self):
        from pipeline.cli import resolve_workflow
        from pipeline.workflow import WorkflowSpec

        spec = WorkflowSpec.from_yaml(resolve_workflow("helical"))
        renders = [s for s in spec.steps if s.step in _SPLAT_RENDERS]
        self.assertGreater(len(renders), 1)
        setting = {s.id: s.params.get("sh_degree") for s in renders}
        self.assertEqual(setting.pop("render_subject"), 2)
        self.assertEqual(set(setting.values()), {None}, setting)


class TestTheSingleViewInput(unittest.TestCase):
    """A single frontal photo rides the same workflow as a sheet: the split
    step decides which it is (`input_layout`, auto by default) and
    publishes the verdict, and `pick_rear_view` reads that verdict after
    pass 1 to fill the reference slot the photo could not. Neither is
    gated — a `when:` resolves against globals before the run, so a
    runtime verdict cannot switch a step off — and the wiring below is what
    makes the two modes share one file (steps/reference_view.py)."""

    def _spec(self):
        from pipeline.cli import resolve_workflow
        from pipeline.workflow import WorkflowSpec

        return WorkflowSpec.from_yaml(resolve_workflow("helical"))

    def _step(self, spec, step_id):
        return next(s for s in spec.steps if s.id == step_id)

    def test_the_split_reads_the_setting_and_publishes_its_verdict(self):
        spec = self._spec()
        setting = next(p for p in spec.settings if p.name == "input_layout")
        self.assertEqual(setting.default, "auto")
        self.assertEqual(list(setting.choices), ["auto", "sheet", "single"])
        split = self._step(spec, "split_sheet")
        self.assertEqual(split.params.get("layout"), "${globals.input_layout}")
        self.assertEqual(split.outputs.get("layout"), "scene.input_layout")

    def test_pick_rear_view_sits_after_pass_1_and_before_the_training(self):
        spec = self._spec()
        order = [s.id for s in spec.steps]
        pick = order.index("pick_rear_view")
        self.assertGreater(pick, order.index("foreground_masks"), "it needs the mattes")
        self.assertGreater(pick, order.index("adjust_denoised"), "the treated frames")
        self.assertLess(pick, order.index("train_splat"))
        self.assertLess(pick, order.index("denoise_pass2"))
        step = self._step(spec, "pick_rear_view")
        self.assertTrue(step.when is True, "runs in both modes; see the class docstring")
        self.assertEqual(step.inputs.get("layout"), "scene.input_layout")
        self.assertEqual(step.inputs.get("reference_image"), "dataset.reference_image")
        self.assertEqual(step.outputs.get("reference_image"), "dataset.reference_image")
        self.assertEqual(step.inputs.get("masks"), "dataset.masks")

    def test_the_reference_has_exactly_two_writers(self):
        """The split (the back panel, or None) and the pick (the rear view,
        or the back panel again). A third would be a second opinion on
        what pass 2 conditions on."""
        spec = self._spec()
        writers = [
            s.id for s in spec.steps
            if "dataset.reference_image" in s.outputs.values()
        ]
        self.assertEqual(writers, ["split_sheet", "pick_rear_view"])

    def test_pass_2_reads_what_the_pick_wrote(self):
        spec = self._spec()
        order = [s.id for s in spec.steps]
        for step_id in ("denoise_pass2",):
            step = self._step(spec, step_id)
            self.assertGreater(order.index(step_id), order.index("pick_rear_view"))
            self.assertEqual(step.inputs.get("reference_image"), "dataset.reference_image")
        # and pass 1 reads the slot BEFORE the pick, when a photo leaves it empty
        self.assertLess(order.index("denoise_pass1"), order.index("pick_rear_view"))


class TestThePixelOpsShipOff(unittest.TestCase):
    """`adjust_denoised` (pixel_ops) runs in the workflow, but every knob on
    it is off unless a setting turns it on. The SH cap on `render_subject`
    is aimed at part of what the highlight suppression was for — the
    specular sparkle a re-render carries into pass 2 — so the two are kept
    separable: the pixel op stays a choice, not a default."""

    def test_specular_suppress_is_zero_and_is_what_the_step_reads(self):
        from pipeline.cli import resolve_workflow
        from pipeline.workflow import WorkflowSpec

        spec = WorkflowSpec.from_yaml(resolve_workflow("helical"))
        setting = next(s for s in spec.settings if s.name == "specular_suppress")
        self.assertEqual(setting.default, 0.0)
        step = next(s for s in spec.steps if s.id == "adjust_denoised")
        self.assertEqual(step.step, "pixel_ops")
        self.assertEqual(step.params["specular_suppress"],
                         "${globals.specular_suppress}")


class TestTheDenoiseSettingsAreTheMeasuredOnes(unittest.TestCase):
    """The two passes ship the settings the 2026-09-08 sweeps settled on
    (docs/vace-denoise-findings-2026-09-07.md, sections 5 and 6), applied
    2026-09-09. Pinned because they are decisions with numbers behind them,
    and a step default drifting under them — the step's own `sampler_shift`
    is 8, which cost pass 2 three points of head sharpness — would be
    silent.
    """

    def _step(self, step_id):
        from pipeline.cli import resolve_workflow
        from pipeline.workflow import WorkflowSpec

        spec = WorkflowSpec.from_yaml(resolve_workflow("helical"))
        return next(s for s in spec.steps if s.id == step_id)

    def test_pass_1_is_run_e4_on_euler(self):
        """The skeleton-leak sweep's E4 (shift 5 erases the ink, the
        structure steps stay at full scale and the four detail steps sit at
        a flat half rather than a taper spent on steps 5-6) — with both
        experts on euler since 2026-09-20 rather than E4's uni_pc on the
        high-noise one: at shift 5 the sampler no longer mattered to the
        leak (E2 = E4), and helical-20260920-220458 ran this way. The low
        sampler is written out because the step's default is uni_pc."""
        params = self._step("denoise_pass1").params
        self.assertEqual((params["sampler_high"], params["sampler_low"]),
                         ("euler", "euler"))
        self.assertEqual(params["sampler_shift"], 5)
        self.assertEqual(params["strength"], [1, 1, 0.5, 0.5, 0.5, 0.5])
        self.assertEqual((params["steps_high"], params["steps_low"]), (2, 4))

    def test_pass_2_is_the_quality_sweep_s_corner_at_0_8(self):
        """Euler on the opening steps at shift 2.5 (shift 8 and uni_pc each
        cost ~3 points of head sharpness on a splat render), a flat 0.8 on
        every step (strength is a fidelity-vs-texture dial; 0.8 is the
        texture compromise, and a taper to 0 is invention), 2/4 kept."""
        params = self._step("denoise_pass2").params
        self.assertEqual((params["sampler_high"], params["sampler_low"]),
                         ("euler", "euler"))
        self.assertEqual(params["sampler_shift"], 2.5)
        self.assertEqual(params["strength"], [0.8] * 6)
        self.assertEqual((params["steps_high"], params["steps_low"]), (2, 4))


class TestTheReoutlineBranch(unittest.TestCase):
    """The experimental branch that redraws the silhouette from a splat of
    the subject (docs/re-outline.md): eight gated steps between the anchor
    injection and the first denoise. What is pinned is what makes it a
    faithful copy of pass 1 and of the first render — a different denoise
    would matte a different subject, a splat trained or rendered on other
    cameras would put silhouette i on the wrong frame — and that with the
    setting off nothing of it runs.
    """

    BRANCH = [
        "reoutline_downscale", "reoutline_denoise", "reoutline_matte",
        "reoutline_upscale", "reoutline_train_splat", "render_reoutline_splat",
        "render_reoutlined_views", "reinject_anchor_reoutlined",
    ]

    def _spec(self):
        from pipeline.cli import resolve_workflow

        return WorkflowSpec.from_yaml(resolve_workflow("helical"))

    def _step(self, spec, step_id):
        return next(s for s in spec.steps if s.id == step_id)

    def test_it_sits_between_the_anchor_injection_and_the_dump(self):
        spec = self._spec()
        order = [s.id for s in spec.steps]
        first = order.index("reinject_anchor_initial") + 1
        self.assertEqual(order[first:first + len(self.BRANCH)], self.BRANCH)
        self.assertEqual(order[first + len(self.BRANCH)], "dump_denoise_input")

    def test_every_step_is_gated_on_the_setting_and_it_defaults_on(self):
        spec = self._spec()
        setting = next(s for s in spec.settings if s.name == "re_outline")
        self.assertIs(setting.default, True)
        for step_id in self.BRANCH:
            with self.subTest(step=step_id):
                self.assertEqual(self._step(spec, step_id).when, "${globals.re_outline}")

    def test_the_extra_denoise_is_pass_1_at_480p_and_four_steps(self):
        """Pass 1's block, apart from the size, the step count (2 high /
        2 low: the pass is kept for its shape, the low expert's extra steps
        are texture) and the per-step list that count reshapes — the
        strength schedule over four entries."""
        spec = self._spec()
        extra = self._step(spec, "reoutline_denoise")
        pass1 = self._step(spec, "denoise_pass1")
        self.assertEqual((pass1.params["steps_high"], pass1.params["steps_low"]), (2, 4))
        self.assertEqual(pass1.params["strength"], [1, 1, 0.5, 0.5, 0.5, 0.5])
        expected = dict(pass1.params, width=480, height=832, steps_low=2,
                        strength=[1, 1, 0.5, 0.5])
        self.assertEqual(extra.params, expected)
        self.assertEqual((extra.dispatch, extra.env, extra.keep_loaded),
                         (pass1.dispatch, pass1.env, pass1.keep_loaded))
        self.assertEqual(extra.inputs["reference_image"], pass1.inputs["reference_image"])
        self.assertEqual(extra.inputs["subject_desc"], pass1.inputs["subject_desc"])
        # It reads the downscaled batch, not the dataset, and writes beside it.
        self.assertEqual(extra.inputs["control_video"], "scene.reoutline.images")
        self.assertEqual(extra.inputs["control_masks"], "scene.reoutline.masks")
        self.assertEqual(extra.outputs, {"images": "scene.reoutline.denoised"})

    def test_the_batch_is_resized_here_not_by_diffusers(self):
        """diffusers fits a control video under the target AREA (464x832 for
        a 720x1280 batch asked for 480x832) — see steps/resize.py."""
        spec = self._spec()
        down = self._step(spec, "reoutline_downscale")
        self.assertEqual(down.params, {"width": 480, "height": 832})
        self.assertEqual(down.inputs, {"images": "dataset.images", "masks": "dataset.masks"})
        up = self._step(spec, "reoutline_upscale")
        self.assertEqual(up.params, {"width": "${globals.resolution.0}",
                                     "height": "${globals.resolution.1}"})
        self.assertEqual(up.inputs, {"images": "scene.reoutline.denoised",
                                     "masks": "scene.reoutline.mattes"})
        self.assertEqual(up.outputs, {"images": "scene.reoutline.frames",
                                      "masks": "scene.reoutline.frame_mattes"})

    def test_the_splat_is_trained_on_the_orbit_cameras_from_the_upscaled_pass(self):
        """The outline is the splat's coverage, so the splat has to be fitted
        on the very cameras the orbit is re-rendered from, to frames at
        those cameras' size; and only to the 480p pass — none of the
        supporting views, weights, rig or mesh the intermediate takes,
        which would pull it toward something other than that pass."""
        spec = self._spec()
        train = self._step(spec, "reoutline_train_splat")
        self.assertEqual(train.step, "brush")
        self.assertEqual(train.inputs, {
            "cameras": "dataset.cameras",
            "image_names": "dataset.image_names",
            "points_3d": "dataset.points_3d",
            "images": "scene.reoutline.frames",
            "masks": "scene.reoutline.frame_mattes",
        })
        self.assertEqual(train.outputs, {"splat_path": "scene.reoutline.splat_path"})
        # The intermediate's silhouette knobs, no alignment, no polish, half
        # the iterations (the texture is thrown away), and the .ply in the
        # debug bundle beside intermediate_splat.ply.
        intermediate = self._step(spec, "train_splat")
        for key in ("polish_steps", "align_iters", "match_alpha_weight"):
            with self.subTest(param=key):
                self.assertEqual(train.params[key], intermediate.params[key])
        self.assertEqual(train.params["total_steps"], intermediate.params["total_steps"] // 2)
        self.assertEqual(train.params["export_dir"], intermediate.params["export_dir"])
        self.assertEqual(train.params["export_name"], "reoutline_splat.ply")
        self.assertNotEqual(train.params["export_name"], intermediate.params["export_name"])
        self.assertNotIn("hollow_weight", train.params)

    def test_the_outline_is_the_splat_rendered_on_the_dataset_cameras(self):
        """No pattern: render_splat reuses the dataset's cameras verbatim,
        which is what puts silhouette i on frame i. Only the alpha is
        published, at the render's size, so `render` accepts it."""
        spec = self._spec()
        render = self._step(spec, "render_reoutline_splat")
        self.assertEqual(render.step, "render_splat")
        self.assertEqual(render.inputs, {"splat_path": "scene.reoutline.splat_path",
                                         "dataset": "dataset"})
        self.assertNotIn("pattern", render.params)
        self.assertNotIn("override_cam_from_mesh", render.params)
        self.assertFalse(render.params.get("confidence", False))
        self.assertEqual((render.params["width"], render.params["height"]),
                         ("${globals.resolution.0}", "${globals.resolution.1}"))
        self.assertEqual(render.outputs, {"masks": "scene.outline_masks"})

    def test_the_re_render_is_the_first_render_plus_the_mattes(self):
        """Same params, so the same cameras; and it republishes nothing about
        them, so nothing can drift. Two params differ: the fill's darkness,
        which is the setting for a re-outlined drawing, and the cleaning of
        the splat's coverage, which a mesh silhouette does not need."""
        spec = self._spec()
        first = self._step(spec, "render_initial_views")
        again = self._step(spec, "render_reoutlined_views")
        self.assertEqual(again.params, dict(first.params, outline_strength="${globals.reoutlined_strength}",
                                            outline_mask_clean_px=9))
        self.assertNotIn("outline_mask_clean_px", first.params)
        self.assertEqual(first.params["outline_strength"], "${globals.outline_strength}")
        self.assertEqual(again.inputs, dict(first.inputs, outline_masks="scene.outline_masks"))
        self.assertEqual(set(again.outputs), {"images", "masks", "inactive_masks"})
        self.assertEqual(again.outputs["images"], "dataset.images")

    def test_the_two_fill_strengths_are_settings_with_one_home_each(self):
        """`outline_strength` is the faint fill of the body model's
        silhouette — what pass 1 sees with the branch off, and what the 480p
        pass sees with it on; `reoutlined_strength` is the darker fill of the
        splat's, what pass 1 sees with the branch on, and is greyed out
        behind the switch. Each is read by exactly one render, as
        `${globals.<name>}`, which is what drops the render step's own
        `outline_strength` from the per-step panel."""
        from pipeline.templating import global_ref

        spec = self._spec()
        by_name = {p.name: p for p in spec.settings}
        plain, reoutlined = by_name["outline_strength"], by_name["reoutlined_strength"]
        self.assertEqual((plain.type, plain.default, plain.requires), (float, 6.25, ""))
        self.assertEqual((reoutlined.type, reoutlined.default, reoutlined.requires),
                         (float, 20.0, "re_outline"))
        readers = {}
        for step in spec.steps:
            ref = global_ref(step.params.get("outline_strength"))
            if ref is not None:
                readers.setdefault(ref, []).append(step.id)
        self.assertEqual(readers, {"outline_strength": ["render_initial_views"],
                                   "reoutlined_strength": ["render_reoutlined_views"]})
        # No render sets a literal darkness of its own.
        for step in spec.steps:
            if step.step == "render" and "outline_strength" in step.params:
                self.assertIsNotNone(global_ref(step.params["outline_strength"]), step.id)

    def test_the_anchor_goes_back_in_after_the_re_render(self):
        spec = self._spec()
        first = self._step(spec, "reinject_anchor_initial")
        again = self._step(spec, "reinject_anchor_reoutlined")
        self.assertEqual(again.inputs, first.inputs)
        self.assertEqual(again.outputs, first.outputs)

    def test_the_480p_pass_lands_in_the_debug_bundle(self):
        from pipeline.templating import resolve

        spec = self._spec()
        matte = self._step(spec, "reoutline_matte")
        scope = {"globals": dict(spec.globals, output_root="/out")}
        self.assertEqual(resolve(matte.params, scope)["debug_dir"], "/out/debug/reoutline")


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
        spec = WorkflowSpec.from_yaml(str(WORKFLOW_DIR / "helical.yaml"))
        self.assertIs(spec.coerce_global("run_upscale", "no"), False)
        self.assertEqual(spec.coerce_global("seed", "7"), 7)
        # An undeclared global has no type to be brought to.
        self.assertEqual(spec.coerce_global("output_root", 3), 3)

    def test_one_seed_reaches_every_stochastic_step(self):
        """It was three step params holding 0, 0 and 42 — one run drawing
        three unrelated samples. Four readers since 2026-09-08: the gated
        re-outline denoise draws the same seed as pass 1, so the silhouette
        it cuts is of the sample pass 1 would have drawn at 480p."""
        stochastic = {"wan22_vace_denoise", "seedvr2"}
        for path in _workflows():
            spec = WorkflowSpec.from_yaml(str(path))
            readers = [step.id for step in spec.steps
                       if step.params.get("seed") == "${globals.seed}"]
            expected = [step.id for step in spec.steps if step.step in stochastic]
            with self.subTest(workflow=path.name):
                # Every denoise and the upscale, and nothing else: four.
                self.assertEqual(readers, expected)
                self.assertGreaterEqual(len(readers), 3)
                self.assertTrue(any("upscale" in r for r in readers))

if __name__ == "__main__":
    unittest.main()
