# The skeleton-leak sweep — six settings sidecars

The six runs of `docs/vace-denoise-findings-2026-09-07.md` section 5, as
per-image settings sidecars. `pipeline/runs.py`'s zip path is built for
exactly this: one image plus one `<stem>.yaml` beside it is one run at its
own settings, so the same reference sheet copied six times under these six
stems is six variants of one subject.

## Building the zip

Copy your reference sheet to each stem, drop the six YAMLs beside it, zip
the lot flat, and upload it with the Subject box holding the prompt:

    for f in docs/vace-leak-sweep/E*.yaml; do
      cp <your-sheet>.png "$(dirname "$f")/$(basename "$f" .yaml).png"
    done
    cd docs/vace-leak-sweep && zip ../../vace-leak-sweep.zip E*.png E*.yaml

`.png`/`.jpg` both work; the extension is not part of the pairing, the stem
is. Put **no `.txt` in the zip** — `pair_images_with_prompts` switches to
per-image prompt files the moment it sees one, and then every image needs
its own. With none, the Subject box is the prompt for all six, which is
what one subject at six settings wants.

Run names come out as `fast_helical_native-E1-stack`, and so on: the stem
is squeezed into the name, so the archive says which variant it is without
opening `log.txt`.

## Leave the param panel empty

Each sidecar is self-contained — every knob the run needs is in the file.
A sidecar MERGES into the submission-wide overrides key by key and wins on
conflict, but it cannot *remove* one, and the two families here differ
partly by what they leave unset (`denoise_pass2.sampler_shift` and
`render_initial_views.outline_strength` are absent from A and present in
B). Anything left in the panel would leak into the family that is supposed
to be silent about it. Six complete files, an empty panel, no interaction
to reason about.

`splat_inactive_mask` is not in any of them: it is on in the workflow as of
2026-09-08, so the reference runs' `--param` for it has nothing left to
say. Do not add it back — an override that restates the workflow is one
more line to keep in step when the workflow moves.

## Reading the results

    scripts/skeleton_leak.py --iou --by-hue <result-dir> ...

plus `debug/alignment/alignment.json`'s iteration-1 mean. The references
these are deltas from, both measured on 2026-09-08:

    A = 31a008   leak  1.74   yoke 14.54   flow 0.986   IoU 0.8525
    B = 467c17   leak 16.62   yoke 55.61   flow 1.039   IoU 0.8501

`export_debug` is the one that matters. It defaults **true**, and every
sidecar spells it out anyway: it carries both halves of the measurement
(`debug/colmap_intermediate/`, pass 1's frames, and
`debug/denoise_pass1_input/`, the control they are compared against), and
a run without it is wasted. Until 2026-09-08 those frames were their own
output, `export_colmap_intermediate`, which defaulted false — archives
from before then keep them at the top level, and
`scripts/skeleton_leak.py` reads both layouts.

## Results

Measured 2026-09-08 evening; the full reading is section 5 of
`docs/vace-denoise-findings-2026-09-07.md` ("Results"). In one line each:
E1 clean (-0.46, yoke 0.2); E2 clean — shift 5 alone was 9e315f; E3
the worst yoke ever measured (116) — `uni_pc` on the high expert at low
shift; E4 clean with the best silhouette of the three (0.8468); E5 the
skeleton painted literally (105); E6 no ink and no pose (IoU 0.57). The
clean three all cost 17-20% of view flow against A.
