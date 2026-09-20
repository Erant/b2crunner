# Latent-splat experiments (E0–E2)

Local measurements behind section 7 of `../latent-splat-guidance-research-2026-09-19.md`.
Run in `~/Projects/masktest/.venv` (torch 2.13 + diffusers installed 2026-09-19) on the 4070 Ti;
the VAE (`vae/` of `linoyts/Wan2.2-VACE-Fun-14B-diffusers`, 569 MB) is in the HF cache.

    V=~/Projects/masktest/.venv/bin/python; S=<scratch dir>
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $V e0_temporal_mixing.py $S/e0     # ~12 min (81 static encodes)
    $V e0b_steady_state.py $S/e0
    $V e1_equivariance.py $S/e1
    $V e1b_band_split.py $S/e0
    # E2 needs $S/e2/{cameras_480.json,cameras_lat.json,rgb/} built as in the session log
    # (colmap_to_cameras.py on cyber_6f/colmap, intrinsics rescaled to 480x832 and 60x104)
    $V e2b_holdout.py $S/e0 $S/e2 $S/e2b          # RGB splat + six latent fits, odd views, held-out even
    $V e2c_frozen_holdout.py $S/e2 $S/e2b frozen  # same fits, geometry frozen / slow

`e2_latent_bake.py` is the first (all-views, frozen-geometry) version and was run on the
wrong frame set (`circular` frames with `colmap` cameras — the two are numbered differently);
kept for the record, its numbers are not in the note. `cyber_6f/colmap` frames are the ones the
cameras belong to.

Trap: `--lr-mean 0` (or any zero rate) makes b2ctrain write NaN positions; freeze with 1e-12.
