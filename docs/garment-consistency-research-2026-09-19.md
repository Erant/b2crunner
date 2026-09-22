# Loose garments: why pass 1 billows them, what the trainer can and cannot do about it, and the generic fix

Research note, 2026-09-19 (night). Subject: run `helical-20260920-010022-c0514e`
(b2crunner `90db726`, i.e. the memorising sync — see §5), a woman in a split
knee-length coat over armour, mid-stride. The coat is in a different place in
every one of the 81 pass-1 frames; the intermediate splat renders its hem as a
translucent cloud; pass 2 does not repair it; the final splat has the same cloud.
Everything measured here ran on the 4070 Ti against the run's exported
`debug/` tree; tools and images are in `~/Projects/b2ctrain/out/garment/`
(README there).

## 1. What the frames actually contain

Two measurements against the naked SAM-3D-Body mesh (`debug/body_refit/mesh_before.ply`,
rasterised at the 81 given pass-1 cameras from `debug/denoise_pass1_input/metadata.json`;
1 px = 1.8 mm at the subject):

**The drawn coat has no rigid part beyond the body.** The strict visual hull of the
81 pass-1 mattes (a voxel survives only if every view puts it inside the subject,
`hull.py`, 1 cm voxels) is **62 litres; the mesh alone is 61**. At a 90 % vote the
hull is 85 l, at 75 % 111–148 l, at 50 % 273 l — a symmetrised tent, not a
garment. Per frame, **7.5 % of the subject's pixels (p90 11.7 %) lie outside the
strict hull**; for a rigid clothed body that number is the matte noise, ~1 %.
The frames disagree about the coat *everywhere*, not just at the flapping panel.

**It is motion, not flicker.** Per row of the image, the matte's lateral excess
over the mesh silhouette (`envelope.py`, `excess_vs_azimuth.png`) is a smooth
function of the camera azimuth — **3.5 px mean change from one frame to the
next** — but no rigid per-row ellipse fits it: mean residual 21.6 px (4 cm), p90
58.7 px (10.6 cm), while the flare itself is 87–96 px (16–17 cm) at its widest
row. The coat swings as the camera goes round. Given a mid-stride pose and a coat
already caught in motion in the photograph, Wan's prior is a walking person; the
prompt's 时间静止 / 人物完全静止 holds the body (the legs follow the skeleton) but
not the cloth, because the cloth has no channel holding it.

**Why it has no channel.** Pass 1's control (`debug/denoise_pass1_input/`,
`pass1_input.jpg`) is the DWPose skeleton plus the naked mesh's silhouette as a
faint fill (#6F on #7F, 4 px blur — faint *by design*, so the naked model does not
override hair and clothing). The coat exists only in the reference image. VACE has
to put it somewhere in each frame, and nothing says where. `re_outline` (docs/
re-outline.md) was built for hair and coats but takes its silhouette from a
480p pilot denoise *per frame* — the pilot's own billow, not a rigid shape.

## 2. What pass 2 saw, and why it did not help

Pass 2 conditioned on the intermediate splat's confidence-gated render (log lines
1417–1425; reproduced locally, `r_inter_conf.jpg`). The gate does its job: the
cloud is culled and what remains is a body-hugging coat, roughly the 90 % hull —
**but with ragged, torn hem edges and grey holes where the flare was**. At strength
0.8 that is an invitation: pass 2 fills the tatters with cloth, in motion again
(`pass2_sheet.jpg`). So the gate gives a *narrow* coat, not a *central* one, and a
torn conditioning render is followed no better than none. Note also that the run
predates `27e3ebf` (the non-memorising sync + `sync_confidence`): the log says
`b2crunner 90db726`, the sync's `sync_max_splats` was 400000 and its renders equal
x0 in `debug/sync_pass1` (`sync_sheet.jpg`). The gated sync has not been tested.

## 3. Can the trainer make a consensus? Measured: no

Two cheap arms on the exported intermediate dataset (`ds_base`, production flags
minus the body rig, 1m50s each on the 4070 Ti; metric `ghost_metric.py` = translucent
fringe (0.1<α<0.7) and gate-culled area, over the garment rows):

| arm | fringe | culled | what it did |
|---|---|---|---|
| baseline (as the pod) | 0.087 | 0.121 | the cloud |
| `cons`: one IRLS round — pixels where a frame disagrees with the gated render on coverage (4 px tolerance) or colour get weight 0 ("unknown, not evidence") | 0.058 | 0.102 | fewer wisps; hem still torn (`base_vs_cons_hem.jpg`) |
| `env`: the clothed envelope of §4 as a hard silhouette — frame pixels outside it → background, envelope pixels the frame leaves empty → weight 0 | 0.083 | 0.084 | coat confined to the envelope, no wisps outside; hem inside still dark and ragged (`base_vs_env_hem.jpg`) |

Both help at the margin, neither produces a solid hem, and the reason is
structural: at a given envelope pixel only a minority of frames show coat, so
whatever the loss does there it has too little evidence to paint. **A splat
trainer cannot manufacture agreement the frames do not contain.** The same holds
for the in-loop sync splat: its 81 x0s disagree in the same way, and a
non-memorising mean of them is a blur across the union — a hint at *some* coat,
not a decision about *which*. The fix has to be upstream, in what pass 1 is told.

## 4. The generic fix: a rigid clothed envelope, built once, drawn into every control frame

The one thing all loose parts (coats, skirts, capes, long hair, platform soles)
have in common is that the photograph shows their extent from one view and the
naked mesh does not have them. The generic object is therefore **the naked mesh
inflated, per body part, by what the photograph's matte shows beyond that part**,
with a depth-to-width ratio as the only prior:

1. For each body part (the rig's skinning; the experiment used "hips + legs =
   vertices below crotch and within 22 cm of the axis" and left the arms alone),
   measure per image row at the anchor camera how far the matte's outer edge lies
   beyond the part's silhouette: `a_x(row)`. Median-filter over 9 rows so the hem
   stays a step. Rows where the matte fills the gap between the two legs are skirt
   rows (`span_fill`).
2. In every view at azimuth θ from the photograph, dilate the part's silhouette by
   `sqrt((a_x cos θ)² + (K a_x sin θ)²)` — an elliptical cross-section with depth
   ratio K — and on skirt rows fill between the outer edges. Union with the full
   mesh silhouette. (`proxy3.py`; the isotropic first attempt `proxy.py` was a tent,
   and inflating whole rows `proxy2.py` gave the hands wings — hence per part.)
3. Render **that** as pass 1's silhouette fill, through `render`'s existing
   `outline_masks` input (re_outline's plumbing, byte-identical fill code), and
   draw its **contour as a line** on top, in the skeleton's own modality: the one
   channel VACE is proven to follow frame for frame here. The hem line is then the
   same rigid curve in all 81 frames.

Against this run's frames (`proxy3_sheet_K0.25.jpg`; red = envelope the frame did
not fill, blue = frame outside the envelope): the envelope is a plausible coat from
every angle; IoU(frame, envelope) 0.80 vs 0.82 for the naked mesh — the same,
because the frames' coats are all over the place — with the matte outside it 9 %
(sleeves, shoulder pads, ponytail: the parts the experiment did not inflate) vs
16.6 % outside the mesh. K: the frames themselves say the side view wants a thin
coat (the rigid fit's depth axis came out 0 at every hem row, IoU 0.808 at K=0 →
0.793 at K=0.5), the photograph's flare says wide; K=0.25 is the compromise drawn
above, and it is a per-run parameter, not a discovery.

Why this is the right *kind* of fix: it is rigid by construction, it is built from
the two things every run already has (the photo's rmbg matte and the mesh), it
costs nothing at denoise time, it is agnostic to what the loose part is, and it
uses the control channel that measurably holds. What it does not do is make the
envelope *correct* — the coat's true depth extent is unknowable from one photo —
but the request was one place, not the right place; the photograph pins the front,
and VACE textures the rest.

**Riders, same direction, weaker:**
- The same envelope as the hard silhouette of the intermediate splat and of the
  sync splat (§3's `env` arm, `envelope_ds.py`). Once the frames agree it is
  redundant; until they do it stops the cloud leaving the envelope and gives the
  sync splat a *decision* to feed back instead of a blur.
- Negative-prompt the motion: 裙摆飘动, 衣物飘动, 布料摆动. Free, unmeasured, and
  it must not ride in the same arm as anything else.
- The heavy generic option: a single-image clothed reconstructor (ECON, SiTH,
  LHM++) as the geometry source instead of the naked body — its silhouettes carry
  the garment natively. Days of integration, VRAM, and the pipeline rethink's
  territory; the envelope above is the two-day version of the same idea.

## 5. What to run on a pod

One change per arm, seed and everything else as c0514e:

- **A0** baseline (`ebda0fe`, sync on — this also runs the non-memorising sync for
  the first time; read `debug/sync_pass1` renders against x0).
- **A1** envelope as the silhouette fill only (`outline_masks` from the new step).
- **A2** envelope fill + contour line.

Readouts, all from `debug/colmap_intermediate` with the tools in
`~/Projects/b2ctrain/out/garment/`: strict-hull volume minus the mesh's (1 l →
tens of litres = a coat that exists in 3D); matte outside the strict hull (7.5 %
→ ~1–2 %); frame-to-frame excess change (3.5 px → ~1 px); the intermediate
splat's garment-row fringe (0.087) and culled area (0.121). Then the pass-2 sheet
by eye: the hem in one place.

## 6. Implementation sketch (not built)

`garment_envelope` step, `in_process`, after `reinject_anchor_initial` and before
`dump_denoise_input`, gated on a `garment_envelope` global (OFF until A1/A2 say
otherwise): inputs = `scene` (mesh + camera, as `render` takes them), the photo's
rmbg matte (helical_shell's `front_matte` step exists), the render's cameras;
params `depth_ratio` (K), `parts` (default: everything below the hips), `contour`
(bool), `contour_grey`; output = 81 silhouettes at the render size → a second
`render` step with `outline_masks`, exactly as `render_reoutlined_views` is wired,
then `inject_anchor` over it. body2colmap's `outline_from_mask` draws the fill; the
contour is a 2 px `cv2.findContours` polyline in `outline_strength`'s grey.
Tests: the envelope contains the mesh silhouette in every view; at θ=0 its outer
edge is within 2 px of the photo's matte on the inflated parts; K=0 reproduces the
mesh at θ=±90° on the un-inflated rows.
