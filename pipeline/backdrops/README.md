# Bundled backdrops

Equirectangular environments `steps/backdrop.py` resolves by name.

| name | file | source |
|---|---|---|
| `studio` | `white_studio_06.jpg` | [Poly Haven — White Studio 06](https://polyhaven.com/a/white_studio_06), Grzegorz Wronkowski, CC0 |

`white_studio_06.jpg` is the 4k `.hdr` (4096x2048) put through
body2colmap's own loader (`background._read_image`, Reinhard tone map) and
saved as JPEG q90 — the same pixels the renderer would make from the `.hdr`
itself, at 0.4 MB instead of 25.

Why this one, of the ~90 studio HDRIs on Poly Haven (2026-10-01): it is the
most EVEN light of the white infinity coves measured — horizontal irradiance
around the subject varies 1.6x over a full turn, against 2-4.7x for the rest
(monochrome_studio_03 3.8, white_studio_04 2.1, studio_small_08 2.0) — and
its light comes more from above than from the sides (a frosted skylight
over a white cove). That is the pass-1 prompt's lighting, "soft, even,
low-contrast light ... from every side and from above, like a large softbox
dome", and what a splat wants: a key light from one side would be baked
into the colours. The cost is a strip of furniture and light stands (a
couch, a pink chair) seen past the cove in the views about a quarter of the
way round the orbit.
