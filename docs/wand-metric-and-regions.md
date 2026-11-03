# Wands: metric, region rule, exposure

Status: **design note** (2026-10-28). Not in flight, so not an ADR. Companion to
[wand-contours-and-lofted-geometry.md](wand-contours-and-lofted-geometry.md) (what happens *after*
a region exists) and §5 of [stain-space-and-stack-transparency.md](stain-space-and-stack-transparency.md)
(the comparator registry). Prototype: `playground/src/wand/{boundaryStrength,BoundaryLayer,gradedNecks}.ts`, with the mask and map tools in `playground/src/selection/`. The plain rule's patch runs as a graph of general ops (`src/gpu/graph/wand.ts`: mixtureDistance → Gaussian → Laplacian ≥ 0 → floodFrom, opened by two distance transforms), pixel-identical to the prototype function; so does the boundary field feeding it (`wandField`: gramChannels → channelMix (standardise and project) → channelStructureTensor, with convolveSeparable over the channel planes for the residual modes), within 1e-5 of the prototype shader
on the transcript modes page.

## Decision

1. **A wand takes a comparator**: a CPU/WGSL twin `dist(a, b) → f32` over a field's channel axis,
   registered per field kind (§1). Nothing else is needed: the region rule uses only d.
2. **"The patch the click is in" is a Marr–Hildreth cell, not a tolerance.**

   `d` to the click → smooth at **σ = 1r** → region around the click where **`∇²d ≥ 0`** → morphological
   **opening at ρ = 0.75r** → the click's piece.

   It is closed by construction, sets its own edge per click, and has two scale settings, both in
   units of the kernel radius r, and **no similarity threshold**. It is not a tolerance in
   disguise: its edge d varies from 1.8 to 3.0 across clicks, and no fixed tolerance reproduces it
   (E7).
3. **"Everywhere similar" stays `d ≤ tol`.** It is a different question, with a different answer.
4. **Gram metric:** the q main modes at `1/√λ`, plus the rest at noise level times `w` (PPCA, §2).
   **Boundaries should use the main modes only (w = 0, the default):** a derivative amplifies the
   leftover modes far beyond their variance share. The prototype's boundary field still follows the
   `w` slider, with the leftover modes blurred first (residual smoothing), so the two can be compared.
5. **Boundary strength is a display**, not a rule. Edges taken from the vector field's directional
   derivative (Canny/Cumani) do not close (§3).

## 1. Fields and their comparators

| Field | Channels | Default comparator |
|---|---|---|
| Transcript / cell Gram map | K standardised densities | PPCA Mahalanobis (§2) |
| Image channels (IF, IMC) | `channel` axis | scaled Euclidean, `1/robust sd` (or the channel `window`) |
| RGB / blend result | display sRGB | OKLab ΔE |
| Stain-unmixed H&E | H, E, residual | scaled Euclidean on optical density |
| Op-graph output | `channelAxis(field)` | scaled Euclidean; an op may declare its own |
| any | — | cosine ("same mixture, any brightness"), optional |

The boundary overlay also needs only d. The structure tensor comes from three distances per pixel
by polarisation, `G₁₁ = d²(x, x+e₁)`, `G₂₂ = d²(x, x+e₂)`, `G₁₂ = ½(d²(x, x+e₁+e₂) − G₁₁ − G₂₂)`.
That is exact for the Gram comparator; for the others it is untested.

## 2. The Gram metric

    d² = Σ_{k≤q} Δy_k²/λ_k  +  w · Σ_{k>q} Δy_k²/σ²,     Δy = Vᵀ Δz,   σ² = mean(λ_{k>q})

`w = 0` is today's 3-mode distance; `q = K` is today's "every mode". On Xenium σ²/λ₁ ≈ 0.1, so the
trailing modes are not near-flat: the 1e-6 floor was never the problem. Their trouble is spatial
roughness below r.

| Use | q | w |
|---|---|---|
| region distance (`d ≤ tol`, ∇²d) | envelope count, else 3 | slider; 0 until shown to help |
| boundaries | same q | 0 (smoothing at ≥ 1r makes w = 1 equal to w = 0, see E1) |

## 3. Region rules compared

Same window (3.6 × 1.9 mm, map 215 × 111), same click unless noted.

| Rule | Closes? | Click 1 cell | Verdict |
|---|---|---|---|
| `d ≤ tol`, connected | yes | depends wholly on tol | "everywhere similar", not "the patch" |
| ridge lines ≥ θ, fill (Canny/Cumani) | **no**: gaps at junctions | 66–98% of map for any θ | dropped |
| `∇²d ≥ 0` | yes | 14.6%, a branching network | right edge, wrong shape |
| + `d ≤ tol` gate | yes | thins to the spine, collapses below tol 3 | dropped |
| `d ≤ tol` + opening, no ∇²d | yes | none for any tol (E7) | dropped |
| **+ opening at 0.75r** | yes | **3.3%, compact patch** | **chosen** |

The opening keeps compact cells intact (click 2: 485 → 457 pixels) and cuts branches narrower than
2ρ. The stroma band's branches are about 1–1.5r thick, the kernel's resolution, so ρ must scale
with r.

## 4. Exposure (playground)

**Presets first, sliders behind "Fine-tune".** The panel shows one choice; touching a slider turns it
to "Custom".

| Preset | σ | ρ | Lay line |
|---|---|---|---|
| **The patch the click is in** (default) | 1r | 0.75r | a compact region of the click's mixture |
| Follow the tissue | 1r | off | everything connected with the same mixture, branches and all |
| Fine detail | 0.5r | 0.5r | edges judged at half the radius (untested) |

All presets use q = 3, w = 0, no tolerance.

| Control (fine-tune) | Lay line | Default |
|---|---|---|
| Only the patch the click is in | where the difference from the click stops speeding up | on |
| Smooth the difference first (σ) | the scale at which the edge is judged | 1r |
| Drop parts narrower than (ρ) | trims thin branches, keeps the body | 0.75r |
| How close counts as similar | for "everywhere similar" only | 1.2 |
| Main modes (q) | modes treated as real structure | envelope count, else 3 |
| Count what the main modes leave out (w) | at noise level; 0 = only what the colours show | 0 |
| Show boundaries | where the mixture changes fastest (display) | off |

The readout gives the patch's size and **the d along its edge**: the tolerance the rule chose.

## 5. Open: two neighbourhoods at once

Today every scale knob (σ, ρ) acts on one Gram computed at one r. Combining a narrow r₁ and a broad
r₂ (two Grams over the same window) could be done three ways:

| Option | What d compares | Answers | Cost |
|---|---|---|---|
| **A. coarse picks, fine places** | broad r₂ ∇²d cell chooses the patch; narrow r₁ ∇²d moves its edge in (edge focusing) | "this patch, with a sharp edge" | two Grams; the opening is no longer needed (r₂ has no thin branches) |
| B. stacked channels | `[z(r₁), z(r₂)]`, 2K channels, one Mahalanobis | "same mixture *and* same surroundings" (texture/context) | 2K ≤ 32 channels; corr across scales handles redundancy |
| C. band-pass | `z(r₁) − z(r₂)` | "same local contrast against its surroundings": a difference of Gaussians, i.e. Marr–Hildreth on the fields | as B, K channels |

**A, tried (E6): not better than the opening.** The broad cell at 3r, intersected with the narrow
one (the narrow edge allowed 1r outside the broad), then the click's piece. Pinned window 4.7 ×
1.8 mm, map 285 × 109, % of map:

| Click | narrow | broad (3r) | A | opening 0.75r |
|---|---|---|---|---|
| stroma band | 21.0 | 16.5 | 8.9 | 9.4 |
| epithelium | 44.6 | 52.8 | 3.1 | 4.4 |

On the band A and the opening agree, with A keeping the narrow edge's detail. In epithelium both
scales say "the whole compartment", which is true, and their intersection breaks into fragments
wherever the two disagree. So A's patch there is an accident of noise. **A broad r does not
prune**: a continuous compartment is continuous at every scale. The opening stays the default.
Two scales remain under "Fine-tune" as an option, not a preset. **B** (same mixture *and* same
surroundings) asks a genuinely different question and is still untried.

## 6. Interaction: hover, and a mask built from clicks (prototyped)

Three tools, as deck `IconWidget`s on the map (stateless, so the page owns the tool and H/W/B stay
in sync). **Hand** (H, default): drag pans, the map is as it was. **Brush** (B): drag paints a disc,
swept between pointer events so a fast drag leaves no gaps; same ops and Space as the wand; its outline
is drawn on the map (true to size at any zoom, which a CSS cursor cannot be). **Wand** (W, turns
boundaries on):

| Input | Effect |
|---|---|
| hover | the patch under the pointer, recomputed as it moves (to a quarter map pixel), with the seed's mixture interpolated bilinearly at the pointer rather than read at the nearest pixel; on a branch the opening drops, nothing is shown |
| click or drag (paint) | new mask; with Shift / Alt / Shift+Alt: add / subtract / intersect (Photoshop's bindings) |
| Alt + click or drag (wand) | subtract what matches the patch *and* is connected, within that overlap, to the point |
| Cmd/Ctrl + Alt + click or drag | remove the whole connected piece of the mask under the pointer, whatever its mixture |
| Space + drag | pan (Photoshop's temporary hand tool); scroll still zooms |

The cursor says what a press will do: a crosshair badged +, − or × with the keys held, a hand when a
drag pans, "progress" while the window's field is computing (a drag then pans: a wand that can
neither paint nor pan reads as broken).
| `[` `]`, or Cmd/Ctrl + scroll | size: the wand's σ (E3: the patch grows with σ), or the brush radius |

Cmd/Ctrl counts only when really held (from the keyboard): a trackpad pinch arrives as a Ctrl+wheel
and must still zoom.

**The preview shows the change, not the footprint.** Before a press, by op: cyan where the mask would
grow, red hatching over its amber where it would shrink; where nothing would change, only the
preview's outline. So Shift over existing mask shows no fill (nothing to add), and a plain press shows
the rest of the mask hatched (it would be replaced).

A drag is one stroke: the patches of the pixels it crosses are unioned, then combined with the mask as
it was when the stroke began, so intersect means "keep what this stroke covers". A fast drag skips
pixels between renders; patches are much larger than pixels, so the gaps fill in.

- **The mask lives on its own grid in the transcripts' frame** (`worldMask.ts`), at the first window's
  pixel size. Window maps are rebuilt on every pan, so each patch is carried onto the grid and the mask
  back onto whatever map is showing. Polygons from its outline are the natural MDV form; not done.
- **The first commit locks the modes**: the metric is fitted per window, so without it each addition
  after a pan would use a different "similar".

Cost per patch, CPU (Node, one thread), 3 modes, r = 3 px:

| Map | d + smooth + ∇²d fill | opening | total |
|---|---|---|---|
| 419 × 156 (today's windows) | 12.8 ms | 3.8 ms | ~17 ms |
| 1000 × 1000 | 192 ms | 56 ms | ~250 ms |
| 2000 × 2000 | 870 ms | 261 ms | ~1.1 s |

The smoothing dominates. Everything but the flood fill is per-pixel and would move to GPU passes; the
fill is ~2 ms per million pixels. Hover is affordable on the main thread at today's sizes; past
~1000² it needs those GPU passes, a worker, or a half-resolution preview.

## Evidence (Xenium 2, 9 channels, r = 50 µm, 2026-10-28)

| # | Experiment | Result |
|---|---|---|
| E1 | boundary strength: 3 modes / every mode / PPCA w = 1 | median 0.80 / 2.00 / 1.91; only 3 modes gives ridges. Smoothing the leftover modes at 1r leaves them +7%, at 2r +0%: no structure ≥ r |
| E2 | θ bands vs zero-crossing lines | bands are a few r wide; lines are thin but open (cell 66% at θ = 0.5) |
| E3 | ∇²d, σ sweep 0 → 3r | cell 0.2, 3.0, 13.6, 14.6, 18.6, 21.5, 50.4%; plateau at 0.5–1r; edge d 3.2–3.6 throughout |
| E4 | + tol gate 4 → 1 | 14.4, 13.6, 4.8, 0.3, 0.2, 0.0%: a cliff at the necks |
| E5 | + opening ρ 0.5 → 2r | 8.8, 3.3, 0.3, 0.4, 0%; compact at 0.75r |
| E6 | two scales (r, 3r), intersected | matches the opening on the band (8.9 vs 9.4%); fragments in epithelium (3.1%); see §5 |
| E7 | plain `d ≤ tol` + opening vs the ∇²d cell, 64 clicks on an 8 × 8 grid (7.1 × 3.6 mm, map 429 × 218) | over the 32 tissue clicks: ∇²d edge d 1.8 / 2.1 / 3.0 (p10/p50/p90); median overlap (IoU) 0.01 / 0.14 / 0.21 / 0.06 at tol 1 / 1.5 / 2 / 2.5, and 0.36 even picking the best tol per click; at tol 2 → 2.5 the patch jumps from 2% to 41% of the map (it percolates). The "d ≈ 3.4" of E3 was that window's value, not a constant |
| E8 | hover stability along one row (map 653 × 218, r = 3 px) | seed read at the nearest pixel: 216 of 317 two-pixel steps change the patch by more than half, and 7 showed a piece off the pointer (now: nothing). The unopened fill is 28–60% of the map for every seed; the opening carves the piece out, and which lobes it keeps flips as a neck crosses ρ. Interpolating the seed's mixture: quarter-pixel steps keep overlap 0.94–1.0, but 4 in 32 still flip a lobe (overlap 0.2–0.7). Not fixed by: averaging the reference over r (191), ∇²d ≥ ε margins (178 at 0.3σ, patches shrink), confining the fill to d's basin (172). Graded necks (score = narrowest neck on the widest path from the seed, smoothstep about ρ; `gradedNecks.ts`, "Graded necks" toggle, off): on a two-blob test the far blob fades 0 → 0.34 → 0.93 → 1 as its neck widens; on the real row it removes the big pops over 8 px (worst soft overlap 0.62 vs 0.22) but over 100 px it is worse (105 vs 75 half-pixel steps below 0.5): behind a ~100 px patch the fill has often percolated to 85 000+ px, and grading shows that cluster faintly (500 ↔ 22 000 px of weight between steps). The fix has to stop the fill percolating, not grade what it reaches |

E1–E6: two clicks in one window of one section; E7: one grid in a second window. Treat the defaults as provisional.

## Next

1. Repeat E3/E5 over more windows, sections and radii: is the 0.5–1r plateau, and ρ = 0.75r, general?
2. Does w > 0 change the ∇²d patch? Untested.
3. Move the rule into the real wand (`useWand`, `wandRegion.ts`), replacing "only the patch" with it,
   with §6's hover and mask.
4. A second comparator (image channels) to check that the rule does not depend on the Gram metric.
