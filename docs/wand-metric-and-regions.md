# Wands: metric, region rule, exposure

Status: **design note** (2026-10-28). Not in flight, so not an ADR. Companion to
[wand-contours-and-lofted-geometry.md](wand-contours-and-lofted-geometry.md) (what happens *after*
a region exists) and §5 of [stain-space-and-stack-transparency.md](stain-space-and-stack-transparency.md)
(the comparator registry). Prototype: `playground/src/transcript/{boundaryStrength,BoundaryLayer}.ts`
on the transcript modes page.

## Decision

1. **A wand takes a comparator**: a CPU/WGSL twin `dist(a, b) → f32` over a field's channel axis,
   registered per field kind (§1). Nothing else is needed: the region rule uses only d.
2. **"The patch the click is in" is a Marr–Hildreth cell, not a tolerance.**

   `d` to the click → smooth at **σ = 1r** → region around the click where **`∇²d ≥ 0`** → morphological
   **opening at ρ = 0.75r** → the click's piece.

   It is closed by construction, sets its own edge (Xenium: d ≈ 3.4 there), and has two scale
   settings, both in units of the kernel radius r, and **no similarity threshold**.
3. **"Everywhere similar" stays `d ≤ tol`.** It is a different question, with a different answer.
4. **Gram metric:** the q main modes at `1/√λ`, plus the rest at noise level times `w` (PPCA, §2).
   **Boundaries use the main modes only (w = 0):** a derivative amplifies the leftover modes far
   beyond their variance share.
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
| **+ opening at 0.75r** | yes | **3.3%, compact patch** | **chosen** |

The opening keeps compact cells intact (click 2: 485 → 457 pixels) and cuts branches narrower than
2ρ. The stroma band's branches are about 1–1.5r thick, the kernel's resolution, so ρ must scale
with r.

## 4. Exposure (playground)

| Control | Lay line | Default |
|---|---|---|
| Only the patch the click is in | where the difference from the click stops speeding up | on |
| Smooth the difference first (σ) | the scale at which the edge is judged | 1r |
| Drop parts narrower than (ρ) | trims thin branches, keeps the body | 0.75r |
| How close counts as similar | for "everywhere similar" only | 1.2 |
| Main modes (q) | modes treated as real structure | envelope count, else 3 |
| Count what the main modes leave out (w) | at noise level; 0 = only what the colours show | 0 |
| Show boundaries | where the mixture changes fastest (display) | off |

The readout gives the patch's size and **the d along its edge**: the tolerance the rule chose.

## Evidence (Xenium 2, 9 channels, r = 50 µm, 2026-10-28)

| # | Experiment | Result |
|---|---|---|
| E1 | boundary strength: 3 modes / every mode / PPCA w = 1 | median 0.80 / 2.00 / 1.91; only 3 modes gives ridges. Smoothing the leftover modes at 1r leaves them +7%, at 2r +0%: no structure ≥ r |
| E2 | θ bands vs zero-crossing lines | bands are a few r wide; lines are thin but open (cell 66% at θ = 0.5) |
| E3 | ∇²d, σ sweep 0 → 3r | cell 0.2, 3.0, 13.6, 14.6, 18.6, 21.5, 50.4%; plateau at 0.5–1r; edge d 3.2–3.6 throughout |
| E4 | + tol gate 4 → 1 | 14.4, 13.6, 4.8, 0.3, 0.2, 0.0%: a cliff at the necks |
| E5 | + opening ρ 0.5 → 2r | 8.8, 3.3, 0.3, 0.4, 0%; compact at 0.75r |

Two clicks in one window of one section. Treat the defaults as provisional.

## Next

1. Repeat E3/E5 over more windows, sections and radii: is the 0.5–1r plateau, and ρ = 0.75r, general?
2. Does w > 0 change the ∇²d patch? Untested.
3. Move the rule into the real wand (`useWand`, `wandRegion.ts`), replacing "only the patch" with it.
4. A second comparator (image channels) to check that the rule does not depend on the Gram metric.
