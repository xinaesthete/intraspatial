# Wavelets & the DWT — interactive primer

A self-contained, dependency-free visualization that teaches the discrete wavelet
transform (DWT) behind JPEG 2000 / HTJ2K, in service of the `tgpu-htj2k` WebGPU
codec learning project. No build step, no npm install, no CDN — everything runs
from plain `<script>` tags.

## How to view

Either:

- **Just open it.** Double-click `viz/index.html` (or `open viz/index.html`).
- **Or serve it** (recommended; avoids any `file://` quirks with image upload):

  ```sh
  cd viz
  python3 -m http.server 8000
  # then visit http://localhost:8000/
  ```

It looks complete on first load with zero clicks (defaults: Mandelbrot slice,
9/7 kernel, 4 levels, ~8% coefficients kept).

## Files

| File         | Purpose |
|--------------|---------|
| `index.html` | Layout, dark-theme CSS, and the woven written explainer (collapsible). |
| `dwt.js`     | The DWT itself — 5/3 reversible (integer) + 9/7 irreversible (float) lifting, separable 2D forward/inverse, 1D transform, and a round-trip self-test. Thoroughly commented; doubles as a reference. |
| `images.js`  | Synthetic test-image generators (radial gradient, rings/edges, sinusoid grating, smooth+noise, Mandelbrot slice) and the upload→grayscale helper. |
| `app.js`     | Canvas rendering and wiring for the three interactive panels. |

## What each panel teaches

1. **Subband decomposition explorer.** Renders the classic Mallat pyramid: the
   top-left **LL** is the coarse approximation; **HL/LH/HH** hold horizontal,
   vertical and diagonal detail at each level. Choose the kernel (5/3 vs 9/7),
   1–6 levels, linear/log detail scaling and a detail-gain slider, or upload your
   own image. Click any detail subband to isolate it and read its energy stats.

2. **Reconstruction & compression demo.** Keep only the top X% of coefficients by
   magnitude, inverse-transform, and compare original / reconstructed / error with
   a live **PSNR** and **sparsity** readout. This makes *energy compaction* —
   the reason wavelets compress so well — visible and visceral.

3. **1D lifting-scheme animator.** Steps through one decomposition level of the
   lifting scheme (**split → predict → update → result**) on a selectable signal
   (step, ramp, sine, noisy), for either kernel. Watch detail coefficients collapse
   toward zero on smooth regions and spike at edges.

The page also includes a collapsible written explainer: what a wavelet is, the
lifting scheme, why detail is near-zero for natural images, multi-resolution /
scale-space intuition, and a section on the **volumetric** angle — extending the
separable transform with a wavelet along *z* across image slices for 3D
scientific data.

## Correctness

The DWT is verified to round-trip: `inverse(forward(x)) ≈ x`.

- **5/3** reconstructs integer input **bit-exactly** (max error `0`).
- **9/7** reconstructs to floating-point precision (max error ≈ `1e-13`).

This is checked across many sizes (including odd lengths and degenerate `1×N`
strips) and all 1–6 levels. A small green badge in the page header runs the
self-test live on load. Boundary handling uses whole-sample symmetric (mirror)
extension, matching the JPEG 2000 convention.

## Limitations / notes

- Grayscale (luma) only; color images are converted to luma on upload.
- Working resolution is fixed at 256×256 (resampled on upload) — comfortably
  performant while keeping the all-JS transform interactive.
- The 9/7 transform here applies the standard CDF 9/7 lifting to the whole
  plane. A production HTJ2K codec adds quantization, tiling, code-blocks,
  precincts and the HT block coder — out of scope for this teaching tool, which
  isolates the wavelet idea itself.
- The "keep top X%" thresholding is a global magnitude threshold to illustrate
  energy compaction; it is not the rate-distortion-optimal truncation a real
  codec performs.
