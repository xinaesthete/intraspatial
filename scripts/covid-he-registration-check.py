#!/usr/bin/env python3
"""Decide which vertical convention `datasources.json` uses for the H&E offsets.

    uv run --with numpy --with imagecodecs --with tifffile \
        python scripts/covid-he-registration-check.py

The cell table's `y` is bottom-up: mapping a centroid to image row `H - y` puts
99.8% of centroids inside the cellmask foreground, against 25.1% unflipped
(`covid-imagery-inventory.py` has the extents; that check is in the write-up).
The question this script settles is whether the *rasters* are placed in the same
bottom-up space, which decides the translation in every H&E affine:

    A  top edge = position_y                     (read the offset as image-space)
    B  top edge = H_roi - position_y - height    (offset is bottom-up, like y)

A and B differ by `H_roi - 2*position_y - height`, which is under 20 um for some
ROIs and over 900 um for others — so only the ROIs with a large separation can
decide it. The test sweeps a vertical shift, scoring each by the correlation
between H&E greyness and arcsinh(DNA1): nuclei are dark in H&E and bright in
DNA1, so the true alignment is the most *negative* correlation. Whichever of A
or B the peak lands on, across the ROIs that can tell them apart, is the answer.

The correlation is weak in absolute terms (~-0.1) because the two stains
disagree about a great deal besides nuclei. It is the *location of the peak*
that carries the result, not its depth.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

STRIDE = 4  # sample the ROI grid every 4 um; the features here are ~10 um
MIN_SAMPLES = 20_000


def xcorr_peak(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """Offset of `b` against `a` by FFT cross-correlation, plus a sharpness score.

    Returns `(dy, dx, z)` in the grid's own units, where `z` is the peak height
    in standard deviations above the rest of the surface. A registered pair gives
    a tall isolated peak; an unregistered one gives a flat surface and a small
    `z`, which is the distinction the 1-D sweep could not make — a sweep always
    returns *some* argmin, registered or not.
    """
    a = a - a.mean()
    b = b - b.mean()
    surface = np.fft.irfft2(np.fft.rfft2(a) * np.conj(np.fft.rfft2(b)), s=a.shape)
    flat = surface.ravel()
    idx = int(flat.argmax())
    z = (flat[idx] - flat.mean()) / (flat.std() or 1.0)
    dy, dx = np.unravel_index(idx, surface.shape)
    h, w = surface.shape
    return (dy - h if dy > h // 2 else dy), (dx - w if dx > w // 2 else dx), float(z)


def he_shape(path: Path) -> tuple[int, int]:
    """Height and width of an H&E PNG, from its IHDR — no decode needed."""
    import struct

    with open(path, "rb") as fh:
        buf = fh.read(24)
    return struct.unpack_from(">I", buf, 20)[0], struct.unpack_from(">I", buf, 16)[0]


def he_grey(path: Path) -> np.ndarray:
    import imagecodecs

    return imagecodecs.png_decode(path.read_bytes())[:, :, :3].mean(axis=2).astype(np.float32)


def dna1(path: Path) -> np.ndarray:
    import tifffile

    with tifffile.TiffFile(path) as tf:
        names = re.findall(r'Channel[^>]*Name="([^"]*)"', tf.ome_metadata or "")
        return tf.pages[names.index("DNA1")].asarray()


def stored_placement(store: Path, roi: str):
    """The H&E's scale and top-left offset as the store actually holds them.

    With `--store` the script stops being an A/B decider and becomes a quality
    report: the residual it prints is then how far each H&E still is from where
    its own pixels say it belongs.
    """
    from spatialdata import read_zarr
    from spatialdata.transformations import get_transformation

    sd = read_zarr(store)
    element = sd.images[f"{roi}_he"]
    m = get_transformation(element, roi).to_affine_matrix(
        input_axes=("x", "y"), output_axes=("x", "y")
    )
    return (m[0, 0], m[1, 1]), (m[0, 2], m[1, 2])


def measure(
    images: Path,
    entry: dict,
    he: dict,
    placement: tuple[tuple[float, float], tuple[float, float]],
) -> tuple[float, float, float] | None:
    """How far the H&E is from where its own pixels say it belongs, in micrometres.

    `placement` is the ((sx, sy), (ox, oy)) the H&E is currently drawn with;
    the returned `(dx, dy)` is what to add to `(ox, oy)` to line it up, and `z`
    says whether the peak is sharp enough to believe. Shared with the refinement
    pass so both measure the same way.
    """
    (sx, sy), (ox, oy) = placement
    g = he_grey(images / he["file"])
    hh, hw = g.shape
    d_full = dna1(images / entry["ome_tiff"])
    h_roi = float(entry["images"]["cellmask"]["height"])
    w_roi = float(entry["images"]["cellmask"]["width"])

    yy, xx = np.mgrid[0 : int(h_roi) : STRIDE, 0 : int(w_roi) : STRIDE]
    d = np.arcsinh(d_full[yy, xx] / 5.0)
    r = np.rint((yy + 0.5 - oy) / sy - 0.5).astype(int)
    c = np.rint((xx + 0.5 - ox) / sx - 0.5).astype(int)
    ok = (r >= 0) & (r < hh) & (c >= 0) & (c < hw)
    if ok.sum() < MIN_SAMPLES:
        return None
    # Nuclei are dark in H&E and bright in DNA1, so negate to make it a maximum.
    v = -g[np.clip(r, 0, hh - 1), np.clip(c, 0, hw - 1)]
    v = np.where(ok, v, v[ok].mean())

    dy, dx, z = xcorr_peak(d, v)
    return float(dx * STRIDE), float(dy * STRIDE), z


def main(covid_dir: Path, store: Path | None) -> None:
    ds = json.loads((covid_dir / "datasources.json").read_text())
    regions = next(s for s in ds if s["name"] == "cells")["regions"]["all_regions"]
    images = covid_dir / "images"

    header = (
        f"{'ROI':<24} {'residual dx':>12} {'residual dy':>12} {'z':>6}"
        if store is not None
        else f"{'ROI':<24} {'sep':>6} {'peak dx':>8} {'peak dy':>8} {'z':>6} {'A err':>7} {'B err':>7}"
    )
    print(header)
    rows = []
    for name, entry in sorted(regions.items()):
        im = entry["images"]
        he = im.get("he") or im.get("un")
        if he is None:
            continue
        h_roi = float(im["cellmask"]["height"])
        if store is not None:
            placement = stored_placement(store, name)
        else:
            # Hypothesis A, so the measured offset IS how far off A is.
            hh, hw = he_shape(images / he["file"])
            placement = (
                (he["width"] / hw, he["height"] / hh),
                (float(he["position"][0]), float(he["position"][1])),
            )
        measured = measure(images, entry, he, placement)
        if measured is None:
            print(f"{name:<24} (no usable overlap)")
            continue
        dx, dy, z = measured

        # A puts the top edge at position_y, so dy = 0; B shifts it by this much.
        oy = placement[1][1]
        b_dy = (h_roi - oy - he["height"]) - oy
        rows.append((abs(b_dy), abs(dy), abs(dy - b_dy), z, name, max(abs(dx), abs(dy))))
        if store is not None:
            flag = "" if max(abs(dx), abs(dy)) <= 30 else ("   <-- offset" if z >= 6 else "   <-- flat, likely unregistered in the source")
            print(f"{name:<24} {dx:>12.0f} {dy:>12.0f} {z:>6.1f}{flag}")
        else:
            print(
                f"{name:<24} {abs(b_dy):>6.0f} {dx:>8.0f} {dy:>8.0f} {z:>6.1f} "
                f"{abs(dy):>7.0f} {abs(dy - b_dy):>7.0f}"
            )

    if store is not None:
        aligned = sum(1 for r in rows if r[5] <= 30)
        refinable = sum(1 for r in rows if r[5] > 30 and r[3] >= 6)
        flat = sum(1 for r in rows if r[5] > 30 and r[3] < 6)
        print(
            f"\n{aligned} of {len(rows)} H&E sit within 30 um of where their own pixels say "
            f"they belong.\n{refinable} are offset but have a sharp enough peak to correct; "
            f"{flat} have a flat correlation surface and look unregistered in the source."
        )
        return

    # Only ROIs whose H&E is actually registered, AND where A and B predict
    # different things, can decide between them. Agreement elsewhere is not
    # evidence, and a flat correlation surface is not a measurement.
    for zcut in (6.0, 10.0):
        decisive = [r for r in rows if r[0] >= 100 and r[3] >= zcut]
        if not decisive:
            continue
        a_wins = sum(1 for r in decisive if r[1] < r[2])
        print(
            f"\nseparation >= 100 um and z >= {zcut}: {len(decisive)} ROIs, "
            f"A closer on {a_wins}, B on {len(decisive) - a_wins}   "
            f"median |err| A {np.median([r[1] for r in decisive]):.0f} um, "
            f"B {np.median([r[2] for r in decisive]):.0f} um"
        )
    weak = [r[4] for r in rows if r[3] < 6.0]
    print(f"\nunregistered / flat correlation surface (z < 6): {len(weak)} — {weak}")


if __name__ == "__main__":
    main(
        Path(sys.argv[1] if len(sys.argv) > 1 else "/Volumes/Crucial X8/covid"),
        Path(sys.argv[2]) if len(sys.argv) > 2 else None,
    )
