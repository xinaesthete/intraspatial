#!/usr/bin/env python3
"""Reproduce the tables in `docs/imc-image-compression-measurements.md`.

Python rather than TypeScript because the point is to measure the codec that
`spatialdata-js-util` actually calls, through the same binding it calls it with.
No install needed:

    uv run --with tifffile --with numpy --with imagecodecs --with numcodecs \\
        python scripts/imc-compression-measure.py stack "<path>.ome.png"

Subcommands:
    stack <ome.png>     lossless baselines + bit-depth sweep (findings 1 and 2)
    crossover <img>     where lossy overtakes lossless, per bit depth (finding 3)
    png <img>...        8-bit PNG cases: H&E and cellmask (finding 5)

Note the `.ome.png` files are LZW OME-TIFFs, not PNGs; `tifffile` reads them by
content, so the extension does not matter here, but it does to anything that
dispatches on the name.
"""

from __future__ import annotations

import re
import sys

import imagecodecs
import numcodecs
import numpy as np

BLOCK = 10
COFACTOR = 5.0

# The presets spatialdata-js-util ships (images.py HTJ2K_PRESETS), plus enough
# coarser steps to bracket the reversible/irreversible crossover.
SWEEP = (0.0002, 0.001, 0.005, 0.02, 0.05, 0.15)


def blosc(cname: str, clevel: int, shuffle: int) -> numcodecs.Blosc:
    return numcodecs.Blosc(cname=cname, clevel=clevel, shuffle=shuffle)


def block_mean(a: np.ndarray) -> np.ndarray:
    h, w = a.shape[-2], a.shape[-1]
    return a.reshape(h // BLOCK, BLOCK, w // BLOCK, BLOCK).mean(axis=(1, 3))


def asinh_quant(a: np.ndarray, levels: int, dtype) -> tuple[np.ndarray, float]:
    """arcsinh(x/5) scaled onto `levels` steps.

    The only quantiser measured that survives the dynamic range: IMC channels
    reach a 5800:1 ratio between p99.9 and max, so linear-to-max spends almost
    every code point on hot pixels (finding 4).
    """
    t = np.arcsinh(a.astype(np.float64) / COFACTOR)
    step = (float(t.max()) or 1.0) / levels
    return np.rint(t / step).astype(dtype), step


def unquant(u: np.ndarray, step: float) -> np.ndarray:
    return np.sinh(u.astype(np.float64) * step) * COFACTOR


def block_err_p99(truth_blocks: np.ndarray, live: np.ndarray, rec: np.ndarray) -> float:
    """p99 relative error of 10x10 block means, over blocks carrying signal.

    A cell is ~10 px across at 1 um/px, so this stands in for the per-cell mean
    intensity the statistics consume. Scoring every pixel would flatter any
    scheme, since most of every channel is empty.
    """
    if not live.any():
        return 0.0
    rb = block_mean(rec)
    return float(np.percentile(np.abs(rb - truth_blocks)[live] / truth_blocks[live] * 100, 99))


def live_blocks(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    b = block_mean(a.astype(np.float64))
    pos = b[b > 0]
    return b, b > (np.percentile(pos, 50) if pos.size else 0)


def channel_names(tf) -> list[str]:
    return re.findall(r'Channel[^>]*Name="([^"]*)"', tf.ome_metadata or "")


def cmd_stack(path: str) -> None:
    import tifffile

    variants = {
        "LZW (today)": None,
        "zstd-5 +shuffle": blosc("zstd", 5, numcodecs.Blosc.SHUFFLE),
        "zstd-5 noshuffle": blosc("zstd", 5, numcodecs.Blosc.NOSHUFFLE),
        "zstd-9 noshuffle": blosc("zstd", 9, numcodecs.Blosc.NOSHUFFLE),
    }
    totals = dict.fromkeys([*variants, "u16 rev", "u8 rev", "raw"], 0)
    errs: dict[str, list[float]] = {}

    with tifffile.TiffFile(path) as tf:
        names = channel_names(tf)
        print(f"{'idx':>3} {'name':<12} {'LZW':>8} {'zstd9':>8} {'u16 rev':>8} {'u8 rev':>8} {'u8 blk%':>8}")
        for i, page in enumerate(tf.pages):
            a = page.asarray()
            ac = np.ascontiguousarray(a)
            truth, live = live_blocks(a)

            totals["raw"] += a.nbytes
            totals["LZW (today)"] += int(sum(page.databytecounts))
            for label, codec in variants.items():
                if codec is not None:
                    totals[label] += len(codec.encode(ac))

            u16, s16 = asinh_quant(a, 65535, np.uint16)
            u8, s8 = asinh_quant(a, 255, np.uint8)
            c16 = len(imagecodecs.htj2k_encode(u16, reversible=True))
            c8 = len(imagecodecs.htj2k_encode(u8, reversible=True))
            totals["u16 rev"] += c16
            totals["u8 rev"] += c8
            errs.setdefault("u8", []).append(block_err_p99(truth, live, unquant(u8, s8)))

            print(
                f"{i:>3} {(names[i] if i < len(names) else '?'):<12} "
                f"{sum(page.databytecounts) / 1e6:>7.2f}M "
                f"{len(variants['zstd-9 noshuffle'].encode(ac)) / 1e6:>7.2f}M "
                f"{c16 / 1e6:>7.2f}M {c8 / 1e6:>7.2f}M {errs['u8'][-1]:>7.3f}%"
            )

    raw, lzw = totals["raw"], totals["LZW (today)"]
    print(f"\n{'raw float32':<20} {raw / 1e6:>7.1f} MB")
    for label in [*variants, "u16 rev", "u8 rev"]:
        n = totals[label]
        print(f"{label:<20} {n / 1e6:>7.1f} MB  {raw / n:>4.2f}x raw  {lzw / n:>4.2f}x LZW")
    for k, v in errs.items():
        print(f"\nblock p99 error, {k}: median {np.median(v):.3f}%  worst {max(v):.3f}%")


def cmd_crossover(path: str, channels: tuple[int, ...] = (0, 16, 33, 42)) -> None:
    import tifffile

    with tifffile.TiffFile(path) as tf:
        names = channel_names(tf)
        for ci in channels:
            a = tf.pages[ci].asarray()
            truth, live = live_blocks(a)
            print(f"\n=== {names[ci] if ci < len(names) else ci} ===")
            for bits, dtype in ((8, np.uint8), (16, np.uint16)):
                u, step = asinh_quant(a, (1 << bits) - 1, dtype)
                rev = len(imagecodecs.htj2k_encode(u, reversible=True))
                print(f"  u{bits} reversible {rev / 1e6:>7.2f}M")
                for lv in SWEEP:
                    cs = bytes(imagecodecs.htj2k_encode(u, level=float(lv), reversible=False))
                    err = block_err_p99(truth, live, unquant(np.asarray(imagecodecs.htj2k_decode(cs)), step))
                    warn = "  <-- LARGER THAN LOSSLESS" if len(cs) > rev else ""
                    print(f"    level {lv:<7g} {len(cs) / 1e6:>7.2f}M  blk p99 {err:>7.3f}%{warn}")


def cmd_png(paths: list[str]) -> None:
    for path in paths:
        raw = open(path, "rb").read()
        a = imagecodecs.png_decode(raw)
        planar = a.ndim == 3 and a.shape[2] > 1
        arr = np.ascontiguousarray(np.moveaxis(a, 2, 0)) if planar else a
        z9 = blosc("zstd", 9, numcodecs.Blosc.NOSHUFFLE)
        print(f"\n=== {path.split('/')[-1]}  {a.shape} {a.dtype}  PNG {len(raw) / 1e6:.2f}M ===")
        print(f"    distinct values     {np.unique(a).size}")
        print(f"    zstd-9              {len(z9.encode(np.ascontiguousarray(a))) / 1e6:>7.3f}M")
        rev = bytes(imagecodecs.htj2k_encode(arr, reversible=True, planar=planar, rgb=False))
        print(f"    HTJ2K reversible    {len(rev) / 1e6:>7.3f}M  {len(raw) / len(rev):>5.2f}x vs PNG")
        for lv in SWEEP:
            cs = bytes(imagecodecs.htj2k_encode(arr, level=float(lv), reversible=False, planar=planar, rgb=False))
            d = np.asarray(imagecodecs.htj2k_decode(cs, planar=planar))
            if planar:
                d = np.moveaxis(d, 0, 2)
            e = np.abs(d.astype(np.int32) - a.astype(np.int32))
            warn = "  <-- LARGER THAN LOSSLESS" if len(cs) > len(rev) else ""
            print(
                f"    HTJ2K level {lv:<7g} {len(cs) / 1e6:>7.3f}M  {len(raw) / len(cs):>5.2f}x vs PNG  "
                f"maxerr {e.max():>4} mean {e.mean():>7.4f}{warn}"
            )


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "stack":
        cmd_stack(args[0])
    elif cmd == "crossover":
        cmd_crossover(args[0])
    elif cmd == "png":
        cmd_png(args)
    else:
        sys.exit(f"unknown subcommand {cmd!r}\n{__doc__}")
