#!/usr/bin/env python3
"""Inventory the COVID imagery ahead of the SpatialData conversion.

Resolves every ROI in `datasources.json` to the files on disk, reads each one's
real extent from its header, and derives the per-image affine. Nothing here
writes; it is the input to `covid-imagery-to-spatialdata.py` and the thing to
re-run when the survey in `docs/covid-imagery-to-spatialdata-plan.md` needs
checking against the volume.

    uv run --with tifffile --with numpy --with imagecodecs \
        python scripts/covid-imagery-inventory.py "/Volumes/Crucial X8/covid"

Two facts the extension hides, both established in the plan doc and re-asserted
here rather than assumed:

- `*.ome.png` are LZW OME-TIFFs (`II*\\0`), not PNGs. Dispatch on content.
- `he` is called `un` on the HEALTHY samples, and is missing entirely on two
  ROIs. Every image key is optional.
"""

from __future__ import annotations

import json
import re
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path

# The `images` keys kept by the plan doc. Everything else there is a rendering
# of channels the 49-channel stack already holds, and is dropped.
HE_KEYS = ("he", "un")
MASK_KEY = "cellmask"


@dataclass
class Raster:
    key: str
    path: Path
    width_px: int
    height_px: int
    # micrometres, from datasources.json: the MDV `position`/`width`/`height`
    # triple, which is an axis-aligned box rather than a general transform.
    offset_um: tuple[float, float]
    extent_um: tuple[float, float]
    dtype: str = ""
    channels: int = 1
    channel_names: list[str] = field(default_factory=list)
    stored_bytes: int = 0

    @property
    def scale_um_px(self) -> tuple[float, float]:
        return (self.extent_um[0] / self.width_px, self.extent_um[1] / self.height_px)


@dataclass
class Roi:
    name: str
    stack: Raster | None = None
    he: Raster | None = None
    mask: Raster | None = None
    roi_box: dict | None = None
    # The ROI's analysis grid height in micrometres — the cellmask's extent, and
    # the height the cell table's bottom-up `y` is measured against.
    height_um: float | None = None


def _u(buf: bytes, off: int, fmt: str) -> int:
    return struct.unpack_from(fmt, buf, off)[0]


def png_header(path: Path) -> tuple[int, int, str]:
    """Width, height, dtype from a real PNG's IHDR."""
    with open(path, "rb") as fh:
        buf = fh.read(33)
    if buf[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"{path.name}: not a PNG (magic {buf[:4]!r})")
    w, h = _u(buf, 16, ">I"), _u(buf, 20, ">I")
    return w, h, f"uint{buf[24]}"


def tiff_stack_header(path: Path) -> tuple[int, int, int, str, list[str]]:
    """Extent, plane count, dtype and channel names of an OME-TIFF stack.

    Read with `tifffile` because the OME XML is where the channel names live,
    and those are real metadata: the panel mixes calibration channels
    (`80ArAr`, `131Xe`, ...) in with the antibodies, so channel index is not
    marker index.
    """
    import tifffile

    with tifffile.TiffFile(path) as tf:
        page = tf.pages[0]
        names = re.findall(r'Channel[^>]*Name="([^"]*)"', tf.ome_metadata or "")
        return (
            int(page.imagewidth),
            int(page.imagelength),
            len(tf.pages),
            str(page.dtype),
            names,
        )


def load(covid_dir: Path) -> tuple[list[Roi], list[str]]:
    sources = json.loads((covid_dir / "datasources.json").read_text())
    cells = next(s for s in sources if s["name"] == "cells")
    regions = cells["regions"]
    base = covid_dir / regions.get("base_url", "images/")
    # scale_unit mm with scale 0.001 => datasources positions are micrometres.
    assert regions["scale_unit"] == "mm" and regions["scale"] == 0.001, regions

    rois: list[Roi] = []
    problems: list[str] = []

    for name, entry in regions["all_regions"].items():
        roi = Roi(name=name, roi_box=entry.get("roi"))
        images = entry.get("images", {})

        stack_file = entry.get("ome_tiff") or entry.get("viv_image", {}).get("file")
        if stack_file:
            p = base / stack_file
            if p.exists():
                w, h, n, dtype, names = tiff_stack_header(p)
                box = entry.get("roi") or {}
                # The stack IS the reference grid: the ROI box is stated in the
                # same micrometre units and matches it 1:1 where it is present.
                ex = (
                    float(box.get("max_x", w) - box.get("min_x", 0)),
                    float(box.get("max_y", h) - box.get("min_y", 0)),
                )
                roi.stack = Raster(
                    key="imc",
                    path=p,
                    width_px=w,
                    height_px=h,
                    offset_um=(float(box.get("min_x", 0)), float(box.get("min_y", 0))),
                    extent_um=ex,
                    dtype=dtype,
                    channels=n,
                    channel_names=names,
                    stored_bytes=p.stat().st_size,
                )
            else:
                problems.append(f"{name}: stack {stack_file} missing")

        # The cellmask's declared extent IS the ROI's analysis grid: it agrees
        # with `roi` on all 32 (and beats it on COVID_SAMPLE_6_ROI_1, where the
        # box rounds 1086 up to 1100), and the cell table's coordinates run to
        # exactly it. Everything else is placed relative to this.
        mask_meta = images.get(MASK_KEY)
        roi.height_um = float(mask_meta["height"]) if mask_meta else None

        for key in HE_KEYS:
            if key in images:
                roi.he = _flat(base, key, images[key], problems, name, roi.height_um)
                break

        if mask_meta:
            roi.mask = _flat(base, MASK_KEY, mask_meta, problems, name, roi.height_um)

        rois.append(roi)

    return rois, problems


def _flat(
    base: Path, key: str, meta: dict, problems: list[str], roi: str, roi_height_um: float | None
) -> Raster | None:
    p = base / meta["file"]
    if not p.exists():
        problems.append(f"{roi}: {key} {meta['file']} missing")
        return None
    try:
        w, h, dtype = png_header(p)
    except ValueError as exc:
        problems.append(f"{roi}: {exc}")
        return None
    pos = meta.get("position", [0, 0])
    extent = (float(meta["width"]), float(meta["height"]))
    # MDV's y axis points UP; NGFF's points down. Every y in datasources.json —
    # and in the cell table — is measured from the bottom of the ROI, so the
    # image-space top edge of a box at `position_y` with height `h` is
    # `roi_height - position_y - h`. Established by cross-correlating each H&E
    # against DNA1: see `scripts/covid-he-registration-check.py`. Reading the
    # offset as image-space instead misplaces 19 of the 30 H&E images, by up to
    # 544 um, and nothing about the result looks wrong.
    top = float(pos[1]) if roi_height_um is None else roi_height_um - float(pos[1]) - extent[1]
    return Raster(
        key=key,
        path=p,
        width_px=w,
        height_px=h,
        offset_um=(float(pos[0]), top),
        extent_um=extent,
        dtype=dtype,
        stored_bytes=p.stat().st_size,
    )


def main(covid_dir: Path) -> None:
    rois, problems = load(covid_dir)
    print(f"{len(rois)} ROIs from datasources.json\n")

    print(f"{'ROI':<24} {'stack':<22} {'he':<26} {'mask':<14}")
    for r in sorted(rois, key=lambda r: r.name):
        s = f"{r.stack.width_px}x{r.stack.height_px}x{r.stack.channels}" if r.stack else "-"
        if r.he:
            sx, sy = r.he.scale_um_px
            he = f"{r.he.key}:{r.he.width_px}x{r.he.height_px}@{sx:.4f}"
        else:
            he = "MISSING"
        m = f"{r.mask.width_px}x{r.mask.height_px}" if r.mask else "MISSING"
        print(f"{r.name:<24} {s:<22} {he:<26} {m:<14}")

    stacks = [r.stack for r in rois if r.stack]
    hes = [r.he for r in rois if r.he]
    masks = [r.mask for r in rois if r.mask]

    raw = sum(s.width_px * s.height_px * s.channels * 4 for s in stacks)
    print(f"\nstacks {len(stacks):>3}  stored {sum(s.stored_bytes for s in stacks)/1e9:>6.2f} GB"
          f"  raw float32 {raw/1e9:>6.2f} GB")
    print(f"he     {len(hes):>3}  stored {sum(h.stored_bytes for h in hes)/1e6:>6.1f} MB")
    print(f"mask   {len(masks):>3}  stored {sum(m.stored_bytes for m in masks)/1e6:>6.1f} MB")

    extents = sorted({(s.width_px, s.height_px) for s in stacks})
    print(f"\nstack extents ({len(extents)} distinct): {extents}")
    chans = {s.channels for s in stacks}
    print(f"stack channel counts: {chans}")
    dtypes = {s.dtype for s in stacks}
    print(f"stack dtypes: {dtypes}")

    ref = stacks[0].channel_names
    same = all(s.channel_names == ref for s in stacks)
    print(f"channel names identical across all stacks: {same}")
    if not same:
        for s in stacks:
            if s.channel_names != ref:
                print(f"  differs: {s.path.name} {s.channel_names}")
    print(f"panel: {ref}")

    scales = sorted({round(h.scale_um_px[0], 4) for h in hes})
    print(f"\nH&E um/px values ({len(scales)} distinct): {scales}")
    offs = sorted({h.offset_um for h in hes})
    print(f"H&E distinct offsets: {len(offs)}")

    # Anisotropy matters: a single `scale` transformation cannot express it, so
    # if any of these is more than rounding the affine needs separate sx, sy.
    print(f"\n{'ROI':<24} {'px':>12} {'extent um':>19} {'offset um':>16} {'sx':>8} {'sy':>8} {'aniso':>7}")
    for r in sorted(rois, key=lambda r: r.name):
        h = r.he
        if not h:
            continue
        sx, sy = h.scale_um_px
        print(
            f"{r.name:<24} {h.width_px:>5}x{h.height_px:<6} "
            f"{h.extent_um[0]:>9.1f}x{h.extent_um[1]:<9.1f} "
            f"{h.offset_um[0]:>7.1f},{h.offset_um[1]:<8.1f} {sx:>8.5f} {sy:>8.5f} "
            f"{abs(sx - sy) / sx * 100:>6.2f}%"
        )

    mismatch = [
        r.name for r in rois
        if r.stack and r.mask and (r.mask.width_px, r.mask.height_px) != (r.stack.width_px, r.stack.height_px)
    ]
    print(f"\ncellmask extent != stack extent: {mismatch or 'none'}")

    if problems:
        print("\nPROBLEMS")
        for p in problems:
            print(f"  {p}")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "/Volumes/Crucial X8/covid"))
