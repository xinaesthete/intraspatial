#!/usr/bin/env python3
"""Migrate the COVID project's imagery from MDV's flat files into a SpatialData zarr.

The survey is `docs/covid-imagery-to-spatialdata-plan.md`, the codec decision is
`docs/imc-image-compression-measurements.md`, and the label reconstruction is
`docs/mdv-zarr-to-spatialdata-tables.md` finding B. This script executes those
decisions; it does not revisit them.

    uv run --with spatialdata --with tifffile --with imagecodecs --with scikit-image \
        --with ~/code/www/SpatialData.ts/python/spatialdata-js-util \
        python scripts/covid-imagery-to-spatialdata.py he
    ... imc / labels / table / omero / refine / retransform / units

`spatialdata-js-util` is needed from the H&E stage onwards even though only that
stage calls it: once the H&E arrays are HTJ2K, opening the store at all needs the
codec registered, and that happens through the package's `zarr.codecs` entry
point — installing it is enough, importing it is not required.

Written to be re-runnable per stage and per ROI (`--rois`), because a full pass
decodes 22.3 GB of LZW and re-encodes it at zstd-19.

Shape of the output: **one store, one coordinate system per ROI.** The 32 ROIs
are separate tissue sections and share no space, so putting them all in `global`
would assert an overlap that does not exist; but they are one dataset and the
cell table already covers all 32 in one store, so they are not 32 stores either.
Elements are `<ROI>_imc`, `<ROI>_he`, `<ROI>_labels`.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parent


def _load_module(filename: str):
    """Import a sibling script despite the hyphenated filename."""
    path = SCRIPTS / filename
    spec = importlib.util.spec_from_file_location(path.stem.replace("-", "_"), path)
    module = importlib.util.module_from_spec(spec)
    # Register before exec: `dataclass` looks the class's module up in
    # sys.modules, and gets None (then AttributeError) if it is not there.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


inventory = _load_module("covid-imagery-inventory.py")
Raster = inventory.Raster
Roi = inventory.Roi


# --------------------------------------------------------------------------- transforms


def raster_affine(r: Raster):
    """The image's own affine into its ROI's micrometre coordinate system.

    MDV stores `position` + `width`/`height` in micrometres, which is an
    axis-aligned box rather than a transform; per-axis scale plus translation
    reproduces exactly what MDV renders, and is what SpatialData wants anyway.
    Scale is per-axis on purpose — one H&E (`COVID_SAMPLE_11_ROI_2`) has a 15%
    difference between its x and y scales, so a single isotropic `scale` would
    silently squash it.
    """
    from spatialdata.transformations import Affine

    sx, sy = r.scale_um_px
    tx, ty = r.offset_um
    return Affine(
        np.array([[sx, 0.0, tx], [0.0, sy, ty], [0.0, 0.0, 1.0]]),
        input_axes=("x", "y"),
        output_axes=("x", "y"),
    )


# --------------------------------------------------------------------------- store


def attach(store: Path):
    """A SpatialData pointed at `store` but holding none of its elements.

    Deliberately not `read_zarr`: an element read back from the store is
    *backed by* it, and spatialdata then refuses both to overwrite it and to
    delete it, so a re-run of any stage is impossible (discussion #520). A
    detached object writing elements one at a time has neither problem, and the
    source of truth for everything written here is the external volume anyway.
    """
    from spatialdata import SpatialData

    if not store.exists():
        SpatialData().write(store)
    sd = SpatialData()
    sd.path = store
    return sd


def set_spatial_unit(store: Path, name: str, kind: str, unit: str = "micrometer") -> None:
    """Replace spatialdata's placeholder axis unit with the real one.

    `spatialdata.transformations.ngff._utils` hardcodes `unit="unit"` on the x,
    y and z axes and says so itself — "not supported by NGFF so the user should
    replace it before saving" — with no per-element API to set it. MDV's
    `scale_unit: mm` / `scale: 0.001` means these coordinates are micrometres,
    which is worth recording rather than leaving as a placeholder.
    """
    import json

    path = store / kind / name / "zarr.json"
    meta = json.loads(path.read_text())
    ome = meta.get("attributes", {}).get("ome", {})
    changed = False
    for multiscale in ome.get("multiscales", []):
        for transform in multiscale.get("coordinateTransformations", []):
            for side in ("input", "output"):
                for axis in transform.get(side, {}).get("axes", []):
                    if axis.get("type") == "space" and axis.get("unit") != unit:
                        axis["unit"] = unit
                        changed = True
    if changed:
        path.write_text(json.dumps(meta))


def write_one(sd, name: str, element, kind: str, compressor):
    """Write (or rewrite) one element.

    `write_element(overwrite=True)` refuses when the target is already
    on disk, so a re-run has to delete first. That is safe here in a way
    spatialdata cannot assume: the source of truth is the OME-TIFF / PNG on the
    external volume, so an interrupted rewrite loses nothing a re-run does not
    restore.
    """
    getattr(sd, kind)[name] = element
    if (sd.path / kind / name).exists():
        sd.delete_element_from_disk(name)
    sd.write_element(name, raster_compressor=compressor)


def finalize(sd, store: Path, written: list[tuple[str, str]]) -> None:
    """Fix up axis units, then re-consolidate — in that order.

    The store carries consolidated metadata, so editing an element's `zarr.json`
    after consolidation leaves a stale copy at the root that readers prefer.
    """
    for name, kind in written:
        set_spatial_unit(store, name, kind)
    if written:
        sd.write_consolidated_metadata()


def element_bytes(store: Path, name: str, kind: str) -> int:
    root = store / kind / name
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file())


# --------------------------------------------------------------------------- H&E


def read_he(r: Raster) -> tuple[np.ndarray, tuple[int, int] | None]:
    """Decode an H&E PNG to `(c, y, x)` uint8 RGB, dropping a constant alpha.

    These really are PNGs — unlike the `*.ome.png` stacks, which are TIFFs.
    """
    import imagecodecs

    a = imagecodecs.png_decode(r.path.read_bytes())
    if a.ndim == 2:
        a = a[:, :, None]
    dropped = None
    if a.shape[2] == 4:
        alpha = a[:, :, 3]
        lo, hi = int(alpha.min()), int(alpha.max())
        dropped = (lo, hi)
        a = a[:, :, :3]
    return np.ascontiguousarray(np.moveaxis(a, 2, 0)), dropped


def cmd_he(args, rois: list[Roi]) -> None:
    from spatialdata.models import Image2DModel

    sd = attach(args.store)
    written: list[tuple[str, str]] = []
    total_src = total_dst = 0

    for roi in rois:
        if roi.he is None:
            print(f"{roi.name:<24} no H&E — skipped (expected on 2 of 32)")
            continue
        arr, alpha_range = read_he(roi.he)
        if alpha_range and alpha_range != (255, 255):
            # The plan asserts alpha is constant 255. If it is not, dropping it
            # loses something, so say so rather than quietly proceeding.
            print(f"  WARNING {roi.he.path.name}: alpha range {alpha_range}, not constant 255")

        name = f"{roi.name}_he"
        img = Image2DModel.parse(
            arr,
            dims=("c", "y", "x"),
            c_coords=["r", "g", "b"][: arr.shape[0]],
            chunks=(1, 1024, 1024),
            transformations={roi.name: raster_affine(roi.he)},
        )
        write_one(sd, name, img, "images", {"zstd": args.zstd})
        written.append((name, "images"))

        src = roi.he.stored_bytes
        dst = element_bytes(args.store, name, "images")
        total_src += src
        total_dst += dst
        sx, sy = roi.he.scale_um_px
        print(
            f"{roi.name:<24} {arr.shape[2]}x{arr.shape[1]} "
            f"@{sx:.4f},{sy:.4f} um/px  PNG {src / 1e6:>6.1f} MB -> {dst / 1e6:>6.1f} MB"
        )

    finalize(sd, args.store, written)

    if total_src:
        print(f"\nH&E total: {total_src / 1e6:.1f} MB PNG -> {total_dst / 1e6:.1f} MB zarr")
        print("Next: `spatialdata-js-util images recompress --codec experimental.openjph_htj2k "
              "--preset lossless --pyramid` — HTJ2K beats PNG losslessly here (1.31x) and the "
              "pyramid matters more than the codec, since nothing in this dataset is pyramidal.")


# --------------------------------------------------------------------------- IMC stack


def imc_dask(r: Raster):
    """A lazy `(c, y, x)` float32 array, one chunk per channel.

    One channel per chunk is the measured decision (findings 8 and 9): chunking
    channels together is flat to three digits because inter-channel correlation
    is ~0.016, and one-per-chunk is what keeps single-channel random access in a
    browser. Lazy so the 784 MB stack is never resident all at once.
    """
    import dask
    import dask.array as da
    import tifffile

    def page(i: int) -> np.ndarray:
        with tifffile.TiffFile(r.path) as tf:
            return tf.pages[i].asarray()

    return da.stack(
        [
            da.from_delayed(dask.delayed(page)(i), shape=(r.height_px, r.width_px), dtype=np.float32)
            for i in range(r.channels)
        ]
    )


def subsample_levels(img, base, levels: int) -> None:
    """Replace spatialdata's coarsen-mean pyramid levels with exact 2x subsampling.

    The source stacks are ALREADY 2-level pyramids — `tf.pages` reports 49
    planes but `tf.series[0].levels` reports two, and the hidden level is 20% of
    the file. That corrects plan open question 2 ("nothing here is pyramidal
    today"), and it means a level 1 here is not an invention.

    The source's level 1 is exactly `level0[::2, ::2]` (checked on three
    channels, max abs difference 0). Reproducing that rather than taking
    spatialdata's default coarsen-mean is worth 1.67x on this data — 33.8 MB
    against 56.4 MB per ROI, measured — because averaging four float32s
    manufactures bit patterns that were not in the data and so destroys the
    repetitive-symbolic structure zstd is exploiting. Same mechanism as finding 2
    of the compression measurements, arriving from a different direction.

    A block mean would be the better *statistic*, but level 1 is a viewing level:
    every number the analysis consumes is read from level 0.

    `to_multiscale` is left to build the tree so the coordinates and per-level
    transformations stay spatialdata's own; only the pixels are swapped, sliced
    to the shape coarsen already chose (`floor(n/2)`, i.e. boundary-trim).
    """
    for k in range(1, levels):
        step = 2**k
        node = img[f"scale{k}"]["image"]
        h, w = node.sizes["y"], node.sizes["x"]
        img[f"scale{k}"]["image"] = node.copy(
            data=base[:, : h * step : step, : w * step : step]
        )


def cmd_imc(args, rois: list[Roi]) -> None:
    from spatialdata.models import Image2DModel

    sd = attach(args.store)
    written: list[tuple[str, str]] = []
    total_src = total_dst = total_raw = 0

    for roi in rois:
        if roi.stack is None:
            print(f"{roi.name:<24} no stack — skipped")
            continue
        r = roi.stack
        name = f"{roi.name}_imc"
        base = imc_dask(r)
        img = Image2DModel.parse(
            base,
            dims=("c", "y", "x"),
            c_coords=list(r.channel_names),
            chunks=(1, r.height_px, r.width_px),
            transformations={roi.name: raster_affine(r)},
            scale_factors=[2] * (args.imc_levels - 1) or None,
        )
        subsample_levels(img, base, args.imc_levels)
        # zstd 19 is the knee: level 22 costs the same time for 0.2% less, and
        # levels 12/15 are worse than Blosc's clevel=9 (finding 9). Never
        # byte-shuffle — the zarr v3 path spatialdata takes does not (finding 1).
        write_one(sd, name, img, "images", {"zstd": args.zstd})
        written.append((name, "images"))

        raw = r.width_px * r.height_px * r.channels * 4
        src = r.stored_bytes
        dst = element_bytes(args.store, name, "images")
        total_raw += raw
        total_src += src
        total_dst += dst
        print(
            f"{roi.name:<24} {r.width_px}x{r.height_px}x{r.channels}  "
            f"raw {raw / 1e6:>7.1f}  LZW {src / 1e6:>7.1f}  zstd-{args.zstd} {dst / 1e6:>7.1f} MB"
            f"  ({src / dst:.2f}x LZW)"
        )

    finalize(sd, args.store, written)

    if total_src:
        print(
            f"\nIMC total: raw {total_raw / 1e9:.2f} GB, LZW {total_src / 1e9:.2f} GB, "
            f"zstd-{args.zstd} {total_dst / 1e9:.2f} GB ({total_src / total_dst:.2f}x LZW)"
        )


# --------------------------------------------------------------------------- cell table


class Cells:
    """The MDV cell table, read from `datafile.h5` rather than from our zarr copy.

    `~/data/covid.mdv.zarr` has `cellID` written as 18 MB of zeros — the h5-to-zarr
    converter reads columns through `h5dump -b LE`, which emits nothing at all for
    `H5T_STRING`, and only the numeric branch checks the length it got back
    (`docs/mdv-zarr-to-spatialdata-tables.md` finding A). `cellID` is the
    `instance_key`, so this stage cannot use that copy; the h5 is fine.
    """

    #: MDV categoricals are a uint8 code array plus a `values` list in
    #: datasources.json, so the codes alone are meaningless without it.
    def __init__(self, covid_dir: Path):
        import h5py

        sources = json.loads((covid_dir / "datasources.json").read_text())
        spec = next(s for s in sources if s["name"] == "cells")
        self.columns = {c["field"]: c for c in spec["columns"]}
        self.h5 = h5py.File(covid_dir / "datafile.h5", "r")["cells"]
        self.n = int(self.h5["x"].shape[0])

        self.sample_ids = self.columns["sample_id"]["values"]
        self.sample_code = self.col("sample_id")
        cell_id = self.h5["cellID"][:]
        if not cell_id.any():
            raise SystemExit("cellID is empty in the h5 too — nothing to key labels by")
        self.cell_id = np.array([s.decode() for s in cell_id])
        # `<SAMPLE>_<ROI>_CELL_<n>`, with n restarting per ROI and NOT contiguous:
        # COVID_SAMPLE_16_ROI_3 has 32,569 cells numbered up to 34,878, so 2,310
        # suffixes belong to cells the published table dropped.
        self.instance = np.array(
            [int(s.rsplit("_CELL_", 1)[1]) for s in self.cell_id], dtype=np.int64
        )

    def col(self, field: str) -> np.ndarray:
        return self.h5[field][:]

    def values(self, field: str) -> list[str] | None:
        return self.columns[field].get("values")

    def rows_for(self, roi_name: str) -> np.ndarray:
        return np.flatnonzero(self.sample_code == self.sample_ids.index(roi_name))


def centroid_pixels(cells: Cells, rows: np.ndarray, roi: Roi) -> tuple[np.ndarray, np.ndarray]:
    """Centroid row/col in image space, applying the y flip.

    The cell table's `y` is measured from the BOTTOM of the ROI. Sampling the
    cellmask at `row = H - y` puts 99.8% of centroids on foreground (99.5-100%
    across all 32 ROIs); reading `y` as a row directly gives 25.1%, which is
    exactly the foreground fraction — i.e. pure chance, and it looks like a
    segmentation that disagrees with its own centroids rather than a flip.
    `H - 1 - y` gives 95.7%, so the flip is over the extent, not the last index.
    """
    h_um = roi.height_um
    col = np.floor(cells.col("x")[rows]).astype(np.int64)
    row = np.floor(h_um - cells.col("y")[rows]).astype(np.int64)
    return row, col


# --------------------------------------------------------------------------- labels


def read_mask(r: Raster) -> np.ndarray:
    import imagecodecs

    a = imagecodecs.png_decode(r.path.read_bytes())
    return a if a.ndim == 2 else a[:, :, 0]


def watershed_labels(mask: np.ndarray, row: np.ndarray, col: np.ndarray, instance: np.ndarray):
    """Seeded watershed: binary foreground + one seed per cell -> uint16 instance ids.

    The elevation is the negated euclidean distance transform, so basins grow
    from each seed and meet at the narrow waists between cells.

    **In practice the watershed does nothing here, and that is the good news.**
    The plan expected touching cells to merge, since the mask is ~41%
    foreground — but `multi_seed_blobs` comes back 0 on every ROI: the stored
    mask already separates every cell with a gap, so each connected component
    contains exactly one centroid and the basin it grows into is the whole
    component. The labels are therefore the *original* per-cell masks recovered
    exactly, not an approximation of them. The watershed is kept because it
    costs about a second per ROI and degrades gracefully if that ever stops
    being true — `multi_seed_blobs` is what would say so.

    Corroboration: the surplus components (34,803 against 32,569 cells in
    COVID_SAMPLE_16_ROI_3) match the gaps in the `_CELL_<n>` numbering — cells
    the published table filtered out but the mask still draws.

    Label 0 is background, so the stored value is `instance + 1`: the `_CELL_<n>`
    suffixes start at 0 in some ROIs and a raw suffix would make those cells
    invisible. Global max suffix is 34,878, so +1 still fits uint16 with room.
    """
    from scipy import ndimage
    from skimage.segmentation import watershed

    fg = mask > 0
    h, w = fg.shape
    inside = (row >= 0) & (row < h) & (col >= 0) & (col < w)
    on_fg = np.zeros(len(row), dtype=bool)
    on_fg[inside] = fg[row[inside], col[inside]]

    markers = np.zeros(fg.shape, dtype=np.int32)
    keep = np.flatnonzero(on_fg)
    # Later writes win where two centroids share a pixel; count them instead of
    # letting one cell silently absorb another.
    markers[row[keep], col[keep]] = instance[keep] + 1
    seeded = int((markers > 0).sum())

    distance = ndimage.distance_transform_edt(fg)
    labels = watershed(-distance, markers, mask=fg).astype(np.uint16)

    components, n_components = ndimage.label(fg)
    seeds_per_blob = np.bincount(
        components[row[keep], col[keep]], minlength=n_components + 1
    )[1:]
    covered = np.unique(labels)
    covered = covered[covered > 0]
    return labels, {
        "cells": len(row),
        "off_foreground": int(len(row) - on_fg.sum()),
        "collided_seeds": int(on_fg.sum() - seeded),
        "labels_present": len(covered),
        "blob_components": int(n_components),
        # 0 everywhere means the mask already separates cells and the watershed
        # is recovering the original segmentation rather than guessing at it.
        "multi_seed_blobs": int((seeds_per_blob >= 2).sum()),
        "foreground_pct": float(fg.mean() * 100),
    }


def cmd_labels(args, rois: list[Roi]) -> None:
    from spatialdata.models import Labels2DModel

    cells = Cells(args.covid_dir)
    sd = attach(args.store)
    written: list[tuple[str, str]] = []

    print(
        f"{'ROI':<24} {'cells':>7} {'labelled':>9} {'off-fg':>7} {'collide':>8} "
        f"{'blobs':>7} {'merged':>7} {'fg%':>6}"
    )
    totals = {}
    for roi in rois:
        if roi.mask is None:
            print(f"{roi.name:<24} no cellmask — skipped")
            continue
        rows = cells.rows_for(roi.name)
        row, col = centroid_pixels(cells, rows, roi)
        mask = read_mask(roi.mask)
        labels, stats = watershed_labels(mask, row, col, cells.instance[rows])

        name = f"{roi.name}_labels"
        element = Labels2DModel.parse(
            labels,
            dims=("y", "x"),
            chunks=(1024, 1024),
            transformations={roi.name: raster_affine(roi.mask)},
        )
        # Labels never go through an image codec: HTJ2K expands this data 8x
        # (0.39 MB PNG -> 3.21 MB reversible). zstd is both smaller and exact,
        # and exactness is not optional — an interpolated label id is a cell
        # that does not exist.
        write_one(sd, name, element, "labels", {"zstd": args.zstd})
        written.append((name, "labels"))

        for k, v in stats.items():
            totals[k] = totals.get(k, 0) + v
        print(
            f"{roi.name:<24} {stats['cells']:>7} {stats['labels_present']:>9} "
            f"{stats['off_foreground']:>7} {stats['collided_seeds']:>8} "
            f"{stats['blob_components']:>7} {stats['multi_seed_blobs']:>7} "
            f"{stats['foreground_pct']:>5.1f}%"
        )

    finalize(sd, args.store, written)
    if totals:
        # The oracle: the table says how many cells each ROI has, so a watershed
        # that over- or under-segments says so per ROI without anyone looking at
        # a picture.
        print(
            f"\ntotal cells {totals['cells']}, labelled {totals['labels_present']} "
            f"({totals['labels_present'] / totals['cells'] * 100:.2f}%), "
            f"off-foreground {totals['off_foreground']}, seed collisions "
            f"{totals['collided_seeds']}, blobs holding two or more cells "
            f"{totals['multi_seed_blobs']}"
        )


# --------------------------------------------------------------------------- table


#: The four 3-D embeddings hiding among the 36 non-measurement columns. Flattened
#: into `obs` they stop being embeddings and become invisible to every tool that
#: looks in `obsm`.
EMBEDDINGS = {
    "X_umap": ("UMAP_1", "UMAP_2", "UMAP_3"),
    "X_umap_lymphocyte": ("lymphocyte_UMAP1", "lymphocyte_UMAP2", "lymphocyte_UMAP3"),
    "X_umap_myeloid": ("myeloid_UMAP1", "myeloid_UMAP2", "myeloid_UMAP3"),
    "X_umap_structural": ("structural_UMAP1", "structural_UMAP2", "structural_UMAP3"),
}


def build_table(cells: Cells, rois: list[Roi], panel: list[str]):
    """Assemble the AnnData: X by panel intersection, embeddings to obsm, rest to obs."""
    import anndata as ad
    import pandas as pd

    markers = [c for c in panel if c in cells.columns]
    embedded = {c for cols in EMBEDDINGS.values() for c in cols}
    obs_fields = [
        f
        for f in cells.columns
        if f not in markers and f not in embedded and f not in {"x", "y", "cellID"}
    ]

    X = np.empty((cells.n, len(markers)), dtype=np.float32)
    for i, m in enumerate(markers):
        X[:, i] = cells.col(m)

    obs = pd.DataFrame(index=pd.Index(cells.cell_id, name="cellID"))
    for f in obs_fields:
        values = cells.values(f)
        raw = cells.col(f)
        if values is not None:
            # MDV stores a categorical as uint8 codes plus a `values` list; AnnData
            # wants a real categorical group. Note MDV's `NA` is category 0, a
            # genuine annotation here, not AnnData's -1 "missing" — keeping it as a
            # category is deliberate, and it is why the codes stay unsigned.
            obs[f] = pd.Categorical.from_codes(raw.astype(np.int16), categories=values)
        else:
            obs[f] = raw

    # The join to the labels rasters. `region` must name elements that exist, and
    # `instance_id` must be the pixel value: `_CELL_<n>` + 1, since label 0 is
    # background and some ROIs number their first cell 0.
    obs["region"] = pd.Categorical(
        [f"{s}_labels" for s in np.asarray(cells.sample_ids, dtype=object)[cells.sample_code]]
    )
    obs["instance_id"] = (cells.instance + 1).astype(np.uint16)

    by_name = {r.name: r for r in rois}
    heights = np.array(
        [by_name[s].height_um for s in np.asarray(cells.sample_ids, dtype=object)[cells.sample_code]],
        dtype=np.float64,
    )
    spatial = np.empty((cells.n, 2), dtype=np.float32)
    spatial[:, 0] = cells.col("x")
    # Same flip as the labels: MDV's y is bottom-up, NGFF's is top-down.
    spatial[:, 1] = heights - cells.col("y")

    obsm = {"spatial": spatial}
    for key, cols in EMBEDDINGS.items():
        obsm[key] = np.stack([cells.col(c) for c in cols], axis=1).astype(np.float32)

    adata = ad.AnnData(
        X=X,
        obs=obs,
        var=pd.DataFrame(index=pd.Index(markers, name="marker")),
        obsm=obsm,
    )
    return adata, markers


def coverage(store: Path, adata, regions: list[str]) -> np.ndarray:
    """Which rows actually have pixels, read back from the written labels.

    Not every cell gets a region: a centroid can land outside the mask's
    foreground (957 of 545,400 do). SpatialData has no way to say "this row
    annotates nothing", and the Xenium store has the same shape — its
    `nucleus_boundaries` covers 23 of 36 cells — so the honest thing is a flag
    the reader can see rather than a silently dangling `instance_id`.

    Computed from the store rather than from the writer's own bookkeeping, so it
    checks the thing that was written.
    """
    import zarr

    present = np.zeros(len(adata), dtype=bool)
    region_col = adata.obs["region"].astype(str).to_numpy()
    instance = adata.obs["instance_id"].to_numpy()
    for region in regions:
        arr = zarr.open_array(store / "labels" / region / "s0", mode="r")
        ids = np.unique(np.asarray(arr))
        rows = region_col == region
        present[rows] = np.isin(instance[rows], ids)
    return present


def cmd_table(args, rois: list[Roi]) -> None:
    from spatialdata.models import TableModel

    cells = Cells(args.covid_dir)
    panel = next(r.stack.channel_names for r in rois if r.stack)
    adata, markers = build_table(cells, rois, panel)

    missing = [c for c in panel if c not in markers]
    print(f"X: {adata.shape[0]} cells x {adata.shape[1]} markers (dense float32, "
          f"{adata.X.nbytes / 1e6:.0f} MB)")
    print(f"  panel channels absent from the table ({len(missing)}): {missing}")
    print(f"  obs columns: {len(adata.obs.columns)}   obsm: {list(adata.obsm)}")

    sd = attach(args.store)
    have = {p.name for p in (args.store / "labels").iterdir()} if (args.store / "labels").is_dir() else set()
    regions = sorted(set(adata.obs["region"].astype(str)))
    absent = [r for r in regions if r not in have]
    if absent:
        # A regions table whose `region` names an element that does not exist is
        # not a valid store, and the failure surfaces far from here.
        sys.exit(f"labels elements missing for {len(absent)} regions: {absent[:3]}... run `labels` first")

    adata.obs["in_labels"] = coverage(args.store, adata, regions)
    n_missing = int((~adata.obs["in_labels"]).sum())
    print(f"  cells with no pixels in their labels element: {n_missing} "
          f"({n_missing / len(adata) * 100:.2f}%) — flagged in obs['in_labels']")

    table = TableModel.parse(
        adata,
        region=regions,
        region_key="region",
        instance_key="instance_id",
    )
    sd.tables["table"] = table
    if (args.store / "tables" / "table").exists():
        sd.delete_element_from_disk("table")
    sd.write_element("table")
    sd.write_consolidated_metadata()
    print(f"\nwrote tables/table annotating {len(regions)} labels elements")


# --------------------------------------------------------------------------- retransform


def cmd_retransform(args, rois: list[Roi]) -> None:
    """Recompute every element's affine from the inventory, without touching pixels.

    The affines are derived entirely from `datasources.json`, so correcting how
    that file is read should not cost a re-encode of 4 GB. `write_transformations`
    rewrites the metadata in place, which is the whole change.
    """
    from spatialdata import read_zarr
    from spatialdata.transformations import get_transformation, set_transformation

    sd = read_zarr(args.store)
    written: list[tuple[str, str]] = []
    for roi in rois:
        for suffix, raster in (("_he", roi.he), ("_imc", roi.stack), ("_labels", roi.mask)):
            name = roi.name + suffix
            if raster is None or name not in sd.images and name not in sd.labels:
                continue
            kind = "images" if name in sd.images else "labels"
            element = getattr(sd, kind)[name]
            # The IMC stack defines the ROI grid — it sits at the image-space
            # origin at 1 um/px, verified by sampling DNA1 at the flipped
            # centroids. Only the boxed rasters carry an offset.
            want = raster_affine(raster) if suffix != "_imc" else raster_affine(roi.stack)
            axes = {"input_axes": ("x", "y"), "output_axes": ("x", "y")}
            before = get_transformation(element, roi.name).to_affine_matrix(**axes)
            set_transformation(element, want, roi.name)
            sd.write_transformations(name)
            written.append((name, kind))
            moved = np.abs(want.to_affine_matrix(**axes) - before)
            print(f"{name:<34} dx {moved[0, 2]:>8.1f} um   dy {moved[1, 2]:>8.1f} um")

    finalize(sd, args.store, written)
    print(f"\nrewrote transformations for {len(written)} elements (no pixels touched)")


# --------------------------------------------------------------------------- refine


def cmd_refine(args, rois: list[Roi]) -> None:
    """Nudge each H&E onto its own IMC stack, where the evidence supports it.

    MDV's stored `position` places 13 of the 30 H&E within 30 um of where their
    own pixels say they belong. The rest are off — by 44-152 um for six of them,
    and by hundreds for eleven that were evidently never registered
    (`position (0,0)`, scale exactly 1.0, and a flat correlation surface).

    So this corrects only what it can justify: an offset is applied when the
    cross-correlation peak clears `--min-z` AND the residual exceeds
    `--min-shift`. Everything else keeps MDV's placement, including the images
    that are already good — moving those on a 4 um measurement would be noise.

    The shift is recorded per element under a namespaced attribute, because a
    coordinate that no longer matches the source file should say so.
    """
    from spatialdata import read_zarr
    from spatialdata.transformations import Affine, set_transformation

    check = _load_module("covid-he-registration-check.py")
    sources = json.loads((args.covid_dir / "datasources.json").read_text())
    regions = next(s for s in sources if s["name"] == "cells")["regions"]
    images = args.covid_dir / regions["base_url"]
    entries = regions["all_regions"]

    sd = read_zarr(args.store)
    applied, report = [], []
    print(f"{'ROI':<24} {'dx':>7} {'dy':>7} {'z':>6}  action")
    for roi in rois:
        name = f"{roi.name}_he"
        if roi.he is None or name not in sd.images:
            continue
        entry = entries[roi.name]
        he_meta = entry["images"].get("he") or entry["images"]["un"]
        placement = check.stored_placement(args.store, roi.name)
        measured = check.measure(images, entry, he_meta, placement)
        if measured is None:
            continue
        dx, dy, z = measured

        if z < args.min_z or max(abs(dx), abs(dy)) <= args.min_shift:
            reason = "keep — already aligned" if z >= args.min_z else "keep — peak too flat to trust"
            print(f"{roi.name:<24} {dx:>7.0f} {dy:>7.0f} {z:>6.1f}  {reason}")
            report.append({"roi": roi.name, "dx": dx, "dy": dy, "z": z, "applied": False})
            continue

        (sx, sy), (ox, oy) = placement
        set_transformation(
            sd.images[name],
            Affine(
                np.array([[sx, 0.0, ox + dx], [0.0, sy, oy + dy], [0.0, 0.0, 1.0]]),
                input_axes=("x", "y"),
                output_axes=("x", "y"),
            ),
            roi.name,
        )
        sd.write_transformations(name)
        applied.append(name)
        report.append({"roi": roi.name, "dx": dx, "dy": dy, "z": z, "applied": True})
        print(f"{roi.name:<24} {dx:>7.0f} {dy:>7.0f} {z:>6.1f}  SHIFTED")

    for name in applied:
        path = args.store / "images" / name / "zarr.json"
        meta = json.loads(path.read_text())
        entry = next(r for r in report if f"{r['roi']}_he" == name)
        meta["attributes"]["covid_migration"] = {
            "he_offset_refined_um": [entry["dx"], entry["dy"]],
            "peak_z": entry["z"],
            "method": "FFT cross-correlation of H&E greyness against arcsinh(DNA1/5)",
            "note": "translation no longer matches datasources.json `position`",
        }
        path.write_text(json.dumps(meta))

    finalize(sd, args.store, [(n, "images") for n in applied])
    (args.store.parent / f"{args.store.name}.he-registration.json").write_text(
        json.dumps(report, indent=1)
    )
    print(f"\nshifted {len(applied)} of {len(report)} H&E; the rest keep MDV's placement")


# --------------------------------------------------------------------------- omero


#: MDV's own answer, from `regions.avivator.default_channels` in datasources.json.
DEFAULT_ACTIVE = ("DNA1",)

#: Not a colour scheme, just enough that two active channels are distinguishable.
CHANNEL_COLOURS = {"DNA1": "0000FF", "DNA3": "6666FF", "aSMA": "FF00FF", "CD68": "00FF00"}


def cmd_omero(args, rois: list[Roi]) -> None:
    """Give each IMC stack per-channel display defaults.

    Without this a viewer shows channel 0, which on this panel is `80ArAr` — a
    calibration channel carrying a non-zero floor and no biology. The first six
    channels are all of that kind, so "open the stack and look" produces noise
    and reads as a broken decoder. The plan doc warned that channel index is not
    marker index; this is the metadata that stops a viewer having to know.

    Windows come from level 1 (a quarter of the pixels, same value distribution)
    so the whole pass is a few seconds per ROI rather than a full decode.
    """
    import zarr

    sd = attach(args.store)
    for roi in rois:
        name = f"{roi.name}_imc"
        group = args.store / "images" / name
        if not group.is_dir():
            continue
        meta = json.loads((group / "zarr.json").read_text())
        omero = meta["attributes"]["ome"].setdefault("omero", {})
        labels = [c["label"] for c in omero.get("channels", [])]
        if not labels:
            print(f"{name}: no omero channel labels — skipped")
            continue

        level = "s1" if (group / "s1").is_dir() else "s0"
        data = zarr.open_array(group / level, mode="r")
        channels = []
        for i, label in enumerate(labels):
            plane = np.asarray(data[i])
            lo, hi = (float(v) for v in np.percentile(plane, (50.0, 99.9)))
            channels.append(
                {
                    "label": label,
                    "active": label in DEFAULT_ACTIVE,
                    "color": CHANNEL_COLOURS.get(label, "FFFFFF"),
                    # p99.9 rather than max: IMC is heavy-tailed enough that
                    # `CD10` reaches 5800x its own p99.9, so a max-scaled window
                    # renders every channel as black with a few hot pixels.
                    "window": {
                        "min": float(plane.min()),
                        "max": float(plane.max()),
                        "start": lo,
                        "end": hi if hi > lo else lo + 1.0,
                    },
                }
            )
        omero["channels"] = channels
        (group / "zarr.json").write_text(json.dumps(meta))
        active = [c["label"] for c in channels if c["active"]]
        print(f"{name}: {len(channels)} channels, active {active}")

    sd.write_consolidated_metadata()
    print(f"\nwrote omero display defaults; active by default: {list(DEFAULT_ACTIVE)}")


# --------------------------------------------------------------------------- units


def cmd_units(args, rois: list[Roi]) -> None:
    """Re-apply the axis unit to every element, then re-consolidate.

    Needed as a separate pass because `spatialdata-js-util images recompress`
    rewrites each element's multiscale metadata from its own reading, which
    resets the unit to spatialdata's `"unit"` placeholder. Run it after any
    external tool has touched the store.
    """
    from spatialdata import read_zarr

    sd = read_zarr(args.store)
    written = [
        (path.name, kind)
        for kind in ("images", "labels")
        if (args.store / kind).is_dir()
        for path in sorted((args.store / kind).iterdir())
        if path.is_dir()
    ]
    finalize(sd, args.store, written)
    print(f"unit set to micrometer on {len(written)} elements; metadata re-consolidated")


# --------------------------------------------------------------------------- CLI


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=("he", "imc", "labels", "table", "omero", "refine", "retransform", "units"))
    p.add_argument("--covid-dir", type=Path, default=Path("/Volumes/Crucial X8/covid"))
    p.add_argument("--store", type=Path, default=Path.home() / "data/covid.spatialdata.zarr")
    p.add_argument("--rois", help="comma-separated ROI names, or a count like '2' for the first N")
    p.add_argument(
        "--min-z", type=float, default=6.0, help="refine: least peak sharpness to trust (default 6)"
    )
    p.add_argument(
        "--min-shift", type=float, default=30.0,
        help="refine: leave placements already this close, in micrometres (default 30)",
    )
    p.add_argument("--zstd", type=int, default=19, help="zstd level for raster arrays (default 19)")
    p.add_argument(
        "--imc-levels",
        type=int,
        default=2,
        help="resolution levels for the IMC stack (default 2, matching the source OME-TIFF)",
    )
    args = p.parse_args()
    args.store = args.store.expanduser()

    rois, problems = inventory.load(args.covid_dir)
    for problem in problems:
        print(f"INVENTORY PROBLEM: {problem}")
    rois.sort(key=lambda r: r.name)

    if args.rois:
        if args.rois.isdigit():
            rois = rois[: int(args.rois)]
        else:
            want = set(args.rois.split(","))
            rois = [r for r in rois if r.name in want]
            missing = want - {r.name for r in rois}
            if missing:
                sys.exit(f"unknown ROIs: {sorted(missing)}")

    {
        "he": cmd_he,
        "imc": cmd_imc,
        "labels": cmd_labels,
        "table": cmd_table,
        "omero": cmd_omero,
        "refine": cmd_refine,
        "retransform": cmd_retransform,
        "units": cmd_units,
    }[
        args.command
    ](args, rois)


if __name__ == "__main__":
    main()
