#!/usr/bin/env python3
"""Check the migrated COVID store end to end, against the sources it came from.

    uv run --with spatialdata --with tifffile --with imagecodecs --with h5py \
        --with ~/code/www/SpatialData.ts/python/spatialdata-js-util \
        python scripts/covid-spatialdata-validate.py ~/data/covid.spatialdata.zarr

Deliberately not a re-run of the writer's own bookkeeping. Every check below
either reads the original file on the external volume, or exercises the join a
consumer would actually make — because the failure this migration is most
exposed to is a silent coordinate error, and a writer that is wrong about the y
axis will happily verify itself.

The centrepiece is check 4: sample the labels raster at each cell's own
`obsm/spatial`, transformed through SpatialData's coordinate machinery rather
than through this script's arithmetic, and require the pixel value to be that
cell's `instance_id`. That one test fails if the y flip, the affine, the +1
label offset, or the region wiring is wrong.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

RNG = np.random.default_rng(0)
SAMPLE_ROIS = 6
SAMPLE_CELLS = 4000


def fail(msg: str) -> None:
    print(f"  FAIL  {msg}")
    fail.count += 1


fail.count = 0


def ok(msg: str) -> None:
    print(f"  ok    {msg}")


def main(store: Path, covid_dir: Path) -> None:
    import tifffile
    import zarr
    from spatialdata import read_zarr
    from spatialdata.transformations import get_transformation

    sd = read_zarr(store)

    print("1. shape of the store")
    imc = sorted(n for n in sd.images if n.endswith("_imc"))
    he = sorted(n for n in sd.images if n.endswith("_he"))
    labels = sorted(sd.labels)
    print(f"     {len(imc)} imc, {len(he)} he, {len(labels)} labels, "
          f"{len(sd.tables)} table(s), {len(sd.coordinate_systems)} coordinate systems")
    if len(imc) != 32:
        fail(f"expected 32 IMC stacks, found {len(imc)}")
    if len(labels) != 32:
        fail(f"expected 32 labels elements, found {len(labels)}")
    if len(he) != 30:
        fail(f"expected 30 H&E (2 ROIs have none), found {len(he)}")
    if len(sd.coordinate_systems) != 32:
        fail(f"expected one coordinate system per ROI, found {len(sd.coordinate_systems)}")
    else:
        ok("one coordinate system per ROI, and nothing landed in `global`")

    print("2. axis units are real, not spatialdata's placeholder")
    placeholder = []
    for name in [*imc, *he, *labels]:
        kind = "labels" if name in sd.labels else "images"
        meta = zarr.open_group(store / kind / name, mode="r").attrs.asdict()
        for ms in meta.get("ome", {}).get("multiscales", []):
            for t in ms.get("coordinateTransformations", []):
                for axis in t.get("output", {}).get("axes", []):
                    if axis.get("type") == "space" and axis.get("unit") != "micrometer":
                        placeholder.append(name)
    if placeholder:
        fail(f"{len(set(placeholder))} elements still carry a placeholder unit")
    else:
        ok("every spatial axis is micrometer")

    print("3. IMC pixels are bit-identical to the source OME-TIFF")
    for name in RNG.choice(imc, size=3, replace=False):
        roi = name[: -len("_imc")]
        src = source_stack(covid_dir, roi)
        el = sd.images[name]
        with tifffile.TiffFile(src) as tf:
            series = tf.series[0]
            for ci in RNG.choice(len(series.levels[0].pages), size=3, replace=False):
                got = np.asarray(el["scale0"]["image"][int(ci)])
                want = series.levels[0].pages[int(ci)].asarray()
                if not np.array_equal(got, want):
                    fail(f"{name} channel {ci} differs from source")
                    break
                got1 = np.asarray(el["scale1"]["image"][int(ci)])
                want1 = series.levels[1].pages[int(ci)].asarray()
                if not np.array_equal(got1, want1):
                    fail(f"{name} channel {ci} level 1 differs from the source's own level 1")
                    break
            else:
                ok(f"{name}: 3 channels exact at both levels")

    print("4. table -> labels join, through SpatialData's own transforms")
    table = sd.tables["table"]
    region_col = table.obs["region"].astype(str).to_numpy()
    instance = table.obs["instance_id"].to_numpy()
    in_labels = table.obs["in_labels"].to_numpy()
    spatial = table.obsm["spatial"]

    checked = matched = 0
    for name in RNG.choice(labels, size=SAMPLE_ROIS, replace=False):
        el = sd.labels[name]
        roi = name[: -len("_labels")]
        transform = get_transformation(el, roi)
        # micrometres -> pixels is the inverse of the element's own affine, so
        # the check exercises the transform a viewer would apply, not a
        # reimplementation of it.
        inverse = np.linalg.inv(
            transform.to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y"))
        )
        rows = np.flatnonzero((region_col == name) & in_labels)
        rows = RNG.choice(rows, size=min(SAMPLE_CELLS, len(rows)), replace=False)
        xy = np.c_[spatial[rows], np.ones(len(rows))] @ inverse.T
        col = np.floor(xy[:, 0]).astype(int)
        row = np.floor(xy[:, 1]).astype(int)
        raster = np.asarray(el)
        got = raster[np.clip(row, 0, raster.shape[0] - 1), np.clip(col, 0, raster.shape[1] - 1)]
        hit = got == instance[rows]
        checked += len(rows)
        matched += int(hit.sum())
        if hit.mean() < 0.99:
            fail(f"{name}: only {hit.mean() * 100:.1f}% of centroids land on their own label")
    if checked:
        rate = matched / checked * 100
        (ok if rate >= 99 else fail)(
            f"{matched}/{checked} sampled cells ({rate:.2f}%) sit on their own label id"
        )

    print("5. the table is a valid regions table")
    attrs = table.uns.get("spatialdata_attrs", {})
    if attrs.get("region_key") != "region" or attrs.get("instance_key") != "instance_id":
        fail(f"region_key/instance_key wrong: {attrs}")
    else:
        ok(f"region_key={attrs['region_key']} instance_key={attrs['instance_key']}")
    named = set(np.atleast_1d(attrs.get("region", [])))
    if named != set(labels):
        fail(f"`region` names {len(named)} elements, store has {len(labels)} labels")
    else:
        ok("every named region exists as a labels element")
    if len(table) != 545_400:
        fail(f"expected 545,400 cells, found {len(table)}")
    if table.X.shape[1] != 37:
        fail(f"expected 37 markers in X, found {table.X.shape[1]}")
    ok(f"X {table.X.shape} {table.X.dtype}, obs {len(table.obs.columns)} cols, obsm {list(table.obsm)}")

    print("6. X values match the source h5 column for column")
    import h5py

    cells = h5py.File(covid_dir / "datafile.h5", "r")["cells"]
    for marker in RNG.choice(list(table.var_names), size=5, replace=False):
        if not np.array_equal(table[:, marker].X.ravel(), cells[marker][:]):
            fail(f"X column {marker} differs from the h5")
            break
    else:
        ok("5 sampled marker columns identical to datafile.h5")

    print("7. consolidated metadata covers the store")
    # Read the on-disk artefact rather than the object model: the point of the
    # check is that a reader taking the one-request path sees the same thing as
    # one that walks the store, and a stale root is exactly how that breaks.
    import json

    root = json.loads((store / "zarr.json").read_text())
    consolidated = (root.get("consolidated_metadata") or {}).get("metadata")
    if not consolidated:
        fail("no consolidated metadata at the store root")
    else:
        want = {f"images/{n}" for n in sd.images} | {f"labels/{n}" for n in sd.labels}
        missing = want - set(consolidated)
        (ok if not missing else fail)(
            f"{len(consolidated)} nodes consolidated"
            + (f", missing {sorted(missing)[:3]}" if missing else "")
        )
        stale = [
            k
            for k, v in consolidated.items()
            for ms in v.get("attributes", {}).get("ome", {}).get("multiscales", [])
            for t in ms.get("coordinateTransformations", [])
            for a in t.get("output", {}).get("axes", [])
            if a.get("type") == "space" and a.get("unit") != "micrometer"
        ]
        if stale:
            fail(f"consolidated copy is stale for {len(set(stale))} elements")
        else:
            ok("the consolidated copy agrees with the elements")

    print(f"\n{'PASS' if not fail.count else f'{fail.count} FAILURES'}")
    sys.exit(1 if fail.count else 0)


def source_stack(covid_dir: Path, roi: str) -> Path:
    import json

    sources = json.loads((covid_dir / "datasources.json").read_text())
    regions = next(s for s in sources if s["name"] == "cells")["regions"]
    return covid_dir / regions["base_url"] / regions["all_regions"][roi]["ome_tiff"]


if __name__ == "__main__":
    main(
        Path(sys.argv[1] if len(sys.argv) > 1 else Path.home() / "data/covid.spatialdata.zarr"),
        Path(sys.argv[2] if len(sys.argv) > 2 else "/Volumes/Crucial X8/covid"),
    )
