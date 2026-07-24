// CSV → `CellTable`, so the cell-stats front is usable without a SpatialData store.
//
// The parsing, inspection and grouping live in `src/spatial/cellCsv.ts` (generic, tested); this is
// only the playground-side glue that turns the grouping into the same `CellTable` the zarr path
// produces. Everything downstream — splats, TCM, cross-PCF, the matrix — is then identical, because
// it only ever sees a `CellTable`.
//
// Placement: a CSV carries no coordinate system, so the cloud is placed at identity in a system
// named after the file. That is an honest "these are in their own space" rather than a fabricated
// transform (ADR-0018: array-space and placed-at-identity are distinct states, and inventing a
// registration we do not have would be the worse lie).

import type { Affine3 } from "../../../src/datasource";
import { centroidsToField } from "../../../src/datasource";
import type { Graph } from "../../../src/gpu/graph/graph";
import type { FieldProvenance, ResolvedPlacement } from "../../../src/gpu/graph/handle";
import { type CsvGroupOptions, groupCsvCells } from "../../../src/spatial/cellCsv";
import type { CellTable, CellTypeCloud } from "./cellTable";

export type { CsvColumnInfo, CsvSchema } from "../../../src/spatial/cellCsv";
export { inspectCsv, parseCsv } from "../../../src/spatial/cellCsv";

const IDENTITY_AFFINE: Affine3 = {
  origin: [0, 0, 0],
  axes: [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
  ],
};

export interface CsvReadOptions extends CsvGroupOptions {
  /** Names the system the identity placement targets. Defaults to "csv". */
  system?: string;
  /** File name, for the HUD. */
  label?: string;
}

/** Build a `CellTable` from parsed CSV rows. */
export function csvToCellTable(rows: string[][], opts: CsvReadOptions): CellTable {
  const system = opts.system ?? "csv";
  const placement: ResolvedPlacement = { system, worldFromArray: IDENTITY_AFFINE };
  const { order, clouds, skipped } = groupCsvCells(rows, opts);

  const types: CellTypeCloud[] = order.map((label, id) => {
    const cloud = clouds.get(label)!;
    const provenance: FieldProvenance = { region: system, instanceKey: "row", cellTypeId: id };
    return {
      id,
      label,
      n: cloud.xs.length,
      xs: cloud.xs,
      ys: cloud.ys,
      source: (g: Graph) => g.source(centroidsToField(cloud.xs, cloud.ys, { placement, provenance }), `cellType:${label}`),
    };
  });

  const total = types.reduce((s, t) => s + t.n, 0);
  return {
    types,
    placement,
    provenance: { region: system, instanceKey: "row" },
    tableName: opts.label ?? "csv",
    typeColumn: opts.typeColumn,
    // A CSV states no unit. Anything else would be a guess dressed up as metadata.
    units: {},
    totalCells: total,
    system,
    label: `${opts.label ?? "csv"} · ${opts.typeColumn} · ${total} cells · ${types.length} types${
      skipped ? ` · ${skipped} rows skipped` : ""
    }`,
  };
}
