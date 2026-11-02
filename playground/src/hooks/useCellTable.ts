// Store-level hooks for cell tables: which tables a store has, one table's centroids, and its
// expression matrix's var names. Each is scoped to its inputs, so another table's value never
// stands in for this one's.

import type { SpatialData } from "@spatialdata/core";
import { useMemo } from "react";
import { type CellTable, listCellTables, NO_TYPE_COLUMN, readCellTable, type TableInfo } from "../datasource/cellTable";
import { listVars, type VarCatalog } from "../datasource/varMatrix";
import { type AsyncState, useAsync } from "./useAsync";

export function useCellTables(sdata: SpatialData | undefined): AsyncState<TableInfo[]> {
  return useAsync(() => (sdata ? listCellTables(sdata) : undefined), [sdata], sdata);
}

/** The coordinate system to place cells in: `global` when the store has it, else its first. */
export const defaultSystem = (sdata: SpatialData | undefined): string | undefined =>
  sdata?.coordinateSystems.includes("global") ? "global" : sdata?.coordinateSystems[0];

/**
 * One table's centroids, read ONCE and untyped: the base read exists to provide `rowOrder`, the row
 * index both `obs` and `X` are aligned to, and must succeed on a store with no annotation at all.
 * Placed in `system`, which should be the one the view draws in — the reader's own default is a
 * guess from root attributes most stores do not carry.
 */
export function useCellTable(sdata: SpatialData | undefined, table: string | undefined, system: string | undefined): AsyncState<CellTable> {
  const scope = useMemo(() => ({ sdata, table, system }), [sdata, table, system]);
  return useAsync(
    () => (sdata && table ? readCellTable(sdata, { table, typeColumn: NO_TYPE_COLUMN, system }) : undefined),
    [sdata, table, system],
    scope,
  );
}

/** The var names of a table's `X` (and its layers), without reading any values. */
export function useVarCatalog(sdata: SpatialData | undefined, table: string | undefined): AsyncState<VarCatalog> {
  const scope = useMemo(() => ({ sdata, table }), [sdata, table]);
  return useAsync(() => (sdata && table ? listVars(sdata, table) : undefined), [sdata, table], scope);
}
