// Where the channels come from: the categories of an `obs` column, or chosen vars of `X`. Most
// wild-type SpatialData stores carry no cell-type annotation at all — a raw Xenium export has
// `cell_id`, areas and counts in `obs` and nothing else — so `X` is not a fallback but often the only
// source. A var selection is applied explicitly: a CSR matrix is read whole for any selection.

import { useMemo, useState } from "react";
import type { ColumnInfo } from "../datasource/cellTable";
import { selectionCost, type VarCatalog } from "../datasource/varMatrix";
import { MAX_CHANNELS } from "./cellChannels";

export type SourceKind = "obs" | "x";

const NO_NAMES: readonly string[] = [];

/** Categorical `obs` columns are the only ones that can define channels: a numeric column has no
 *  categories to split on, and one with as many distinct values as cells is an identifier. */
export const usableObsColumns = (columns: readonly ColumnInfo[] | undefined): ColumnInfo[] =>
  (columns ?? []).filter((c) => c.kind === "categorical" && (c.nCategories ?? 0) >= 2);

interface Props {
  readonly kind: SourceKind;
  readonly onKind: (k: SourceKind) => void;
  readonly obsColumns: readonly ColumnInfo[];
  readonly obsColumn: string | undefined;
  readonly onObsColumn: (c: string) => void;
  readonly catalog: VarCatalog | undefined;
  readonly matrix: string;
  readonly onMatrix: (m: string) => void;
  /** The vars in use; `onApply` replaces them with the draft. */
  readonly applied: readonly number[];
  readonly onApply: (vars: number[]) => void;
}

export function ChannelSource(p: Props) {
  const [query, setQuery] = useState("");
  const [draft, setDraft] = useState<number[]>([]);
  const names = p.catalog?.names ?? NO_NAMES;
  const matches = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return [];
    return names.flatMap((n, i) => (n.toLowerCase().includes(q) && !draft.includes(i) ? [i] : []));
  }, [query, names, draft]);
  const full = draft.length >= MAX_CHANNELS;
  const add = (vars: number[]) => setDraft((d) => [...d, ...vars.filter((v) => !d.includes(v))].slice(0, MAX_CHANNELS));
  const dirty = draft.length !== p.applied.length || draft.some((v, i) => v !== p.applied[i]);
  const encoding =
    p.catalog?.encoding === "csr"
      ? "stored CSR — any selection reads the whole matrix"
      : p.catalog?.encoding === "csc"
        ? "stored CSC — a selection reads only its own columns"
        : "dense";

  return (
    <div className="picker">
      <div className="tabs" role="tablist">
        <button type="button" role="tab" className="tab" aria-selected={p.kind === "obs"} onClick={() => p.onKind("obs")}>
          cell types (obs)
        </button>
        <button type="button" role="tab" className="tab" aria-selected={p.kind === "x"} onClick={() => p.onKind("x")}>
          expression (X)
        </button>
      </div>
      {p.kind === "obs" ? (
        p.obsColumns.length ? (
          <label>
            Column
            <select value={p.obsColumn ?? ""} onChange={(e) => p.onObsColumn(e.target.value)}>
              {p.obsColumns.map((c) => (
                <option key={c.name} value={c.name}>
                  {c.name} — {c.nCategories} categories
                </option>
              ))}
            </select>
          </label>
        ) : (
          <p className="hint">No categorical obs column with two or more categories — this store has no annotation, so use X.</p>
        )
      ) : !p.catalog || p.catalog.error || p.catalog.nVars === 0 ? (
        <p className="hint">{p.catalog?.error ? `No readable X — ${p.catalog.error}` : "Reading var names…"}</p>
      ) : (
        <>
          <label>
            Matrix
            <select value={p.matrix} onChange={(e) => p.onMatrix(e.target.value)}>
              {p.catalog.matrices.map((m) => (
                <option key={m} value={m}>
                  {m === "X" ? "X" : `layers/${m}`}
                </option>
              ))}
            </select>
          </label>
          <span className="row">
            <input placeholder="Filter vars…" value={query} onChange={(e) => setQuery(e.target.value)} disabled={full} />
            <button type="button" disabled={full || matches.length === 0} onClick={() => add(matches)}>
              add all {matches.length || ""}
            </button>
          </span>
          {matches.length > 0 && (
            <div className="chips">
              {matches.slice(0, 24).map((v) => (
                <button key={v} type="button" className="chip" disabled={full} onClick={() => add([v])}>
                  + {names[v]}
                </button>
              ))}
              {matches.length > 24 && <span className="chip-note">… {matches.length - 24} more — narrow the filter</span>}
            </div>
          )}
          <div className="picker-head">
            Vars: {draft.length} of at most {MAX_CHANNELS}
          </div>
          {draft.length > 0 && (
            <div className="chips">
              {draft.map((v) => (
                <button key={v} type="button" className="chip on" onClick={() => setDraft((d) => d.filter((x) => x !== v))}>
                  {names[v] ?? `var${v}`} ×
                </button>
              ))}
            </div>
          )}
          <span className="row">
            <button type="button" disabled={!dirty || draft.length < 2} onClick={() => p.onApply(draft)}>
              use these vars
            </button>
            <button type="button" disabled={draft.length === 0} onClick={() => setDraft([])}>
              clear
            </button>
          </span>
          <p className="hint">
            {p.catalog.nVars} vars × {p.catalog.nCells} cells; {encoding}
            {draft.length > 0 ? ` — this selection reads ~${(selectionCost(p.catalog, draft.length) * 100).toFixed(0)}% of it.` : "."}
          </p>
        </>
      )}
    </div>
  );
}
