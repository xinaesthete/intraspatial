import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, describe, expect, it } from "vitest";
import * as zarr from "zarrita";
import { fuzzySimplicialSet, knnBruteForceCpu } from "../spatial/umapGraph";
import { fuzzyGraphToCsr, knnToCsr, openStore, readExpressionMatrix, readNObs, writeNeighborsUns, writeObsm, writeObsp } from "./annDataIo";

// These tests build real zarr stores on disk and read them back. `annDataIo.ts` is the
// repo's only WRITE path, and the failure mode it guards against — a store that looks
// written but that anndata cannot load — is invisible to any test that only reads back
// through the same library that wrote it. So the layout is asserted at the
// metadata/attribute level here, and the writes were additionally validated against
// zarr-python 2.18 by hand (see docs/umap-on-anndata.md §5).

const dirs: string[] = [];
function tempDir(): string {
  const d = mkdtempSync(join(tmpdir(), "annobsm-"));
  dirs.push(d);
  return d;
}
afterAll(() => {
  for (const d of dirs) rmSync(d, { recursive: true, force: true });
});

const encoder = new TextEncoder();

/** Build a minimal zarr **v2** AnnData store: dense X, var index, obs index. */
async function makeV2Store(nCells: number, nVars: number, fill: (i: number, g: number) => number): Promise<string> {
  const dir = tempDir();
  const mod = (await import("@zarrita/storage/fs")) as { default: new (p: string) => { set(k: string, v: Uint8Array): Promise<void> } };
  const store = new mod.default(dir);
  const put = (k: string, v: unknown) => store.set(k, encoder.encode(JSON.stringify(v)));

  await put("/.zgroup", { zarr_format: 2 });
  await put("/.zattrs", { "encoding-type": "anndata", "encoding-version": "0.1.0" });

  await put("/X/.zarray", {
    zarr_format: 2,
    shape: [nCells, nVars],
    chunks: [nCells, nVars],
    dtype: "<f4",
    compressor: null,
    fill_value: 0,
    order: "C",
    filters: null,
  });
  await put("/X/.zattrs", { "encoding-type": "array", "encoding-version": "0.2.0" });
  const x = new Float32Array(nCells * nVars);
  for (let i = 0; i < nCells; i++) for (let g = 0; g < nVars; g++) x[i * nVars + g] = fill(i, g);
  await store.set("/X/0.0", new Uint8Array(x.buffer.slice(0)));

  // `obs` / `var` index arrays, as fixed-width unicode (what zarr-python writes for
  // string columns). Only the LENGTH is read by this module, so the payload is zeros.
  for (const [group, count] of [
    ["obs", nCells],
    ["var", nVars],
  ] as const) {
    await put(`/${group}/.zgroup`, { zarr_format: 2 });
    await put(`/${group}/.zattrs`, { "encoding-type": "dataframe", "encoding-version": "0.2.0", _index: "_index", "column-order": [] });
    await put(`/${group}/_index/.zarray`, {
      zarr_format: 2,
      shape: [count],
      chunks: [count],
      dtype: "<U8",
      compressor: null,
      fill_value: null,
      order: "C",
      filters: null,
    });
    await put(`/${group}/_index/.zattrs`, { "encoding-type": "string-array", "encoding-version": "0.2.0" });
    await store.set(`/${group}/_index/0`, new Uint8Array(count * 8 * 4));
  }
  return dir;
}

describe("readExpressionMatrix", () => {
  it("reads a dense v2 X as row-major", async () => {
    const dir = await makeV2Store(6, 4, (i, g) => i * 10 + g);
    const loc = await openStore(dir);
    const m = await readExpressionMatrix(loc);
    expect(m.nCells).toBe(6);
    expect(m.nVars).toBe(4);
    expect(Array.from(m.values.slice(0, 4))).toEqual([0, 1, 2, 3]);
    expect(Array.from(m.values.slice(4, 8))).toEqual([10, 11, 12, 13]);
  });

  it("selects and reorders a gene subset", async () => {
    const dir = await makeV2Store(5, 6, (i, g) => i * 10 + g);
    const loc = await openStore(dir);
    const m = await readExpressionMatrix(loc, { vars: [4, 1] });
    expect(m.nVars).toBe(2);
    // Row 2 should be [24, 21] — column 4 then column 1.
    expect(Array.from(m.values.slice(4, 6))).toEqual([24, 21]);
  });

  it("reports nObs from the obs index", async () => {
    const dir = await makeV2Store(7, 3, () => 1);
    const loc = await openStore(dir);
    expect(await readNObs(loc)).toBe(7);
  });

  it("gives a clear error when the matrix is absent", async () => {
    const dir = await makeV2Store(4, 3, () => 1);
    const loc = await openStore(dir);
    await expect(readExpressionMatrix(loc, { matrix: "counts" })).rejects.toThrow(/no layers\/counts/);
  });
});

describe("writeObsm", () => {
  it("writes a v2 array with the AnnData attributes and correct values", async () => {
    const dir = await makeV2Store(5, 3, (i, g) => i + g);
    const loc = await openStore(dir);
    const emb = Float32Array.from([0, 0, 1, 1, 2, 2, 3, 3, 4, 4]);
    const target = await writeObsm(loc, emb, 5, 2, { key: "X_umap" });
    expect(target).toBe("obsm/X_umap");

    // Read back through zarrita, which is format-agnostic on read.
    const arr = await zarr.open(loc.resolve("obsm/X_umap"), { kind: "array" });
    expect(arr.shape).toEqual([5, 2]);
    expect(arr.attrs["encoding-type"]).toBe("array");
    expect(arr.attrs["encoding-version"]).toBe("0.2.0");
    const chunk = await zarr.get(arr as never);
    expect(Array.from(chunk.data as ArrayLike<number>)).toEqual([0, 0, 1, 1, 2, 2, 3, 3, 4, 4]);

    const group = await zarr.open(loc.resolve("obsm"), { kind: "group" });
    expect(group.attrs["encoding-type"]).toBe("dict");
  });

  it("matches the surrounding store's zarr format (v2 in, v2 out)", async () => {
    const dir = await makeV2Store(4, 2, () => 1);
    const loc = await openStore(dir);
    await writeObsm(loc, new Float32Array(8), 4, 2, { key: "X_umap" });
    const store = (loc as unknown as { store: { get(k: string): Promise<Uint8Array | undefined> } }).store;
    // The v2 marker must be present and the v3 one absent — a mixed-format store is
    // exactly what anndata cannot read.
    expect(await store.get("/obsm/X_umap/.zarray")).toBeDefined();
    expect(await store.get("/obsm/X_umap/zarr.json")).toBeUndefined();
  });

  it("refuses to overwrite an existing key without force", async () => {
    const dir = await makeV2Store(4, 2, () => 1);
    const loc = await openStore(dir);
    await writeObsm(loc, new Float32Array(8), 4, 2, { key: "X_umap" });
    await expect(writeObsm(loc, new Float32Array(8), 4, 2, { key: "X_umap" })).rejects.toThrow(/already exists/);
  });

  it("overwrites with force", async () => {
    const dir = await makeV2Store(4, 2, () => 1);
    const loc = await openStore(dir);
    await writeObsm(loc, new Float32Array(8), 4, 2, { key: "X_umap" });
    await writeObsm(loc, Float32Array.from([9, 9, 9, 9, 9, 9, 9, 9]), 4, 2, { key: "X_umap", force: true });
    const arr = await zarr.open(loc.resolve("obsm/X_umap"), { kind: "array" });
    const chunk = await zarr.get(arr as never);
    expect(Array.from(chunk.data as ArrayLike<number>)[0]).toBe(9);
  });

  it("refuses a row count that disagrees with obs", async () => {
    const dir = await makeV2Store(5, 2, () => 1);
    const loc = await openStore(dir);
    await expect(writeObsm(loc, new Float32Array(8), 4, 2, { key: "X_umap" })).rejects.toThrow(/obs has 5 rows/);
  });

  it("rejects an embedding whose length disagrees with nObs x dim", async () => {
    const dir = await makeV2Store(4, 2, () => 1);
    const loc = await openStore(dir);
    await expect(writeObsm(loc, new Float32Array(7), 4, 2, { key: "X_umap" })).rejects.toThrow(/expected 8/);
  });

  it("zero-pads the trailing chunk rather than truncating it", async () => {
    // 5 rows with a 2-row chunking leaves a 1-row tail; v2 requires it padded to full
    // chunk size on disk, and a truncated tail is read as garbage rather than an error.
    const dir = await makeV2Store(5, 2, () => 1);
    const loc = await openStore(dir);
    const emb = Float32Array.from([1, 1, 2, 2, 3, 3, 4, 4, 5, 5]);
    await writeObsm(loc, emb, 5, 2, { key: "X_umap" });
    const arr = await zarr.open(loc.resolve("obsm/X_umap"), { kind: "array" });
    const chunk = await zarr.get(arr as never);
    expect(Array.from(chunk.data as ArrayLike<number>)).toEqual([1, 1, 2, 2, 3, 3, 4, 4, 5, 5]);
  });
});

describe("fuzzyGraphToCsr", () => {
  it("converts a symmetric COO edge list to sorted CSR", () => {
    // 0-1 (w .5), 1-2 (w .25), both directions present as the graph builder emits them.
    const csr = fuzzyGraphToCsr({
      n: 3,
      head: Uint32Array.from([0, 1, 1, 2]),
      tail: Uint32Array.from([1, 0, 2, 1]),
      weight: Float32Array.from([0.5, 0.5, 0.25, 0.25]),
      nEdges: 4,
    });
    expect(Array.from(csr.indptr)).toEqual([0, 1, 3, 4]);
    expect(Array.from(csr.indices)).toEqual([1, 0, 2, 1]);
    expect(Array.from(csr.data)).toEqual([0.5, 0.5, 0.25, 0.25]);
  });

  it("sorts column indices within each row (scipy assumes it)", () => {
    // Row 0's neighbours arrive out of order.
    const csr = fuzzyGraphToCsr({
      n: 4,
      head: Uint32Array.from([0, 0, 0]),
      tail: Uint32Array.from([3, 1, 2]),
      weight: Float32Array.from([0.3, 0.1, 0.2]),
      nEdges: 3,
    });
    expect(Array.from(csr.indices.slice(0, 3))).toEqual([1, 2, 3]);
    // Values must travel with their columns, not stay put. Compared as f32, since that
    // is what the array actually holds.
    expect(Array.from(csr.data.slice(0, 3))).toEqual(Array.from(Float32Array.from([0.1, 0.2, 0.3])));
  });

  it("round-trips a real graph: every COO edge appears once in the CSR", () => {
    const { data } = twoBlobsForCsr(30, 5, 3);
    const knn = knnBruteForceCpu(data, 60, 5, 8);
    const graph = fuzzySimplicialSet(knn, { nNeighbors: 9 });
    const csr = fuzzyGraphToCsr(graph);
    expect(csr.data.length).toBe(graph.nEdges);
    expect(csr.indptr[csr.rows]).toBe(graph.nEdges);

    const fromCoo = new Map<string, number>();
    for (let e = 0; e < graph.nEdges; e++) fromCoo.set(`${graph.head[e]}-${graph.tail[e]}`, graph.weight[e]!);
    let mismatches = 0;
    for (let i = 0; i < csr.rows; i++) {
      for (let p = csr.indptr[i]!; p < csr.indptr[i + 1]!; p++) {
        if (fromCoo.get(`${i}-${csr.indices[p]}`) !== csr.data[p]!) mismatches++;
      }
    }
    expect(mismatches).toBe(0);
  });
});

describe("knnToCsr", () => {
  it("gives each row exactly k entries, column-sorted", () => {
    const csr = knnToCsr({ n: 2, k: 3, indices: Uint32Array.from([5, 1, 3, 0, 4, 2]), distances: Float32Array.from([9, 1, 5, 2, 8, 4]) });
    expect(Array.from(csr.indptr)).toEqual([0, 3, 6]);
    expect(Array.from(csr.indices.slice(0, 3))).toEqual([1, 3, 5]);
    expect(Array.from(csr.data.slice(0, 3))).toEqual([1, 5, 9]);
    expect(Array.from(csr.indices.slice(3, 6))).toEqual([0, 2, 4]);
  });
});

describe("writeObsp", () => {
  it("writes a csr_matrix group with data/indices/indptr", async () => {
    const dir = await makeV2Store(4, 2, () => 1);
    const loc = await openStore(dir);
    const csr = fuzzyGraphToCsr({
      n: 4,
      head: Uint32Array.from([0, 1, 1, 2]),
      tail: Uint32Array.from([1, 0, 2, 1]),
      weight: Float32Array.from([0.5, 0.5, 0.25, 0.25]),
      nEdges: 4,
    });
    const target = await writeObsp(loc, csr, { key: "connectivities" });
    expect(target).toBe("obsp/connectivities");

    const group = await zarr.open(loc.resolve(target), { kind: "group" });
    expect(group.attrs["encoding-type"]).toBe("csr_matrix");
    expect(group.attrs.shape).toEqual([4, 4]);

    const dataArr = await zarr.open(loc.resolve(`${target}/data`), { kind: "array" });
    const dataChunk = await zarr.get(dataArr as never);
    expect(Array.from(dataChunk.data as ArrayLike<number>)).toEqual([0.5, 0.5, 0.25, 0.25]);
    expect(
      Array.from(
        (await zarr.get((await zarr.open(loc.resolve(`${target}/indices`), { kind: "array" })) as never)).data as ArrayLike<number>,
      ),
    ).toEqual([1, 0, 2, 1]);
    const indptrArr = await zarr.open(loc.resolve(`${target}/indptr`), { kind: "array" });
    const indptrChunk = await zarr.get(indptrArr as never);
    // n+1 entries for n rows; row 3 is empty, so the last two offsets coincide.
    expect(Array.from(indptrChunk.data as ArrayLike<number>)).toEqual([0, 1, 3, 4, 4]);
  });

  it("refuses a non-square matrix and a size that disagrees with obs", async () => {
    const dir = await makeV2Store(4, 2, () => 1);
    const loc = await openStore(dir);
    const square = fuzzyGraphToCsr({
      n: 9,
      head: Uint32Array.from([0]),
      tail: Uint32Array.from([1]),
      weight: Float32Array.from([1]),
      nEdges: 1,
    });
    await expect(writeObsp(loc, square, { key: "connectivities" })).rejects.toThrow(/obs has 4 rows/);
    await expect(writeObsp(loc, { ...square, rows: 2, cols: 3 }, { key: "c" })).rejects.toThrow(/must be square/);
  });

  it("refuses to overwrite without force", async () => {
    const dir = await makeV2Store(3, 2, () => 1);
    const loc = await openStore(dir);
    const csr = fuzzyGraphToCsr({
      n: 3,
      head: Uint32Array.from([0]),
      tail: Uint32Array.from([1]),
      weight: Float32Array.from([1]),
      nEdges: 1,
    });
    await writeObsp(loc, csr, { key: "connectivities" });
    await expect(writeObsp(loc, csr, { key: "connectivities" })).rejects.toThrow(/already exists/);
    await writeObsp(loc, csr, { key: "connectivities", force: true });
  });
});

describe("writeNeighborsUns", () => {
  it("records the keys scanpy looks for", async () => {
    const dir = await makeV2Store(3, 2, () => 1);
    const loc = await openStore(dir);
    await writeNeighborsUns(loc, { connectivitiesKey: "connectivities", distancesKey: "distances", nNeighbors: 15 });
    const group = await zarr.open(loc.resolve("uns/neighbors"), { kind: "group" });
    expect(group.attrs.connectivities_key).toBe("connectivities");
    expect(group.attrs.distances_key).toBe("distances");
    expect((group.attrs.params as { n_neighbors: number }).n_neighbors).toBe(15);
  });
});

/** Local fixture: the CSR tests need a real graph, and importing the shared blob helper
 *  from the spatial tests would couple two suites for one function. */
function twoBlobsForCsr(nPer: number, dim: number, seed: number) {
  let s = seed >>> 0;
  const rnd = () => {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const n = nPer * 2;
  const data = new Float64Array(n * dim);
  for (let i = 0; i < n; i++) for (let c = 0; c < dim; c++) data[i * dim + c] = rnd() * 2 - 1 + (c === 0 && i >= nPer ? 25 : 0);
  return { data, n };
}
