import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, describe, expect, it } from "vitest";
import * as zarr from "zarrita";
import { openStore, readExpressionMatrix, readNObs, writeObsm } from "./annDataObsm";

// These tests build real zarr stores on disk and read them back. `annDataObsm.ts` is
// the repo's first WRITE path, and the failure mode it guards against — a store that
// looks written but that anndata cannot load — is invisible to any test that only
// reads back through the same library that wrote it. So the v2 layout is asserted at
// the BYTE/metadata level here, and `annDataObsm.python.test.ts` (skipped unless a
// zarr-python venv is present) closes the loop with a foreign reader.

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
