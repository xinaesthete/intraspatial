#!/usr/bin/env tsx
// MDV project (`datafile.h5` + `datasources.json`) → columnar zarr store.
//
// A STOPGAP, and worth saying so plainly: MDV is getting its own zarr store, and when it lands this
// script is redundant — the reader (`src/datasource/mdvStore.ts`) is the half that survives. What
// this buys in the meantime is that the reader can be written and tested against the real covid
// project now, and that the layout it assumes is a concrete proposal rather than a guess.
//
// The conversion is close to mechanical because MDV's h5 is *already* columnar: one contiguous
// uncompressed dataset per column, categoricals stored as integer codes. Only two things have to be
// added, both of them metadata that MDV keeps outside the file in `datasources.json` — the category
// labels a text column's codes index into, and the display names. Those are copied into the zarr
// attrs so the resulting store is self-describing (see `mdvStore.ts` for the layout).
//
// COLUMNS ARE ENUMERATED FROM THE H5, NOT FROM THE JSON. The covid project has 143 datasets against
// 135 configured columns: `heterogeneity`, `median`, `skew` and `LCH_image` exist in the file and
// are simply not surfaced by any view. Driving off the config would drop them silently, which is
// the one failure mode a converter must not have. Config metadata is joined on where it exists and
// the rest is carried through with what the h5 itself says.
//
// Requires the HDF5 CLI tools on PATH (`h5ls`, `h5dump` — `brew install hdf5`). Reading HDF5 from
// TypeScript would mean another dependency for a file format we are converting *away* from.
//
// Not yet compressed: zarrita's write path here emits raw chunks, so the store is the same size as
// the h5 (160 MB for covid). Fine for a local file and for the columns the statistics need (x, y,
// annotations for all 545,400 cells is 4.9 MB); worth revisiting before anything is served over a
// network.
//
//   pnpm mdv:zarr <project-dir> <out.zarr> [options]
//
//     --only <a,b>    convert only these datasources
//     --force         overwrite an existing output directory
//     --dry-run       report what would be written, write nothing

import { execFileSync } from "node:child_process";
import { existsSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import * as zarr from "zarrita";
import { assignKeys, MDV_ZARR_VERSION, type MdvColumn } from "../src/datasource/mdvStore";

/** h5 native type → zarr dtype. Deliberately exhaustive-by-exception: an unmapped type stops the
 *  conversion rather than guessing a width, because a silently mis-typed column reads back as
 *  plausible nonsense. */
const H5_TO_ZARR: Record<string, zarr.NumberDataType> = {
  "native float": "float32",
  "native double": "float64",
  "native char": "int8",
  "native signed char": "int8",
  "native unsigned char": "uint8",
  "native short": "int16",
  "native unsigned short": "uint16",
  "native int": "int32",
  "native unsigned int": "uint32",
  "native long": "int64",
  "native long long": "int64",
};

const TYPED: Record<zarr.NumberDataType, new (b: ArrayBuffer) => zarr.TypedArray<zarr.NumberDataType>> = {
  int8: Int8Array,
  int16: Int16Array,
  int32: Int32Array,
  int64: BigInt64Array,
  uint8: Uint8Array,
  uint16: Uint16Array,
  uint32: Uint32Array,
  uint64: BigUint64Array,
  float32: Float32Array,
  float64: Float64Array,
} as never;

const ROWS_PER_CHUNK = 65536;

interface H5Dataset {
  readonly name: string;
  readonly rows: number;
  /** Set for a numeric column. */
  readonly dtype?: zarr.NumberDataType;
  /** Set for a fixed-length string column; the byte width per row. */
  readonly stringWidth?: number;
}

function h5(bin: string, args: string[]): string {
  try {
    return execFileSync(bin, args, { encoding: "utf8", maxBuffer: 1 << 28 });
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    throw new Error(`${bin} failed: ${msg}\n(HDF5 CLI tools required — brew install hdf5)`);
  }
}

/** Dataset names in a group. `h5ls` escapes spaces in its listing (`Cell\ Type\ 1`); the real name
 *  is the unescaped one, and that is what `h5dump -d` wants. */
function listGroup(file: string, group: string): string[] {
  return h5("h5ls", [`${file}/${group}`])
    .split("\n")
    .filter((l) => /\sDataset\s/.test(l))
    .map((l) => l.replace(/\s+Dataset\s.*$/, "").replace(/\\(.)/g, "$1"))
    .filter((n) => n.length > 0);
}

function describe(file: string, group: string, name: string): H5Dataset {
  const v = h5("h5ls", ["-v", `${file}/${group}/${name}`]);
  const shape = /Dataset\s*\{(\d+)/.exec(v);
  if (!shape) throw new Error(`cannot read shape of ${group}/${name}`);
  const rows = Number(shape[1]);
  const type = /Type:\s*(.+)/.exec(v)?.[1]?.trim() ?? "";
  const str = /^(\d+)-byte\s.*string$/.exec(type);
  if (str) return { name, rows, stringWidth: Number(str[1]) };
  const dtype = H5_TO_ZARR[type];
  if (!dtype) throw new Error(`${group}/${name}: unmapped HDF5 type '${type}'`);
  return { name, rows, dtype };
}

/** Raw little-endian bytes of one dataset, via `h5dump -b`. */
function readRaw(file: string, group: string, name: string, scratch: string): Buffer {
  const out = join(scratch, "col.bin");
  h5("h5dump", ["-d", `/${group}/${name}`, "-b", "LE", "-o", out, "-y", "-A", "0", file]);
  const buf = readFileSync(out);
  rmSync(out, { force: true });
  return buf;
}

function toArrayBuffer(b: Buffer): ArrayBuffer {
  return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) as ArrayBuffer;
}

interface Args {
  project: string;
  out: string;
  only?: Set<string>;
  force: boolean;
  dryRun: boolean;
}

function parseArgs(argv: string[]): Args {
  const pos: string[] = [];
  let only: Set<string> | undefined;
  let force = false;
  let dryRun = false;
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i]!;
    if (a === "--only") only = new Set(argv[++i]!.split(",").map((s) => s.trim()));
    else if (a === "--force") force = true;
    else if (a === "--dry-run") dryRun = true;
    else if (a.startsWith("--")) throw new Error(`unknown option ${a}`);
    else pos.push(a);
  }
  if (pos.length !== 2) throw new Error("usage: pnpm mdv:zarr <project-dir> <out.zarr> [--only a,b] [--force] [--dry-run]");
  return { project: pos[0]!, out: pos[1]!, only, force, dryRun };
}

interface ConfigColumn {
  field: string;
  name?: string;
  datatype?: string;
  values?: string[];
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const file = join(args.project, "datafile.h5");
  const configPath = join(args.project, "datasources.json");
  if (!existsSync(file)) throw new Error(`no datafile.h5 in ${args.project}`);
  if (!existsSync(configPath)) throw new Error(`no datasources.json in ${args.project}`);
  if (existsSync(args.out) && !args.force && !args.dryRun) {
    throw new Error(`${args.out} exists — pass --force to overwrite`);
  }

  const config = JSON.parse(readFileSync(configPath, "utf8")) as { name: string; columns: ConfigColumn[]; regions?: unknown }[];
  const configByDs = new Map(config.map((d) => [d.name, new Map(d.columns.map((c) => [c.field, c]))]));
  // The `regions` block is datasource-level metadata, not a column: it names the positional and
  // region-splitting columns and — the reason it must survive the conversion — carries `scale` /
  // `scale_unit`, the only statement of physical scale in the whole project. Copied verbatim so the
  // reader can resolve µm-per-unit rather than assume it.
  const regionsByDs = new Map(config.filter((d) => d.regions).map((d) => [d.name, d.regions]));

  const groups = h5("h5ls", [file])
    .split("\n")
    .filter((l) => /\sGroup\s*$/.test(l))
    .map((l) => l.replace(/\s+Group\s*$/, "").replace(/\\(.)/g, "$1"))
    .filter((n) => n.length > 0 && (!args.only || args.only.has(n)));
  if (groups.length === 0) throw new Error("no datasources matched");

  const scratch = mkdtempSync(join(tmpdir(), "mdv-zarr-"));
  const rootLoc = args.dryRun ? undefined : zarr.root(new (await import("@zarrita/storage/fs")).default(args.out));
  const dsKeys = assignKeys(groups);
  const rootIndex: { name: string; key: string; rows: number }[] = [];

  try {
    // The root group is written LAST, once `rootIndex` is complete: a store whose root attrs are
    // present is a store the reader will trust, so it must not exist until every column does.
    for (let gi = 0; gi < groups.length; gi++) {
      const group = groups[gi]!;
      const dsKey = dsKeys[gi]!;
      const names = listGroup(file, group);
      const meta = names.map((n) => describe(file, group, n));
      const rows = meta[0]?.rows ?? 0;
      const ragged = meta.filter((m) => m.rows !== rows);
      if (ragged.length > 0) {
        throw new Error(`datasource '${group}' is ragged: ${ragged.map((m) => `${m.name}=${m.rows}`).join(", ")} vs ${rows}`);
      }

      const cfg = configByDs.get(group);
      const colKeys = assignKeys(names);
      const columns: MdvColumn[] = names.map((field, i) => {
        const c = cfg?.get(field);
        return {
          field,
          key: colKeys[i]!,
          name: c?.name ?? field,
          datatype: c?.datatype ?? (meta[i]!.stringWidth ? "unique" : "double"),
          ...(c?.values ? { values: c.values } : {}),
        };
      });
      const unconfigured = names.filter((n) => !cfg?.has(n));

      console.log(
        `${group}: ${names.length} columns × ${rows} rows${unconfigured.length ? `  (${unconfigured.length} not in datasources.json: ${unconfigured.join(", ")})` : ""}`,
      );
      rootIndex.push({ name: group, key: dsKey, rows });
      if (args.dryRun) continue;

      const dsLoc = rootLoc!.resolve(dsKey);
      const regions = regionsByDs.get(group);
      await zarr.create(dsLoc, { attributes: { mdv: { name: group, rows, columns, ...(regions ? { regions } : {}) } } });

      for (let i = 0; i < names.length; i++) {
        const m = meta[i]!;
        const col = columns[i]!;
        const raw = readRaw(file, group, m.name, scratch);
        if (m.stringWidth !== undefined) {
          // Fixed-length null-padded UTF-8, stored as [rows, width] bytes. Kept rather than dropped
          // so the conversion is lossless; `mdvStore` decodes it on demand.
          const arr = await zarr.create(dsLoc.resolve(col.key), {
            shape: [rows, m.stringWidth],
            chunkShape: [Math.min(rows, ROWS_PER_CHUNK), m.stringWidth],
            dtype: "uint8",
            attributes: { mdv: { ...col, stringWidth: m.stringWidth } },
          });
          await zarr.set(arr as never, null, {
            data: new Uint8Array(toArrayBuffer(raw)),
            shape: [rows, m.stringWidth],
            stride: [m.stringWidth, 1],
          } as never);
        } else {
          const data = new TYPED[m.dtype!](toArrayBuffer(raw));
          if (data.length !== rows) throw new Error(`${group}/${m.name}: dumped ${data.length} values, expected ${rows}`);
          const arr = await zarr.create(dsLoc.resolve(col.key), {
            shape: [rows],
            chunkShape: [Math.min(rows, ROWS_PER_CHUNK)],
            dtype: m.dtype!,
            attributes: { mdv: col },
          });
          await zarr.set(arr as never, null, { data, shape: [rows], stride: [1] } as never);
        }
      }
    }

    if (rootLoc) {
      await zarr.create(rootLoc, { attributes: { mdv: { version: MDV_ZARR_VERSION, datasources: rootIndex } } });
    }
  } finally {
    rmSync(scratch, { recursive: true, force: true });
  }
  console.log(args.dryRun ? "\ndry run — nothing written" : `\nwrote ${args.out}`);
}

main().catch((e) => {
  console.error(e instanceof Error ? e.message : e);
  process.exit(1);
});
