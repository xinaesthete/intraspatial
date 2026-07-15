// Decode the real HbS + weights GeoTIFFs (the app assets shipped in playground/public/hspf) and
// check the scaffold is sane: matching dimensions, an ocean/land split via the sentinel, plausible
// land allele frequencies, non-negative weights, and an Africa-ish geographic extent.
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { scaffoldFromGeoTIFFs } from "./scaffold";

async function asset(name: string): Promise<ArrayBuffer> {
  const buf = await readFile(resolve(process.cwd(), "playground/public/hspf", name));
  return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength) as ArrayBuffer;
}

describe("HsPf scaffold from GeoTIFF", () => {
  it("decodes the real HbS + weights rasters into a coherent scaffold", async () => {
    const [hbsBuf, weightsBuf] = await Promise.all([asset("hbsfilter.tif"), asset("pf2000.tif")]);
    const scaffold = await scaffoldFromGeoTIFFs(hbsBuf, weightsBuf);

    expect(scaffold.width).toBeGreaterThan(100);
    expect(scaffold.height).toBeGreaterThan(100);
    expect(scaffold.hbs.length).toBe(scaffold.width * scaffold.height);
    expect(scaffold.weights.length).toBe(scaffold.hbs.length);

    let ocean = 0;
    let land = 0;
    let maxHbs = 0;
    let minWeight = Infinity;
    for (let i = 0; i < scaffold.hbs.length; i++) {
      const v = scaffold.hbs[i] ?? -2;
      if (v < 0) {
        ocean++;
      } else {
        land++;
        maxHbs = Math.max(maxHbs, v);
      }
      minWeight = Math.min(minWeight, scaffold.weights[i] ?? 0);
    }

    // The map is a real coastline: a substantial mix of ocean (sentinel) and land cells.
    expect(ocean).toBeGreaterThan(1000);
    expect(land).toBeGreaterThan(1000);
    // HbS is an allele frequency: land values live in a plausible [0, 1) band, not raw counts.
    expect(maxHbs).toBeGreaterThan(0);
    expect(maxHbs).toBeLessThan(1);
    // Weights are non-negative (missing ⇒ 0).
    expect(minWeight).toBeGreaterThanOrEqual(0);

    // Geographic extent overlaps Africa (rough bounds; the source is a Sub-Saharan map).
    const [minX, minY, maxX, maxY] = scaffold.geo.bbox;
    expect(minX).toBeGreaterThanOrEqual(-40); // Atlantic-west edge (real extent ≈ −31°)
    expect(maxX).toBeLessThanOrEqual(65);
    expect(minY).toBeGreaterThanOrEqual(-45);
    expect(maxY).toBeLessThanOrEqual(45);
    expect(maxX).toBeGreaterThan(minX);
    expect(maxY).toBeGreaterThan(minY);
  });
});
