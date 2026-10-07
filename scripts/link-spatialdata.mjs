#!/usr/bin/env node
/**
 * Point the playground at a local SpatialData.js checkout (its built `dist`) via pnpm `link:`,
 * for co-developing against an unpublished change. Ported from MDV's script of the same name.
 *
 *   SPATIALDATA_ROOT=~/code/www/SpatialData.ts pnpm link:spatialdata
 *   pnpm unlink:spatialdata
 *
 * The checkout must be built (`pnpm build` there). Rebuild it after upstream edits, then restart
 * Vite. `playground/vite.config.ts` notices the link on its own (aliases, `fs.allow`, dedupe).
 *
 * No `overrides`: nothing else in this workspace depends on these packages, so linking the
 * direct dependencies is the whole job.
 */

import { execSync } from "node:child_process";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { fileURLToPath } from "node:url";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const manifestPath = path.join(repoRoot, "playground/package.json");

const PACKAGE_DIRS = {
  "@spatialdata/core": "packages/core",
  "@spatialdata/layers": "packages/layers",
  "@spatialdata/react": "packages/react",
  "@spatialdata/vis": "packages/vis",
  zarrextra: "packages/zarrextra",
};

/** What `unlink` restores: the playground's own published dependencies, so keep it in step with
 *  playground/package.json. Linked packages not listed here (layers, react: reached through vis)
 *  are removed again on unlink. */
const PUBLISHED_RANGES = {
  "@spatialdata/core": "^0.12.0",
  "@spatialdata/vis": "^0.12.0",
  zarrextra: "0.5.1",
};

const expandHome = (p) => (p.startsWith("~/") ? path.join(os.homedir(), p.slice(2)) : p);

function resolveRoot() {
  const candidates = [process.env.SPATIALDATA_ROOT?.trim(), "~/code/www/SpatialData.ts"].filter(Boolean).map(expandHome);
  for (const root of candidates) if (fs.existsSync(path.join(root, "packages/vis/package.json"))) return path.resolve(root);
  throw new Error(`No SpatialData.js checkout found; set SPATIALDATA_ROOT (tried: ${candidates.join(", ")})`);
}

function assertBuilt(root) {
  const missing = Object.entries(PACKAGE_DIRS)
    .filter(([, rel]) => !fs.existsSync(path.join(root, rel, "dist/index.js")))
    .map(([name, rel]) => `${name} (${rel}/dist)`);
  if (missing.length) throw new Error(`Build the checkout first (pnpm build in ${root}). Missing:\n  ${missing.join("\n  ")}`);
}

const read = () => JSON.parse(fs.readFileSync(manifestPath, "utf8"));
const write = (pkg) => fs.writeFileSync(manifestPath, `${JSON.stringify(pkg, null, 2)}\n`);

function link() {
  const root = resolveRoot();
  assertBuilt(root);
  const pkg = read();
  for (const [name, rel] of Object.entries(PACKAGE_DIRS)) pkg.dependencies[name] = `link:${path.join(root, rel)}`;
  write(pkg);
  console.log(`Linked @spatialdata/{core,layers,react,vis} + zarrextra → ${root}`);
  execSync("pnpm install", { cwd: repoRoot, stdio: "inherit" });
  console.log("Restart Vite with a cold cache: rm -rf playground/node_modules/.vite");
}

function unlink() {
  const pkg = read();
  for (const name of Object.keys(PACKAGE_DIRS)) delete pkg.dependencies[name];
  for (const [name, range] of Object.entries(PUBLISHED_RANGES)) pkg.dependencies[name] = range;
  pkg.dependencies = Object.fromEntries(Object.entries(pkg.dependencies).sort(([a], [b]) => a.localeCompare(b)));
  write(pkg);
  console.log("Restored the published @spatialdata/core, vis and zarrextra ranges");
  execSync("pnpm install", { cwd: repoRoot, stdio: "inherit" });
}

if (process.argv.includes("--unlink")) unlink();
else link();
