#!/usr/bin/env node
// Pack every publishable workspace package exactly as `pnpm publish` would, then lint the tarballs:
// publint (manifest / exports / files) and arethetypeswrong (types resolve under node16 + bundler).
// Also fails on anything in the PACKED manifest a registry consumer can't install (`file:`,
// `link:`, `workspace:`, `catalog:` — pnpm should have rewritten the last two).
//
// Expects dist/ to be built already (`pnpm release` builds first; prepack would rebuild anyway).
// Usage:   node scripts/check-publish.mjs

import { execFileSync } from "node:child_process";
import { mkdtempSync, readdirSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const PACKAGES = [".", "packages/deck"];
const UNINSTALLABLE = /^(file|link|workspace|catalog):/;

const run = (cmd, args, opts = {}) => execFileSync(cmd, args, { stdio: ["ignore", "pipe", "inherit"], encoding: "utf8", ...opts });

const out = mkdtempSync(join(tmpdir(), "intraspatial-pack-"));
let failed = false;
try {
  for (const dir of PACKAGES) {
    const before = new Set(readdirSync(out));
    run("pnpm", ["pack", "--pack-destination", out], { cwd: dir });
    const tarball = join(
      out,
      readdirSync(out).find((f) => !before.has(f)),
    );
    console.log(`\n== ${tarball}`);

    const manifest = JSON.parse(run("tar", ["xzOf", tarball, "package/package.json"]));
    for (const field of ["dependencies", "optionalDependencies", "peerDependencies"]) {
      for (const [name, range] of Object.entries(manifest[field] ?? {})) {
        if (UNINSTALLABLE.test(range)) {
          console.error(`✗ ${manifest.name}: ${field}.${name} = "${range}" won't install from npm`);
          failed = true;
        }
      }
    }

    for (const [cmd, ...args] of [
      ["publint", "run", tarball],
      ["@arethetypeswrong/cli", tarball, "--profile", "esm-only"],
    ]) {
      try {
        execFileSync("pnpm", ["dlx", cmd, ...args], { stdio: "inherit" });
      } catch {
        failed = true;
      }
    }
  }
} finally {
  rmSync(out, { recursive: true, force: true });
}

if (failed) {
  console.error("\ncheck-publish: FAILED");
  process.exit(1);
}
console.log("\ncheck-publish: all packages OK");
