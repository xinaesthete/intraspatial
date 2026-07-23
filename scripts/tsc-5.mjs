#!/usr/bin/env node
// Run the STABLE JS TypeScript 5.x `tsc`, NOT the default 7.0.2 native compiler.
//
// Why: 7.0.2 (the native-port line) balloons to tens of GB on this project's type surface — it took
// the host to the macOS OOM wall twice (see the toolchain memory / docs). 5.9.3 (present transitively)
// finishes the same check in ~6s and cannot balloon (JS, mature). We resolve it from the pnpm store by
// glob so this doesn't hard-code a path that shifts on reinstall; if 5.x ever isn't present, add
// `"typescript-5": "npm:typescript@5.9.3"` as a devDependency and point this at it.
//
// Launch it as a single node process (this file → node tsc) so scripts/guarded.mjs can track and cap
// the whole tree — never via `pnpm --filter`, whose process layering let the native tsc reparent out
// of the guard's view (the reason the watchdog missed a runaway once).
//
// Usage: node scripts/tsc-5.mjs [tsc args...]

import { spawn } from "node:child_process";
import { existsSync, readdirSync } from "node:fs";

const STORE = "node_modules/.pnpm";
const dir = existsSync(STORE)
  ? readdirSync(STORE)
      .filter((d) => /^typescript@5\./.test(d))
      .sort()
      .pop()
  : undefined;

if (!dir) {
  console.error("tsc-5: no typescript@5.x found in node_modules/.pnpm — add `\"typescript-5\": \"npm:typescript@5.9.3\"` as a devDependency.");
  process.exit(2);
}

const tsc = `${STORE}/${dir}/node_modules/typescript/bin/tsc`;
const child = spawn(process.execPath, [tsc, ...process.argv.slice(2)], { stdio: "inherit" });
child.on("exit", (code, signal) => process.exit(signal ? 1 : (code ?? 0)));
child.on("error", (e) => {
  console.error("tsc-5: failed to launch", tsc, e.message);
  process.exit(2);
});
