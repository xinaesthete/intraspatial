#!/usr/bin/env node
// Give the relative imports in emitted `.js` / `.d.ts` files explicit `.js` / `/index.js` endings.
//
// Why: `src/` is written for `moduleResolution: Bundler`, so its relative imports are
// extensionless (`./gpu/device`). tsc copies specifiers verbatim, so its output fails wherever ESM
// resolution is strict: Node itself, webpack's fully-specified rule for `"type": "module"`, and
// consumers typechecking under `node16` / `nodenext`. Vite's lib build (the root package's JS)
// already writes `.js` endings; tsc's declarations, and `packages/*` built by tsc alone, don't.
//
// It also turns `#src/…` into the equivalent relative path. `#src/*` is the root package.json
// "imports" alias for src/*.ts — an in-repo convenience: src/ is not published, so dist/ must not
// depend on it, and tsc copies it into the declarations verbatim. `<distDir>` mirrors src/
// (rootDir / preserveModulesRoot), so `#src/x` is `<distDir>/x`.
//
// Usage:   node scripts/fix-esm-specifiers.mjs <distDir> [<distDir>...]

import { existsSync, readdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { dirname, join, relative, resolve } from "node:path";

const dirs = process.argv.slice(2);
if (dirs.length === 0) {
  console.error("fix-esm-specifiers: usage: node scripts/fix-esm-specifiers.mjs <distDir>...");
  process.exit(2);
}

// `from "./x"`, `import("./x")`, `import "./x"` — relative (or `#src/`) specifiers with no extension.
const SPECIFIER = /(\bfrom\s*|\bimport\s*\(\s*|\bimport\s+)(["'])(\.\.?\/[^"']*?|#src\/[^"']*?)\2/g;

function* walk(dir) {
  for (const name of readdirSync(dir)) {
    const path = join(dir, name);
    if (statSync(path).isDirectory()) yield* walk(path);
    else if (/\.(d\.ts|js)$/.test(name)) yield path;
  }
}

/** `#src/x` as the relative path from `file` to `<root>/x`; any other specifier unchanged. */
function fromAlias(file, root, spec) {
  if (!spec.startsWith("#src/")) return spec;
  const rel = relative(dirname(file), join(root, spec.slice("#src/".length)));
  return rel.startsWith(".") ? rel : `./${rel}`;
}

/** The relative `spec` with its `.js` / `/index.js` ending, or undefined when it resolves to nothing. */
function withExtension(file, spec) {
  const base = resolve(dirname(file), spec);
  const ext = file.endsWith(".d.ts") ? ".d.ts" : ".js";
  if (existsSync(`${base}${ext}`)) return `${spec}.js`;
  if (existsSync(join(base, `index${ext}`))) return `${spec.replace(/\/$/, "")}/index.js`;
  return undefined;
}

let files = 0;
let rewritten = 0;
const unresolved = [];
for (const dir of dirs) {
  const root = resolve(dir);
  for (const file of walk(root)) {
    const text = readFileSync(file, "utf8");
    const out = text.replace(SPECIFIER, (match, lead, quote, raw) => {
      const spec = fromAlias(file, root, raw);
      if (/\.(m?js|json)$/.test(spec)) return match;
      const fixed = withExtension(file, spec);
      if (!fixed) {
        unresolved.push(`${file}: ${spec}`);
        return match;
      }
      rewritten++;
      return `${lead}${quote}${fixed}${quote}`;
    });
    if (out !== text) {
      writeFileSync(file, out);
      files++;
    }
  }
}

console.log(`fix-esm-specifiers: ${rewritten} specifiers in ${files} files`);
if (unresolved.length > 0) {
  console.error(`fix-esm-specifiers: ${unresolved.length} relative specifiers did not resolve:`);
  for (const u of unresolved.slice(0, 20)) console.error(`  ${u}`);
  process.exit(1);
}
