// Mirror the built playground prototypes into the docs site's public/ so Astro serves them at
// <base>/playground/ (the "Prototypes" page links to each). `playground/` stays the single source of
// truth; this copy is generated and git-ignored — the same arrangement as sync-primer.mjs.
//
// The playground is a separate Vite multi-page app, so it must be BUILT first
// (`pnpm build:playground`, which CI runs before the docs build). If its dist is missing we warn and
// skip rather than fail: a docs-only build should still succeed, it just won't carry the prototypes.
// The playground build sets `base: "./"` precisely so its assets resolve under this sub-path.
import { access, cp, mkdir, readdir, rm } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const src = resolve(here, "../../playground/dist");
const dest = resolve(here, "../public/playground");

const exists = await access(src).then(
  () => true,
  () => false,
);
if (!exists) {
  console.warn(`[sync-playground] ${src} not found — skipping. Run \`pnpm build:playground\` first to include the prototypes.`);
  process.exit(0);
}

await rm(dest, { recursive: true, force: true });
await mkdir(dest, { recursive: true });
await cp(src, dest, { recursive: true });

const pages = (await readdir(dest)).filter((f) => f.endsWith(".html")).sort();
console.log(`[sync-playground] ${pages.length} pages → public/playground/ (${pages.join(", ")})`);
