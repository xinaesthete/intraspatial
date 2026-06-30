// Mirror the standalone, dependency-free DWT primer (`viz/`) into the docs site's
// public/ so Astro serves it at /dwt-primer/ (the interactive `concepts/dwt-draw`
// page links to it as its "static sibling"). `viz/` stays the single source of
// truth; this copy is generated and git-ignored. Run by the dev/build scripts.
import { cp, mkdir, rm } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const src = resolve(here, "../../viz");
const dest = resolve(here, "../public/dwt-primer");

// The primer is a flat set of static files (index.html + its three scripts). Copy
// those, not the README (it's developer-facing, not part of the served page).
const FILES = ["index.html", "dwt.js", "images.js", "app.js"];

await rm(dest, { recursive: true, force: true });
await mkdir(dest, { recursive: true });
for (const f of FILES) {
  await cp(resolve(src, f), resolve(dest, f));
}
console.log(`[sync-primer] ${FILES.length} files → public/dwt-primer/`);
