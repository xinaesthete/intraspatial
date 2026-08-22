// @ts-check

import react from "@astrojs/react";
import starlight from "@astrojs/starlight";
import { defineConfig, passthroughImageService } from "astro/config";
import rehypeKatex from "rehype-katex";
import remarkMath from "remark-math";
// Transpiles `"use gpu"` TypeGPU kernels (tgpu.fn / computeFn) to WGSL at build time — the
// dancer's GPU-resident sim (src/gpu/sim/dancerGpu.ts) needs it, same as vitest.gpu.config.ts.
import typegpu from "unplugin-typegpu/vite";

// Deployed to GitHub Pages as a *project* site, so it lives under a sub-path
// (https://xinaesthete.github.io/intraspatial/). `site` + `base` must match the repo
// name; Starlight prepends `base` to all its generated nav/asset links for us.
const SITE = "https://xinaesthete.github.io";
const BASE = "/intraspatial";

// Author-written root-absolute links in Markdown/MDX bodies (e.g. `](/primitives/anni/)`)
// are NOT base-prefixed by Astro — only Starlight's own nav is. Without this they 404 on
// Pages. This rehype pass rewrites `href`/`src` that start with a single `/` to include
// `base`, so content stays authored as clean site-root paths.
function rehypeBasePaths() {
  const rewrite = (url) => {
    if (typeof url !== "string" || !url.startsWith("/") || url.startsWith("//")) return url;
    if (url === BASE || url.startsWith(`${BASE}/`)) return url; // already prefixed
    return BASE + url;
  };
  const visit = (node) => {
    if (node.type === "element" && node.properties) {
      if (node.properties.href) node.properties.href = rewrite(node.properties.href);
      if (node.properties.src) node.properties.src = rewrite(node.properties.src);
    }
    if (node.children) for (const child of node.children) visit(child);
  };
  return (tree) => visit(tree);
}

// https://astro.build/config
export default defineConfig({
  site: SITE,
  base: BASE,
  // Avoid the native `sharp` image pipeline (its build script is skipped under
  // pnpm); these docs use no optimised images.
  image: { service: passthroughImageService() },
  // Math: `$inline$` / `$$display$$` in Markdown → KaTeX (CSS loaded via Starlight
  // customCss below). Shared with the playground's KaTeX op-help rendering.
  markdown: { remarkPlugins: [remarkMath], rehypePlugins: [rehypeKatex, rehypeBasePaths] },
  // Let components import the toolbox's own source (one level up, e.g.
  // `../../../src/spatial/...`) so demos are driven by the real library code
  // rather than re-ported copies.
  vite: { server: { fs: { allow: [".."] } }, plugins: [typegpu()] },
  integrations: [
    react(),
    starlight({
      title: "GPU Spatial Primitives",
      description: "Composable, interpretable GPU primitives for discrete-cell spatial analysis (TypeGPU / WebGPU).",
      customCss: ["katex/dist/katex.min.css"],
      social: [{ icon: "github", label: "GitHub", href: "https://github.com/xinaesthete/intraspatial" }],
      sidebar: [
        {
          label: "Start here",
          items: [
            { label: "Overview", slug: "index" },
            { label: "Viewer prototypes", slug: "prototypes" },
            { label: "Composable & interpretable", slug: "concepts/composable-interpretable" },
            { label: "FAIR by design", slug: "concepts/fair" },
            { label: "Roadmap & status", slug: "roadmap" },
          ],
        },
        {
          label: "Datasource & rendering",
          items: [
            // Prototype — surfaced for visibility; scope/structure to be reviewed later.
            { label: "Brick-atlas page table", slug: "datasource/brick-atlas-page-table" },
            { label: "Multi-image scene editor", slug: "datasource/spatial-scene" },
          ],
        },
        {
          label: "Procedural geometry",
          items: [
            // First slice (ADR-0010) — the swept horn grammar + a self-contained Mutator demo.
            { label: "The swept horn grammar", slug: "geometry/horn-grammar" },
          ],
        },
        {
          label: "Concepts",
          items: [
            { label: "Windowing, not quadrats", slug: "concepts/windowing" },
            { label: "Render vs compute", slug: "concepts/render-vs-compute" },
            { label: "Operation graphs", slug: "concepts/operation-graphs" },
            { label: "Fuzzy TDA", slug: "concepts/fuzzy-tda" },
            { label: "Filtrations & persistence", slug: "concepts/filtrations" },
            { label: "Draw in the DWT domain", slug: "concepts/dwt-draw" },
          ],
        },
        {
          label: "Primitives",
          items: [
            { label: "Nearest-neighbour distance", slug: "primitives/nearest-neighbour-distance" },
            { label: "Average Nearest Neighbour Index", slug: "primitives/anni" },
            { label: "Empty-space function", slug: "primitives/empty-space" },
            { label: "KDE density splat", slug: "primitives/kde-splat" },
            { label: "Separable convolution", slug: "primitives/separable-convolution" },
            { label: "Getis-Ord hotspots", slug: "primitives/getis-ord-hotspots" },
            { label: "Fuzzy adjacency", slug: "primitives/fuzzy-adjacency" },
            { label: "k-th neighbour distance", slug: "primitives/kth-neighbour-distance" },
            { label: "CkNN rescaled distance", slug: "primitives/cknn" },
          ],
        },
      ],
    }),
  ],
});
