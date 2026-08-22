# IntraSpatial docs site

The documentation site for [`intraspatial`](../README.md), built with
[Astro](https://astro.build) + [Starlight](https://starlight.astro.build). Beyond prose,
it hosts **interactive demos** as React islands (`@astrojs/react`) and renders maths with
KaTeX (`remark-math` + `rehype-katex`) — the same KaTeX setup the playground uses.

## Run

From the **repo root** (preferred — uses the workspace + Volta Node 22 pin):

```sh
pnpm dev:docs        # → http://localhost:4321
pnpm build:docs      # production build to docs-site/dist/
```

Or from inside this directory:

```sh
pnpm dev             # astro dev → http://localhost:4321
pnpm build           # astro build
pnpm preview         # preview the production build
```

## Structure

- `src/content/docs/` — the pages. Each `.md`/`.mdx` file is a route based on its path
  (e.g. `concepts/operation-graphs.md` → `/concepts/operation-graphs`).
- `src/components/` & `src/lib/` — React islands and their logic, e.g. the
  **“Draw in the DWT domain”** demo and `src/lib/dwt.ts` (the DWT math ported from the
  toolbox so the demo stays numerically identical).
- `src/assets/` — images embedded in Markdown via relative links.
- `public/` — static assets (favicons, etc.).
- `astro.config.mjs` — Starlight config, including the React, `remark-math`, and
  `rehype-katex` integrations.

## Authoring notes

- **Maths:** inline `$…$` and display `$$…$$` render via KaTeX. The KaTeX stylesheet is
  wired in `astro.config.mjs`.
- **Interactive demos:** add a React component and mount it as an island with a
  `client:*` directive (e.g. `client:visible`) from an `.mdx` page. Keep any shared math
  in `src/lib/` so the docs and the toolbox don't drift.
