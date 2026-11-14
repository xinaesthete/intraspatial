// Gene modules off the main thread: the PCA gene embedding and its e-means partition take ~6 s on a
// 4.6M-molecule Atera sample, which on the host would freeze pan and hover for the whole run.
// geneModules.ts is plain arithmetic, so it loads here unchanged.

import { binSizeFor, geneEmbedding, geneModules } from "../../../src/spatial/geneModules";

export interface GeneModulesRequest {
  readonly xs: Float32Array;
  readonly ys: Float32Array;
  readonly codes: Int32Array;
  readonly nFeatures: number;
  /** Codes to leave out (negative controls). */
  readonly exclude: Int32Array;
  /** Area the sample covers, to size bins for ~750 molecules each. */
  readonly area: number;
  readonly k: number;
}

export interface GeneModulesSummary {
  /** Disjoint gene sets as feature codes, most molecules first. */
  readonly modules: readonly (readonly number[])[];
  readonly molecules: number;
  readonly binSize: number;
  readonly bins: number;
  readonly components: number;
  /** Share of the standardised variance the kept components explain. */
  readonly explained: number;
  readonly genesEmbedded: number;
  readonly unstructured: number;
  readonly dropped: { readonly excluded: number; readonly rare: number };
  readonly ms: number;
}

export type GeneModulesResponse =
  | { readonly ok: true; readonly summary: GeneModulesSummary }
  | { readonly ok: false; readonly error: string };

self.onmessage = (e: MessageEvent<GeneModulesRequest>) => {
  const { xs, ys, codes, nFeatures, exclude, area, k } = e.data;
  const t0 = performance.now();
  try {
    const binSize = binSizeFor(codes.length, area);
    const emb = geneEmbedding(xs, ys, codes, nFeatures, { binSize, exclude: new Set(exclude) });
    const mods = geneModules(emb, k);
    let explained = 0;
    for (let c = 0; c < emb.components; c++) explained += emb.explained[c]!;
    const summary: GeneModulesSummary = {
      modules: mods.modules.map((m) => m.codes),
      molecules: emb.molecules,
      binSize,
      bins: emb.bins,
      components: emb.components,
      explained,
      genesEmbedded: emb.codes.length,
      unstructured: mods.unstructured.length,
      dropped: emb.dropped,
      ms: performance.now() - t0,
    };
    postMessage({ ok: true, summary } satisfies GeneModulesResponse);
  } catch (err) {
    postMessage({ ok: false, error: err instanceof Error ? err.message : String(err) } satisfies GeneModulesResponse);
  }
};
