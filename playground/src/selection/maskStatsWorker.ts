// The mask panel's cross-PCF, off the main thread: two `crossPCFMatrixBinned` passes, one inside the
// mask and one over the analysis window. On a whole Xenium window each takes ~1.7 s, which on the
// host froze hover and pan for the whole run. pcf.ts, domain.ts and edgeCorrection.ts are plain
// arithmetic, nothing GPU- or DOM-bound, so they load here unchanged.

import { type MaskDomain, rectDomain } from "../../../src/spatial/domain";
import { crossPCFMatrixBinned, type PcfMatrixBinnedResult } from "../../../src/spatial/pcf";

export interface MaskStatsRequest {
  readonly xs: Float64Array;
  readonly ys: Float64Array;
  readonly typeId: Int32Array;
  readonly nTypes: number;
  readonly domain: MaskDomain;
  readonly window: readonly [number, number, number, number];
  readonly rMax: number;
  readonly nBins: number;
}

export type MaskStatsResponse =
  | { readonly ok: true; readonly inMask: PcfMatrixBinnedResult; readonly inWindow: PcfMatrixBinnedResult }
  | { readonly ok: false; readonly error: string };

self.onmessage = (e: MessageEvent<MaskStatsRequest>) => {
  const { xs, ys, typeId, nTypes, domain, window, rMax, nBins } = e.data;
  try {
    // `LabelledCells` takes plain arrays; the copy is milliseconds against seconds of counting.
    const cells = { xs: Array.from(xs), ys: Array.from(ys), typeId: Array.from(typeId) };
    const common = { bbox: window, rMax, nBins, nTypes, edgeCorrected: true };
    const inMask = crossPCFMatrixBinned(cells, { ...common, domain });
    const inWindow = crossPCFMatrixBinned(cells, { ...common, domain: rectDomain(window) });
    const res: MaskStatsResponse = { ok: true, inMask, inWindow };
    postMessage(res, { transfer: [inMask.g.buffer, inMask.pairs.buffer, inWindow.g.buffer, inWindow.pairs.buffer] });
  } catch (err) {
    const res: MaskStatsResponse = { ok: false, error: err instanceof Error ? err.message : String(err) };
    postMessage(res);
  }
};
