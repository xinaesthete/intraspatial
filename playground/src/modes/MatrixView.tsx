// A K×K channel matrix as a grid of swatches. Two scales, for the two matrices the mode pages show:
// `g` (1 = together as often as chance, >1 together, <1 apart), coloured by log₂ g so doubling and
// halving look equally strong; and `corr`, a correlation over [−1, 1].

import { cssRgb, diverging } from "../../../src/color/ramps";

type MatrixScale = "log2" | "signed";

const RAMP = { centreL: 0.19, endL: 0.8, endC: 0.17 };

const position = (v: number, scale: MatrixScale): number =>
  scale === "log2" ? (v > 0 ? Math.max(-1, Math.min(1, Math.log2(v) / 3)) : -1) : Math.max(-1, Math.min(1, v));

interface Props {
  readonly labels: readonly string[];
  /** Row-major, `values[i*K + j]`. */
  readonly values: ArrayLike<number>;
  readonly scale: MatrixScale;
  /** The hover text for one cell. Default: the pair and its value. */
  readonly describe?: (i: number, j: number) => string;
}

export function MatrixView({ labels, values, scale, describe }: Props) {
  const K = labels.length;
  const text = describe ?? ((i: number, j: number) => `${labels[i]} × ${labels[j]}: ${(values[i * K + j] ?? 0).toFixed(2)}`);
  return (
    <div className="matrix-wrap">
      <div className="matrix" style={{ gridTemplateColumns: `minmax(90px, max-content) repeat(${K}, 18px)` }}>
        {labels.map((a, i) => [
          <div key={`l${a}`} className="matrix-label">
            {i + 1}. {a}
          </div>,
          ...labels.map((b, j) => (
            <div
              key={`${a}|${b}`}
              className="matrix-cell"
              style={{ background: cssRgb(diverging(position(values[i * K + j] ?? 0, scale), RAMP)) }}
              title={text(i, j)}
            />
          )),
        ])}
      </div>
      <p className="hint">Hover a cell for its value. Columns follow the row numbering.</p>
    </div>
  );
}
