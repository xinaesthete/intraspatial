// `@intraspatial/deck`'s DensityLayer on synthetic points — no store, no sd.js — so the page tests
// the layer alone: units, levels, the reused grid as the view moves, and two layers splatting in the
// same frame (the "two categories" mode, which is how a host draws one contour set per category).

import { OrthographicView, type OrthographicViewState } from "@deck.gl/core";
import { ScatterplotLayer } from "@deck.gl/layers";
import DeckGL from "@deck.gl/react";
import { type Color, DensityLayer } from "@intraspatial/deck";
import type { Device } from "@luma.gl/core";
import { webgpuAdapter } from "@luma.gl/webgpu";
import { useMemo, useState } from "react";
import { Explain } from "../transcript/Explain";

const EXTENT = 2000;

/** A deterministic cloud: Gaussian clusters of different spreads, a ring, and a sparse background.
 *  `cat` splits it in two — even clusters and the ring are category 0, the rest category 1. */
function makeCloud(n: number) {
  let s = 0x9e3779b9;
  const rnd = () => {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const gauss = () => Math.sqrt(-2 * Math.log(rnd() || 1e-12)) * Math.cos(2 * Math.PI * rnd());
  const clusters = [
    [400, 500, 60],
    [700, 1400, 140],
    [1300, 600, 30],
    [1500, 1500, 90],
    [1050, 1050, 220],
    [1700, 300, 45],
  ] as const;
  const x = new Float32Array(n);
  const y = new Float32Array(n);
  const cat = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    const u = rnd();
    if (u < 0.08) {
      x[i] = rnd() * EXTENT;
      y[i] = rnd() * EXTENT;
      cat[i] = i & 1;
    } else if (u < 0.2) {
      const a = rnd() * 2 * Math.PI;
      const r = 300 + gauss() * 12;
      x[i] = 500 + r * Math.cos(a) + 900;
      y[i] = 1000 + r * Math.sin(a) - 100;
      cat[i] = 0;
    } else {
      const k = Math.floor(rnd() * clusters.length);
      const [cx, cy, sd] = clusters[k]!;
      x[i] = cx + gauss() * sd;
      y[i] = cy + gauss() * sd;
      cat[i] = k % 2;
    }
  }
  return { x, y, cat };
}

const BLUES: Color[] = [
  [198, 219, 239],
  [107, 174, 214],
  [33, 113, 181],
  [8, 48, 107],
];
const ORANGES: Color[] = [
  [253, 208, 162],
  [253, 141, 60],
  [217, 72, 1],
  [127, 39, 4],
];

const INITIAL_VIEW: OrthographicViewState = { target: [EXTENT / 2, EXTENT / 2, 0], zoom: -1.2 };
const VIEW = new OrthographicView({ id: "ortho", flipY: false });
/** Module-level, so deck doesn't rebuild its device. The layer needs `float32-blendable`. It does
 *  NOT need intraspatial to adopt the device: it passes deck's own to the kernels. */
const DEVICE_PROPS = { type: "webgpu" as const, adapters: [webgpuAdapter], optionalFeatures: ["float32-blendable" as const] };

export function DensityDemo() {
  const [deviceStatus, setDeviceStatus] = useState("waiting for deck's device…");
  const [count, setCount] = useState(200_000);
  const [bandwidth, setBandwidth] = useState(25);
  const [scale, setScale] = useState<"relative" | "absolute">("relative");
  const [fillLevel, setFillLevel] = useState(0.1);
  const [interval, setLineInterval] = useState(0.2);
  const [fillOpacity, setFillOpacity] = useState(0.6);
  const [edgeOpacity, setEdgeOpacity] = useState(1);
  const [split, setSplit] = useState(false);
  const [showPoints, setShowPoints] = useState(false);
  // Bumping the canvas key remounts deck: a new Deck, a new GPU device, the old one destroyed — what
  // MDV's backend switch or an error-boundary reset does.
  const [mount, setMount] = useState(0);

  const cloud = useMemo(() => makeCloud(count), [count]);
  // One weight vector per category: a point counts 1 in its own category's field, 0 in the other's.
  const weights = useMemo(() => {
    const w0 = new Float32Array(count);
    const w1 = new Float32Array(count);
    for (let i = 0; i < count; i++) (cloud.cat[i] ? w1 : w0)[i] = 1;
    return [w0, w1] as const;
  }, [cloud, count]);

  // In absolute mode the sliders' levels are densities. A uniform cloud of `count` points over the
  // square has density count / EXTENT², so express the sliders in multiples of that.
  const mean = count / (EXTENT * EXTENT);
  const level = (v: number) => (scale === "relative" ? v : v * mean * 20);

  const shared = {
    x: cloud.x,
    y: cloud.y,
    bandwidth,
    scale,
    fillLevel: level(fillLevel),
    contourInterval: interval > 0 ? level(interval) : null,
    fillOpacity,
    edgeOpacity,
  };
  const layers = [
    ...(split
      ? [
          new DensityLayer({ ...shared, id: "density-0", weights: weights[0], colorRange: BLUES }),
          new DensityLayer({ ...shared, id: "density-1", weights: weights[1], colorRange: ORANGES }),
        ]
      : [new DensityLayer({ ...shared, id: "density" })]),
    ...(showPoints && count <= 200_000
      ? [
          new ScatterplotLayer({
            id: "points",
            data: { length: count },
            getPosition: (_: unknown, { index }: { index: number }) => [cloud.x[index]!, cloud.y[index]!],
            getFillColor: [255, 255, 255, 90],
            radiusUnits: "pixels",
            getRadius: 1,
            updateTriggers: { getPosition: cloud },
          }),
        ]
      : []),
  ];

  return (
    <>
      <h2 className="page">Density layer (WebGPU)</h2>
      <p className="page">
        A kernel density estimate of the points, computed and drawn on the GPU deck renders with. The bandwidth is in the data's own units,
        so the shapes don't change as you zoom; only the detail does.
      </p>
      <div className="layout">
        <div className="controls">
          <label>
            Points
            <select value={count} onChange={(e) => setCount(Number(e.target.value))}>
              <option value={20_000}>20 thousand</option>
              <option value={200_000}>200 thousand</option>
              <option value={2_000_000}>2 million</option>
            </select>
          </label>
          <label>
            Bandwidth σ: {bandwidth.toFixed(1)} data units
            <input
              type="range"
              min={Math.log(2)}
              max={Math.log(300)}
              step={0.01}
              value={Math.log(bandwidth)}
              onChange={(e) => setBandwidth(Math.exp(Number(e.target.value)))}
            />
          </label>
          <label>
            Levels
            <select value={scale} onChange={(e) => setScale(e.target.value as "relative" | "absolute")}>
              <option value="relative">relative: fractions of the peak</option>
              <option value="absolute">absolute: points per unit area</option>
            </select>
          </label>
          <label>
            Fill above: {scale === "relative" ? `${(fillLevel * 100).toFixed(0)}% of peak` : `${level(fillLevel).toPrecision(2)} /unit²`}
            <input type="range" min={0} max={1} step={0.01} value={fillLevel} onChange={(e) => setFillLevel(Number(e.target.value))} />
          </label>
          <label>
            Lines every:{" "}
            {interval > 0
              ? scale === "relative"
                ? `${(interval * 100).toFixed(0)}% of peak`
                : `${level(interval).toPrecision(2)} /unit²`
              : "off"}
            <input type="range" min={0} max={0.5} step={0.01} value={interval} onChange={(e) => setLineInterval(Number(e.target.value))} />
          </label>
          <label>
            Fill opacity: {fillOpacity.toFixed(2)}
            <input type="range" min={0} max={1} step={0.01} value={fillOpacity} onChange={(e) => setFillOpacity(Number(e.target.value))} />
          </label>
          <label>
            Edge opacity: {edgeOpacity.toFixed(2)}
            <input type="range" min={0} max={1} step={0.01} value={edgeOpacity} onChange={(e) => setEdgeOpacity(Number(e.target.value))} />
          </label>
          <label className="check">
            <input type="checkbox" checked={split} onChange={(e) => setSplit(e.target.checked)} />
            Two categories (two layers, one weight vector each)
          </label>
          <label className="check">
            <input type="checkbox" checked={showPoints} disabled={count > 200_000} onChange={(e) => setShowPoints(e.target.checked)} />
            Show points (up to 200 thousand)
          </label>
          <button type="button" onClick={() => setMount((m) => m + 1)}>
            Remount canvas (new GPU device)
          </button>
          <div className="status">
            {deviceStatus}
            {mount > 0 && ` · remounted ${mount}×`}
          </div>
        </div>
        <div className="views">
          <div className="deck-canvas">
            <DeckGL
              key={mount}
              views={VIEW}
              initialViewState={INITIAL_VIEW}
              controller
              layers={layers}
              deviceProps={DEVICE_PROPS}
              onDeviceInitialized={(d: Device) =>
                setDeviceStatus(
                  d.type === "webgpu"
                    ? `WebGPU · float32-blendable ${d.features.has("float32-blendable") ? "✓" : "✗ (the layer needs it)"}`
                    : `deck chose ${d.type}; this page needs WebGPU`,
                )
              }
            />
          </div>
          <div className="explainers">
            <Explain
              title="What the colours mean"
              math={String.raw`\hat f(p) = \frac{1}{2\pi\sigma^2}\sum_i w_i \exp\!\left(-\frac{\lVert p - p_i\rVert^2}{2\sigma^2}\right)`}
            >
              Every point is replaced by a small smooth bump of width σ, and the bumps are added up. Where many points sit close together
              the bumps pile up and the density is high. With absolute levels the numbers are points per unit area, so a fill level means
              the same thing on any data; with relative levels they are fractions of the highest density in view.
            </Explain>
          </div>
        </div>
      </div>
    </>
  );
}
