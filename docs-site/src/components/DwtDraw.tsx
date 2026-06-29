/*
 * DwtDraw.tsx — paint and erase directly in the DWT (wavelet) domain and watch
 * the inverse transform resynthesise the image live.
 *
 *   • Forward-transform a source image into the Mallat pyramid.
 *   • Brush over the coefficients: ERASE (kill detail), ADD + / ADD − (inject
 *     signed coefficients → wavelet-shaped ripples appear in the picture).
 *   • Every stroke runs the inverse DWT and re-renders the resynthesised image,
 *     with a live PSNR vs the original.
 *
 * The DWT math is the project's own reference port (`../lib/dwt`).
 */
import { useCallback, useEffect, useRef, useState } from 'react';
import {
  dwt2dForward,
  dwt2dInverse,
  type Band,
  type Decomposition,
  type Kernel,
} from '../lib/dwt';
import { SYNTH_IMAGES, imageElementToPlane, type ImagePlane } from '../lib/dwtImages';

const SIZE = 256; // working resolution

type BrushMode = 'erase' | 'add' | 'sub';

const BRUSH_MODES: { id: BrushMode; label: string; hint: string }[] = [
  { id: 'erase', label: 'Erase', hint: 'zero coefficients — remove detail' },
  { id: 'add', label: 'Add +', hint: 'inject positive detail (warm ripples)' },
  { id: 'sub', label: 'Add −', hint: 'inject negative detail (cool ripples)' },
];

/* ---- coefficient → colour, matching the standalone primer ---- */
function shadeDetail(
  v: number,
  detailMax: number,
  logScale: boolean,
  gain: number,
): [number, number, number] {
  const a = Math.abs(v) / detailMax;
  let s = logScale ? Math.log1p(a * 40) / Math.log1p(40) : a;
  s = Math.min(1, s * gain);
  const mid = 128;
  const amp = 127 * s;
  if (v >= 0) return [mid + amp, mid + amp * 0.55, mid - amp * 0.2];
  return [mid - amp * 0.2, mid + amp * 0.55, mid + amp];
}

function renderDecomposition(
  canvas: HTMLCanvasElement,
  dec: Decomposition,
  coeffs: Float64Array,
  detailMax: number,
  logScale: boolean,
  gain: number,
) {
  const { width, height, bands } = dec;
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d')!;
  const img = ctx.createImageData(width, height);
  const out = img.data;

  // LL is the last band; auto-range it to a grayscale ramp.
  const ll = bands[bands.length - 1];
  let mn = Infinity, mx = -Infinity;
  for (let y = 0; y < ll.h; y++)
    for (let x = 0; x < ll.w; x++) {
      const v = coeffs[(ll.y + y) * width + (ll.x + x)];
      if (v < mn) mn = v;
      if (v > mx) mx = v;
    }
  if (mx === mn) mx = mn + 1;

  for (let y = 0; y < ll.h; y++)
    for (let x = 0; x < ll.w; x++) {
      const v = coeffs[(ll.y + y) * width + (ll.x + x)];
      const g = (255 * (v - mn)) / (mx - mn);
      const idx = ((ll.y + y) * width + (ll.x + x)) * 4;
      out[idx] = out[idx + 1] = out[idx + 2] = g;
      out[idx + 3] = 255;
    }
  for (const b of bands) {
    if (b.type === 'LL') continue;
    for (let y = 0; y < b.h; y++)
      for (let x = 0; x < b.w; x++) {
        const v = coeffs[(b.y + y) * width + (b.x + x)];
        const [r, gg, bb] = shadeDetail(v, detailMax, logScale, gain);
        const idx = ((b.y + y) * width + (b.x + x)) * 4;
        out[idx] = r; out[idx + 1] = gg; out[idx + 2] = bb; out[idx + 3] = 255;
      }
  }
  ctx.putImageData(img, 0, 0);

  // subband grid
  ctx.save();
  ctx.lineWidth = 1;
  let curW = width, curH = height;
  for (let lvl = 0; lvl < dec.levels && curW >= 2 && curH >= 2; lvl++) {
    const lowW = (curW + 1) >> 1, lowH = (curH + 1) >> 1;
    ctx.strokeStyle = 'rgba(90,200,250,0.55)';
    ctx.beginPath();
    ctx.moveTo(lowW + 0.5, 0); ctx.lineTo(lowW + 0.5, curH);
    ctx.moveTo(0, lowH + 0.5); ctx.lineTo(curW, lowH + 0.5);
    ctx.stroke();
    curW = lowW; curH = lowH;
  }
  ctx.restore();
}

function renderPlaneGray(canvas: HTMLCanvasElement, plane: ImagePlane) {
  const { width, height, data } = plane;
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d')!;
  const img = ctx.createImageData(width, height);
  for (let i = 0; i < width * height; i++) {
    let g = data[i];
    g = g < 0 ? 0 : g > 255 ? 255 : g;
    img.data[i * 4] = img.data[i * 4 + 1] = img.data[i * 4 + 2] = g;
    img.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}

export default function DwtDraw() {
  const [sourceKey, setSourceKey] = useState('fractal');
  const [kernel, setKernel] = useState<Kernel>('9/7');
  const [levels, setLevels] = useState(4);
  const [brushMode, setBrushMode] = useState<BrushMode>('erase');
  const [brushRadius, setBrushRadius] = useState(8);
  const [brushStrength, setBrushStrength] = useState(60);
  const [editLL, setEditLL] = useState(false);
  const [logScale, setLogScale] = useState(true);
  const [gain, setGain] = useState(1.5);
  const [metrics, setMetrics] = useState({ psnr: Infinity, modified: 0, energyPct: 100 });
  const [selfTestNote, setSelfTestNote] = useState('');

  const origCanvas = useRef<HTMLCanvasElement>(null);
  const decompCanvas = useRef<HTMLCanvasElement>(null);
  const reconCanvas = useRef<HTMLCanvasElement>(null);
  const uploadInput = useRef<HTMLInputElement>(null);

  // Numerical state that must not trigger React re-renders on every brush move.
  const planeRef = useRef<ImagePlane | null>(null); // original source
  const baseDecRef = useRef<Decomposition | null>(null); // pristine coefficients
  const coeffsRef = useRef<Float64Array | null>(null); // edited working copy
  const detailMaxRef = useRef(1e-6); // stable shading scale from pristine coeffs
  const baseEnergyRef = useRef(1); // pristine detail energy (for the % readout)

  const paintingRef = useRef(false);
  const lastPtRef = useRef<{ x: number; y: number } | null>(null);

  // Live values the rAF render loop reads without being re-created each move.
  const liveRef = useRef({ brushMode, brushRadius, brushStrength, editLL, logScale, gain });
  useEffect(() => {
    liveRef.current = { brushMode, brushRadius, brushStrength, editLL, logScale, gain };
  }, [brushMode, brushRadius, brushStrength, editLL, logScale, gain]);

  /* Re-render the resynthesis from the current coefficients + recompute metrics. */
  const resynth = useCallback(() => {
    const dec = baseDecRef.current;
    const coeffs = coeffsRef.current;
    const plane = planeRef.current;
    if (!dec || !coeffs || !plane) return;

    if (decompCanvas.current) {
      renderDecomposition(
        decompCanvas.current, dec, coeffs, detailMaxRef.current,
        liveRef.current.logScale, liveRef.current.gain,
      );
    }
    const rec = dwt2dInverse({
      data: coeffs, width: dec.width, height: dec.height, levels: dec.levels, kernel: dec.kernel,
    });
    if (reconCanvas.current) renderPlaneGray(reconCanvas.current, rec);

    const n = dec.width * dec.height;
    let mse = 0;
    for (let i = 0; i < n; i++) {
      const d = rec.data[i] - plane.data[i];
      mse += d * d;
    }
    mse /= n;
    const psnr = mse < 1e-9 ? Infinity : 10 * Math.log10((255 * 255) / mse);

    // How much edited detail differs from pristine, + current detail energy.
    let modified = 0, energy = 0;
    const base = dec.data;
    for (const b of dec.bands) {
      if (b.type === 'LL') continue;
      for (let y = 0; y < b.h; y++)
        for (let x = 0; x < b.w; x++) {
          const idx = (b.y + y) * dec.width + (b.x + x);
          if (coeffs[idx] !== base[idx]) modified++;
          energy += coeffs[idx] * coeffs[idx];
        }
    }
    const energyPct = (100 * energy) / baseEnergyRef.current;
    setMetrics({ psnr, modified, energyPct });
  }, []);

  /* Forward-transform the current source → pristine coeffs + working copy. */
  const recompute = useCallback(() => {
    const plane = planeRef.current;
    if (!plane) return;
    const dec = dwt2dForward(plane, kernel, levels);
    baseDecRef.current = dec;
    coeffsRef.current = Float64Array.from(dec.data);

    // Stable detail-shading scale + baseline detail energy from pristine coeffs.
    let detailMax = 1e-6, energy = 1e-6;
    for (const b of dec.bands) {
      if (b.type === 'LL') continue;
      for (let y = 0; y < b.h; y++)
        for (let x = 0; x < b.w; x++) {
          const v = dec.data[(b.y + y) * dec.width + (b.x + x)];
          if (Math.abs(v) > detailMax) detailMax = Math.abs(v);
          energy += v * v;
        }
    }
    detailMaxRef.current = detailMax;
    baseEnergyRef.current = energy;

    if (origCanvas.current) renderPlaneGray(origCanvas.current, plane);
    resynth();
  }, [kernel, levels, resynth]);

  // (Re)transform when the source, kernel or levels change. recompute() is
  // re-created on kernel/levels change, so this fires for those too — including
  // for uploaded images, whose plane is already set by the upload handler.
  useEffect(() => {
    if (sourceKey !== 'upload') planeRef.current = SYNTH_IMAGES[sourceKey].gen(SIZE, SIZE);
    if (planeRef.current) recompute();
  }, [sourceKey, recompute]);

  // One-time round-trip sanity badge.
  useEffect(() => {
    let cancelled = false;
    import('../lib/dwt').then(({ selfTest }) => {
      if (cancelled) return;
      const r = selfTest();
      setSelfTestNote(`round-trip ✓  5/3 err=${r['5/3']}  ·  9/7 err≈${r['9/7'].toExponential(1)}`);
    });
    return () => { cancelled = true; };
  }, []);

  /* ---- brush application ---- */
  const stamp = useCallback((cx: number, cy: number) => {
    const dec = baseDecRef.current;
    const coeffs = coeffsRef.current;
    if (!dec || !coeffs) return;
    const { brushMode, brushRadius, brushStrength, editLL } = liveRef.current;
    const r = brushRadius;
    const r2 = r * r;
    const { width, height, llW, llH } = dec;
    const x0 = Math.max(0, Math.floor(cx - r));
    const x1 = Math.min(width - 1, Math.ceil(cx + r));
    const y0 = Math.max(0, Math.floor(cy - r));
    const y1 = Math.min(height - 1, Math.ceil(cy + r));
    for (let y = y0; y <= y1; y++)
      for (let x = x0; x <= x1; x++) {
        const dx = x - cx, dy = y - cy;
        const d2 = dx * dx + dy * dy;
        if (d2 > r2) continue;
        // Skip the LL approximation block unless explicitly enabled.
        if (!editLL && x < llW && y < llH) continue;
        const idx = y * width + x;
        const t = 1 - Math.sqrt(d2) / r; // linear falloff, 1 at centre
        const falloff = t * t * (3 - 2 * t); // smoothstep
        if (brushMode === 'erase') {
          coeffs[idx] *= 1 - falloff;
        } else if (brushMode === 'add') {
          coeffs[idx] += brushStrength * falloff;
        } else {
          coeffs[idx] -= brushStrength * falloff;
        }
      }
  }, []);

  const eventToCoeff = (ev: React.PointerEvent<HTMLCanvasElement>) => {
    const cv = decompCanvas.current!;
    const rect = cv.getBoundingClientRect();
    return {
      x: ((ev.clientX - rect.left) / rect.width) * cv.width,
      y: ((ev.clientY - rect.top) / rect.height) * cv.height,
    };
  };

  const onPointerDown = (ev: React.PointerEvent<HTMLCanvasElement>) => {
    ev.preventDefault();
    try { decompCanvas.current!.setPointerCapture(ev.pointerId); } catch {}
    paintingRef.current = true;
    const p = eventToCoeff(ev);
    lastPtRef.current = p;
    stamp(p.x, p.y);
    resynth();
  };

  const onPointerMove = (ev: React.PointerEvent<HTMLCanvasElement>) => {
    if (!paintingRef.current) return;
    const p = eventToCoeff(ev);
    const last = lastPtRef.current ?? p;
    // Interpolate so fast drags leave a continuous stroke.
    const dist = Math.hypot(p.x - last.x, p.y - last.y);
    const steps = Math.max(1, Math.ceil(dist / Math.max(1, liveRef.current.brushRadius * 0.5)));
    for (let i = 1; i <= steps; i++) {
      stamp(last.x + ((p.x - last.x) * i) / steps, last.y + ((p.y - last.y) * i) / steps);
    }
    lastPtRef.current = p;
    resynth();
  };

  const endStroke = (ev: React.PointerEvent<HTMLCanvasElement>) => {
    paintingRef.current = false;
    lastPtRef.current = null;
    try { decompCanvas.current!.releasePointerCapture(ev.pointerId); } catch {}
  };

  const reset = useCallback(() => {
    const dec = baseDecRef.current;
    if (!dec) return;
    coeffsRef.current = Float64Array.from(dec.data);
    resynth();
  }, [resynth]);

  const eraseAllDetail = useCallback(() => {
    const dec = baseDecRef.current;
    const coeffs = coeffsRef.current;
    if (!dec || !coeffs) return;
    for (const b of dec.bands) {
      if (b.type === 'LL') continue;
      for (let y = 0; y < b.h; y++)
        for (let x = 0; x < b.w; x++) coeffs[(b.y + y) * dec.width + (b.x + x)] = 0;
    }
    resynth();
  }, [resynth]);

  const onUpload = (ev: React.ChangeEvent<HTMLInputElement>) => {
    const file = ev.target.files?.[0];
    if (!file) return;
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      planeRef.current = imageElementToPlane(img, SIZE);
      setSourceKey('upload');
      // recompute() reads kernel/levels via closure of the effect; call directly.
      recompute();
      URL.revokeObjectURL(url);
    };
    img.src = url;
  };

  const psnrText =
    metrics.psnr === Infinity ? '∞ dB (identical)' : metrics.psnr.toFixed(2) + ' dB';

  return (
    // `not-content` opts the whole subtree out of Starlight's markdown prose
    // spacing, which otherwise injects margin-top into every flex/grid child
    // after the first and misaligns the control columns.
    <div className="dwtdraw not-content">
      <div className="controls">
        <div className="ctl">
          <label>Source</label>
          <div className="row">
            {Object.entries(SYNTH_IMAGES).map(([key, info]) => (
              <button
                key={key}
                type="button"
                className={'chip' + (key === sourceKey ? ' active' : '')}
                onClick={() => setSourceKey(key)}
              >
                {info.label}
              </button>
            ))}
            <button type="button" className="chip" onClick={() => uploadInput.current?.click()}>
              Upload…
            </button>
            <input
              ref={uploadInput}
              type="file"
              accept="image/*"
              style={{ display: 'none' }}
              onChange={onUpload}
            />
          </div>
        </div>
        <div className="ctl">
          <label>Kernel</label>
          <select value={kernel} onChange={(e) => setKernel(e.target.value as Kernel)}>
            <option value="9/7">9/7 irreversible (lossy)</option>
            <option value="5/3">5/3 reversible (lossless)</option>
          </select>
        </div>
        <div className="ctl">
          <label>Levels: {levels}</label>
          <input
            type="range" min={1} max={6} step={1} value={levels}
            onChange={(e) => setLevels(parseInt(e.target.value, 10))}
          />
        </div>
      </div>

      <div className="controls">
        <div className="ctl">
          <label>Brush</label>
          <div className="row" id="brush-modes">
            {BRUSH_MODES.map((m) => (
              <button
                key={m.id}
                type="button"
                title={m.hint}
                className={'chip' + (m.id === brushMode ? ' active' : '')}
                onClick={() => setBrushMode(m.id)}
              >
                {m.label}
              </button>
            ))}
          </div>
        </div>
        <div className="ctl">
          <label>Brush size: {brushRadius}px</label>
          <input
            type="range" min={1} max={40} step={1} value={brushRadius}
            onChange={(e) => setBrushRadius(parseInt(e.target.value, 10))}
          />
        </div>
        <div className="ctl">
          <label>Inject strength: {brushStrength}</label>
          <input
            type="range" min={5} max={300} step={5} value={brushStrength}
            onChange={(e) => setBrushStrength(parseInt(e.target.value, 10))}
          />
        </div>
        <div className="ctl">
          <label>&nbsp;</label>
          <label className="toggle">
            <input type="checkbox" checked={editLL} onChange={(e) => setEditLL(e.target.checked)} />
            edit LL too
          </label>
        </div>
        <div className="ctl">
          <label>&nbsp;</label>
          <div className="row">
            <button type="button" className="btn" onClick={reset}>Reset coeffs</button>
            <button type="button" className="btn" onClick={eraseAllDetail}>Erase all detail</button>
          </div>
        </div>
      </div>

      <div className="metrics">
        <div className="metric"><div className="k">Resynthesis PSNR</div><div className="v good">{psnrText}</div></div>
        <div className="metric"><div className="k">Detail coeffs edited</div><div className="v">{metrics.modified.toLocaleString()}</div></div>
        <div className="metric"><div className="k">Detail energy vs original</div><div className="v">{metrics.energyPct.toFixed(1)}%</div></div>
      </div>

      <div className="grid3">
        <figure>
          <div className="frame"><canvas ref={origCanvas} /></div>
          <figcaption><b>Original</b> · 256×256</figcaption>
        </figure>
        <figure>
          <div className="frame draw">
            <canvas
              ref={decompCanvas}
              onPointerDown={onPointerDown}
              onPointerMove={onPointerMove}
              onPointerUp={endStroke}
              onPointerLeave={endStroke}
              style={{ touchAction: 'none', cursor: 'crosshair' }}
            />
            <span className="drawbadge">draw here ✎</span>
          </div>
          <figcaption><b>DWT domain</b> · brush the coefficients</figcaption>
        </figure>
        <figure>
          <div className="frame"><canvas ref={reconCanvas} /></div>
          <figcaption><b>Resynthesis</b> · inverse DWT, live</figcaption>
        </figure>
      </div>

      <div className="displayrow">
        <label className="toggle">
          <input type="checkbox" checked={logScale} onChange={(e) => setLogScale(e.target.checked)} />
          log detail scale
        </label>
        <label className="ctl inline">
          detail gain {gain.toFixed(1)}×
          <input
            type="range" min={0.5} max={6} step={0.5} value={gain}
            onChange={(e) => setGain(parseFloat(e.target.value))}
          />
        </label>
        {selfTestNote && <span className="selftest">{selfTestNote}</span>}
      </div>

      <style>{css}</style>
    </div>
  );
}

const css = `
.dwtdraw {
  --bg:#0b0f1a; --panel:#131a2b; --ink:#e7ecf6; --ink-dim:#9aa6be;
  --line:#243049; --accent:#5ac8fa; --accent-2:#7bed9f;
  color: var(--ink);
  font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  background: linear-gradient(180deg,#131a2b 0%,#0f1626 100%);
  border: 1px solid var(--line); border-radius: 14px; padding: 18px;
}
.dwtdraw .controls { display:flex; flex-wrap:wrap; gap:14px 20px; align-items:flex-end; margin-bottom:14px; }
.dwtdraw .ctl { display:flex; flex-direction:column; gap:5px; }
.dwtdraw .ctl.inline { flex-direction:row; align-items:center; gap:8px; color:var(--ink-dim); }
.dwtdraw .ctl > label { font-size:11px; color:var(--ink-dim); text-transform:uppercase; letter-spacing:.04em; }
.dwtdraw .row { display:flex; align-items:center; gap:7px; flex-wrap:wrap; }
.dwtdraw select {
  background:#0c1322; color:var(--ink); border:1px solid var(--line);
  border-radius:8px; padding:6px 9px; font:inherit;
}
.dwtdraw input[type="range"] { width:150px; accent-color:var(--accent); }
.dwtdraw .chip {
  background:#0c1322; color:var(--ink-dim); border:1px solid var(--line);
  border-radius:999px; padding:6px 12px; font:inherit; font-size:12.5px; cursor:pointer; transition:all .12s;
}
.dwtdraw .chip:hover { color:var(--ink); border-color:#34507f; }
.dwtdraw .chip.active { background:rgba(90,200,250,.16); color:var(--accent); border-color:rgba(90,200,250,.5); }
.dwtdraw .btn {
  background:#0c1322; color:var(--ink); border:1px solid var(--line);
  border-radius:8px; padding:7px 12px; font:inherit; cursor:pointer; transition:all .12s;
}
.dwtdraw .btn:hover { border-color:#34507f; }
.dwtdraw .toggle { display:flex; align-items:center; gap:7px; cursor:pointer; color:var(--ink-dim); font-size:13px; }
.dwtdraw .metrics { display:flex; flex-wrap:wrap; gap:12px; margin:6px 0 16px; }
.dwtdraw .metric { flex:1 1 150px; background:#0c1322; border:1px solid var(--line); border-radius:10px; padding:10px 13px; }
.dwtdraw .metric .k { font-size:10.5px; color:var(--ink-dim); text-transform:uppercase; letter-spacing:.05em; }
.dwtdraw .metric .v { font-size:19px; font-weight:650; margin-top:3px; }
.dwtdraw .metric .v.good { color:var(--accent-2); }
.dwtdraw .grid3 { display:grid; grid-template-columns:repeat(3,1fr); gap:14px; }
.dwtdraw figure { margin:0; }
.dwtdraw .frame { position:relative; border:1px solid var(--line); border-radius:10px; overflow:hidden; background:#05080f; line-height:0; }
.dwtdraw .frame.draw { border-color:rgba(90,200,250,.5); box-shadow:0 0 0 1px rgba(90,200,250,.25); }
.dwtdraw canvas { width:100%; height:auto; display:block; image-rendering:pixelated; }
.dwtdraw .drawbadge {
  position:absolute; top:8px; left:8px; font-size:11px; line-height:1; padding:4px 8px; border-radius:999px;
  background:rgba(90,200,250,.18); color:var(--accent); border:1px solid rgba(90,200,250,.5);
  pointer-events:none;
}
.dwtdraw figcaption { font-size:12px; color:var(--ink-dim); margin-top:7px; text-align:center; }
.dwtdraw figcaption b { color:var(--ink); }
.dwtdraw .displayrow { display:flex; flex-wrap:wrap; gap:16px; align-items:center; margin-top:14px; font-size:13px; }
.dwtdraw .selftest { margin-left:auto; font:11px/1 ui-monospace, monospace; color:var(--accent-2); }
@media (max-width:760px) { .dwtdraw .grid3 { grid-template-columns:1fr; } }
`;
