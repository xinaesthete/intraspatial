// SpatialScene — the React frontend for the multi-image SpatialData transform editor. It owns the
// WebGPU renderer + one `MultiImageScene`, and drives it from a side panel: open a store, add image
// elements to the scene, click one to select it (a 3D gizmo appears), and edit its placement either
// by dragging the gizmo (R/T/S = rotate/translate/scale) or with the numeric fields here.
//
// Selection is tracked in the scene (raycast on click) and surfaced back via callbacks, so the panel
// always reflects "which object is selected" and shows that object's live transform. State is
// runtime-only for now (no persistence) — see the note in the panel.
import { type ChangeEvent, useCallback, useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { WebGPURenderer } from "three/webgpu";
import { type GizmoMode, type ImageTransform, type LayerBlend, MultiImageScene, type SceneImageInfo } from "./datasource/multiImageScene";
import { openSpatialData, type SpatialDataHandle } from "./datasource/spatialDataLoader";
import type { BlendMode } from "./datasource/tileChannelMaterial";

const DEFAULT_STORE = "http://localhost:8080/xenium_2.q0.001.htj2k.index-permutations.zarr";

type Status = "init" | "running" | "unsupported" | "error";

/** Panel-editable form of a transform: translation, Euler angles in DEGREES, scale. */
interface TransformForm {
  tx: number;
  ty: number;
  tz: number;
  rx: number;
  ry: number;
  rz: number;
  sx: number;
  sy: number;
  sz: number;
}

const IDENTITY_FORM: TransformForm = { tx: 0, ty: 0, tz: 0, rx: 0, ry: 0, rz: 0, sx: 1, sy: 1, sz: 1 };
const DEG = 180 / Math.PI;

function transformToForm(t: ImageTransform): TransformForm {
  const q = new THREE.Quaternion(...t.rotation);
  const e = new THREE.Euler().setFromQuaternion(q, "XYZ");
  return {
    tx: t.translation[0],
    ty: t.translation[1],
    tz: t.translation[2],
    rx: e.x * DEG,
    ry: e.y * DEG,
    rz: e.z * DEG,
    sx: t.scale[0],
    sy: t.scale[1],
    sz: t.scale[2],
  };
}

function formToTransform(f: TransformForm): ImageTransform {
  const e = new THREE.Euler(f.rx / DEG, f.ry / DEG, f.rz / DEG, "XYZ");
  const q = new THREE.Quaternion().setFromEuler(e);
  return {
    translation: [f.tx, f.ty, f.tz],
    rotation: [q.x, q.y, q.z, q.w],
    scale: [f.sx, f.sy, f.sz],
  };
}

const rgbHex = (c: readonly [number, number, number]): string =>
  `#${c
    .map((v) =>
      Math.round(Math.max(0, Math.min(1, v)) * 255)
        .toString(16)
        .padStart(2, "0"),
    )
    .join("")}`;
const hexRgb = (h: string): [number, number, number] => [
  Number.parseInt(h.slice(1, 3), 16) / 255,
  Number.parseInt(h.slice(3, 5), 16) / 255,
  Number.parseInt(h.slice(5, 7), 16) / 255,
];

export default function SpatialScene() {
  const stageRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const sceneRef = useRef<MultiImageScene | null>(null);
  const handlesRef = useRef<Map<string, SpatialDataHandle>>(new Map());

  const [status, setStatus] = useState<Status>("init");
  const [error, setError] = useState<string>("");
  const [storeUrl, setStoreUrl] = useState(DEFAULT_STORE);
  const [openedStore, setOpenedStore] = useState<string>("");
  const [imageNames, setImageNames] = useState<string[]>([]);
  const [pickName, setPickName] = useState<string>("");
  const [busy, setBusy] = useState(false);

  const [images, setImages] = useState<SceneImageInfo[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [form, setForm] = useState<TransformForm>(IDENTITY_FORM);
  const [mode, setMode] = useState<GizmoMode>("translate");
  const [q, setQ] = useState(1);
  const [blend, setBlend] = useState<BlendMode>("additive");
  const [tick, setTick] = useState(0); // force re-read of in-place channel mutations

  // ---- WebGPU renderer + scene lifecycle ----------------------------------------------------
  useEffect(() => {
    const canvas = canvasRef.current;
    const stage = stageRef.current;
    if (!canvas || !stage) return;
    if (typeof navigator === "undefined" || !("gpu" in navigator)) {
      setStatus("unsupported");
      return;
    }
    let scene: MultiImageScene | null = null;
    let renderer: WebGPURenderer | null = null;
    let disposed = false;
    let ro: ResizeObserver | null = null;

    const r = new WebGPURenderer({ canvas, antialias: true, alpha: true });
    r.init()
      .then(() => {
        if (disposed) {
          r.dispose();
          return;
        }
        renderer = r;
        scene = new MultiImageScene(canvas, r);
        scene.setCallbacks({
          onImagesChange: setImages,
          onSelectionChange: (id) => setSelectedId(id),
          onTransformChange: (id, t) => {
            // Only the currently-selected image's panel needs live gizmo values.
            setSelectedId((cur) => {
              if (cur === id) setForm(transformToForm(t));
              return cur;
            });
          },
        });
        sceneRef.current = scene;
        (window as unknown as { __scene: unknown }).__scene = scene; // runtime-inspectable
        const rect = stage.getBoundingClientRect();
        scene.resize(rect.width, rect.height);
        ro = new ResizeObserver(() => {
          const b = stage.getBoundingClientRect();
          scene?.resize(b.width, b.height);
        });
        ro.observe(stage);
        setStatus("running");
      })
      .catch((e: unknown) => {
        setStatus("error");
        setError(e instanceof Error ? e.message : String(e));
      });

    return () => {
      disposed = true;
      ro?.disconnect();
      scene?.dispose();
      sceneRef.current = null;
      renderer?.dispose();
    };
  }, []);

  // When the selection changes (via click), sync the numeric form to that image's transform.
  useEffect(() => {
    const scene = sceneRef.current;
    if (!scene || !selectedId) {
      setForm(IDENTITY_FORM);
      return;
    }
    const t = scene.getTransform(selectedId);
    if (t) setForm(transformToForm(t));
  }, [selectedId]);

  // ---- Store / image loading ----------------------------------------------------------------
  const openStore = useCallback(async (url: string) => {
    const trimmed = url.trim() || DEFAULT_STORE;
    setBusy(true);
    setError("");
    try {
      let handle = handlesRef.current.get(trimmed);
      if (!handle) {
        handle = await openSpatialData(trimmed);
        handlesRef.current.set(trimmed, handle);
      }
      setOpenedStore(trimmed);
      setImageNames(handle.imageNames);
      setPickName(handle.imageNames[0] ?? "");
      if (handle.imageNames.length === 0) setError("No image elements found — is this a SpatialData store?");
    } catch (e) {
      setError(`Open failed: ${e instanceof Error ? e.message : String(e)}`);
      setImageNames([]);
    } finally {
      setBusy(false);
    }
  }, []);

  // Auto-open the default store once the scene is running.
  useEffect(() => {
    if (status === "running" && !openedStore) void openStore(DEFAULT_STORE);
  }, [status, openedStore, openStore]);

  const addImage = useCallback(async () => {
    const scene = sceneRef.current;
    const handle = handlesRef.current.get(openedStore);
    if (!scene || !handle || !pickName) return;
    setBusy(true);
    setError("");
    try {
      const src = await handle.image(pickName);
      scene.addImage(src, { store: openedStore, element: pickName });
    } catch (e) {
      setError(`Load failed: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setBusy(false);
    }
  }, [openedStore, pickName]);

  // ---- Transform editing --------------------------------------------------------------------
  const applyForm = useCallback(
    (next: TransformForm) => {
      setForm(next);
      if (selectedId) sceneRef.current?.setTransform(selectedId, formToTransform(next));
    },
    [selectedId],
  );
  const setField = (k: keyof TransformForm) => (e: ChangeEvent<HTMLInputElement>) => {
    const v = Number(e.target.value);
    if (Number.isFinite(v)) applyForm({ ...form, [k]: v });
  };

  const changeMode = (m: GizmoMode) => {
    setMode(m);
    sceneRef.current?.setGizmoMode(m);
  };
  const resetSelected = () => {
    if (!selectedId) return;
    sceneRef.current?.resetTransform(selectedId);
    setForm(IDENTITY_FORM);
  };

  const selected = images.find((i) => i.id === selectedId) ?? null;

  if (status === "unsupported")
    return (
      <div className="spatial-scene not-content">
        <style>{CSS}</style>
        <div className="overlay">WebGPU is not available in this browser. Open in Chrome or Edge.</div>
      </div>
    );

  return (
    <div className="spatial-scene not-content">
      <style>{CSS}</style>
      <div className="stage" ref={stageRef}>
        <canvas ref={canvasRef} />
        {status !== "running" && <div className="overlay">{status === "error" ? `Renderer error: ${error}` : "starting WebGPU…"}</div>}
      </div>

      <div className="panel">
        <h1>Multi-image scene</h1>
        <p className="sub">
          Add SpatialData images, then click one to select it — a 3D gizmo appears. Drag to move it, or press <b>R</b>/<b>T</b>/<b>S</b> for
          rotate/translate/scale. Edit the numbers below for precision. Orbit with the empty background.
        </p>
        <p className="note">
          ⚠ Runtime-only — nothing is persisted yet. Expects a local CORS SpatialData / OME-Zarr store (default <code>localhost:8080</code>
          ).
        </p>

        <div className="field">
          <label htmlFor="store">store url</label>
          <input id="store" type="text" value={storeUrl} onChange={(e) => setStoreUrl(e.target.value)} />
          <button type="button" disabled={busy} onClick={() => void openStore(storeUrl)}>
            Open store
          </button>
        </div>

        <div className="field row">
          <select value={pickName} onChange={(e) => setPickName(e.target.value)} disabled={!imageNames.length}>
            {imageNames.length === 0 && <option value="">— no images —</option>}
            {imageNames.map((n) => (
              <option key={n} value={n}>
                {n}
              </option>
            ))}
          </select>
          <button type="button" disabled={busy || !pickName} onClick={() => void addImage()}>
            Add to scene
          </button>
        </div>

        <div className="section">scene ({images.length})</div>
        <div className="imglist">
          {images.length === 0 && <div className="muted">No images yet — add one above.</div>}
          {images.map((img) => (
            <div key={img.id} className={`imgrow${img.id === selectedId ? " sel" : ""}`}>
              <input
                type="checkbox"
                checked={img.visible}
                title="visible"
                onChange={(e) => sceneRef.current?.setVisible(img.id, e.target.checked)}
              />
              <button type="button" className="imgname" title={img.label} onClick={() => sceneRef.current?.selectImage(img.id)}>
                {img.name}
              </button>
              <button type="button" className="rm" title="remove" onClick={() => sceneRef.current?.removeImage(img.id)}>
                ×
              </button>
            </div>
          ))}
        </div>

        {selected && (
          <>
            <div className="section">transform · {selected.name}</div>
            <div className="modes">
              {(["translate", "rotate", "scale"] as GizmoMode[]).map((m) => (
                <button key={m} type="button" className={mode === m ? "on" : ""} onClick={() => changeMode(m)}>
                  {m[0]?.toUpperCase()}
                  {m.slice(1)}
                </button>
              ))}
              <button type="button" className="reset" onClick={resetSelected}>
                Reset
              </button>
            </div>
            <TripletRow
              label="pos"
              a={form.tx}
              b={form.ty}
              c={form.tz}
              step={1}
              onA={setField("tx")}
              onB={setField("ty")}
              onC={setField("tz")}
            />
            <TripletRow
              label="rot°"
              a={form.rx}
              b={form.ry}
              c={form.rz}
              step={1}
              onA={setField("rx")}
              onB={setField("ry")}
              onC={setField("rz")}
            />
            <TripletRow
              label="scale"
              a={form.sx}
              b={form.sy}
              c={form.sz}
              step={0.05}
              onA={setField("sx")}
              onB={setField("sy")}
              onC={setField("sz")}
            />

            <div className="section">layer</div>
            <div className="field row">
              <label htmlFor="opacity">opacity {selected.opacity.toFixed(2)}</label>
              <input
                id="opacity"
                type="range"
                min={0}
                max={1}
                step={0.02}
                value={selected.opacity}
                onChange={(e) => sceneRef.current?.setOpacity(selected.id, Number(e.target.value))}
              />
            </div>
            <div className="field row">
              <label htmlFor="lblend">blend</label>
              <select
                id="lblend"
                value={selected.blend}
                onChange={(e) => sceneRef.current?.setBlend(selected.id, e.target.value as LayerBlend)}
              >
                <option value="normal">normal (over)</option>
                <option value="additive">additive</option>
              </select>
            </div>

            <div className="section">channels</div>
            <div className="chpanel" data-tick={tick}>
              {selected.channels.map((ch) => (
                <div className="chrow" key={`${selected.id}:${ch.label}`}>
                  <input
                    type="checkbox"
                    checked={ch.visible}
                    title="visible"
                    onChange={(e) => {
                      ch.visible = e.target.checked;
                      sceneRef.current?.updateChannels(selected.id);
                      setTick((t) => t + 1);
                    }}
                  />
                  <input
                    type="color"
                    value={rgbHex(ch.color)}
                    title="colour"
                    onChange={(e) => {
                      ch.color = hexRgb(e.target.value);
                      sceneRef.current?.updateChannels(selected.id);
                      setTick((t) => t + 1);
                    }}
                  />
                  <span className="chname" title={ch.label}>
                    {ch.label}
                  </span>
                  <input
                    type="range"
                    min={0}
                    max={1}
                    step={0.005}
                    value={ch.contrastLimits[0]}
                    className="chslider"
                    onChange={(e) => {
                      ch.contrastLimits = [Math.min(Number(e.target.value), ch.contrastLimits[1]), ch.contrastLimits[1]];
                      sceneRef.current?.updateChannels(selected.id);
                      setTick((t) => t + 1);
                    }}
                  />
                  <input
                    type="range"
                    min={0}
                    max={1}
                    step={0.005}
                    value={ch.contrastLimits[1]}
                    className="chslider"
                    onChange={(e) => {
                      ch.contrastLimits = [ch.contrastLimits[0], Math.max(Number(e.target.value), ch.contrastLimits[0])];
                      sceneRef.current?.updateChannels(selected.id);
                      setTick((t) => t + 1);
                    }}
                  />
                </div>
              ))}
            </div>
          </>
        )}

        <div className="section">render</div>
        <div className="field row">
          <label htmlFor="q">detail q {q.toFixed(2)}</label>
          <input
            id="q"
            type="range"
            min={0.25}
            max={4}
            step={0.05}
            value={q}
            onChange={(e) => {
              const v = Number(e.target.value);
              setQ(v);
              sceneRef.current?.setQ(v);
            }}
          />
        </div>
        <div className="field row">
          <label htmlFor="blend">channels</label>
          <select
            id="blend"
            value={blend}
            onChange={(e) => {
              const b = e.target.value as BlendMode;
              setBlend(b);
              sceneRef.current?.setChannelBlend(b);
            }}
          >
            <option value="additive">additive</option>
            <option value="max">max</option>
          </select>
        </div>

        {error && <div className="err">{error}</div>}
      </div>
    </div>
  );
}

function TripletRow(props: {
  label: string;
  a: number;
  b: number;
  c: number;
  step: number;
  onA: (e: ChangeEvent<HTMLInputElement>) => void;
  onB: (e: ChangeEvent<HTMLInputElement>) => void;
  onC: (e: ChangeEvent<HTMLInputElement>) => void;
}) {
  const fmt = (n: number) => (Math.round(n * 1000) / 1000).toString();
  return (
    <div className="triplet">
      <span className="tlabel">{props.label}</span>
      <input type="number" step={props.step} value={fmt(props.a)} onChange={props.onA} />
      <input type="number" step={props.step} value={fmt(props.b)} onChange={props.onB} />
      <input type="number" step={props.step} value={fmt(props.c)} onChange={props.onC} />
    </div>
  );
}

const CSS = `
.spatial-scene { position: relative; width: 100%; height: min(80vh, 720px); border: 1px solid #1e293b; border-radius: 10px; overflow: hidden; background: #0b1020; color: #e2e8f0; font: 13px/1.5 ui-sans-serif, system-ui, sans-serif; }
.spatial-scene * { box-sizing: border-box; }
.spatial-scene .stage { position: absolute; inset: 0; }
.spatial-scene canvas { width: 100%; height: 100%; display: block; }
.spatial-scene .overlay { position: absolute; inset: 0; display: flex; align-items: center; justify-content: center; text-align: center; padding: 24px; color: #94a3b8; }
.spatial-scene .panel { position: absolute; top: 12px; left: 12px; width: 288px; max-height: calc(100% - 24px); overflow-y: auto; padding: 14px 16px; background: rgba(15,23,42,0.86); border: 1px solid #1e293b; border-radius: 10px; backdrop-filter: blur(6px); }
.spatial-scene h1 { margin: 0 0 2px; font-size: 14px; font-weight: 600; }
.spatial-scene .sub { margin: 0 0 8px; color: #94a3b8; font-size: 11px; }
.spatial-scene .note { color: #fcd34d; font-size: 10px; line-height: 1.4; margin: 0 0 8px; }
.spatial-scene .note code { background: #0f172a; border: 1px solid #334155; border-radius: 3px; padding: 0 3px; }
.spatial-scene .field { display: flex; flex-direction: column; gap: 4px; margin: 8px 0; }
.spatial-scene .field.row { flex-direction: row; align-items: center; justify-content: space-between; gap: 8px; }
.spatial-scene label { color: #cbd5e1; }
.spatial-scene input[type=text] { width: 100%; background: #0f172a; color: #cbd5e1; border: 1px solid #334155; border-radius: 6px; padding: 3px 6px; font-size: 11px; }
.spatial-scene select { flex: 1; background: #0f172a; color: #e2e8f0; border: 1px solid #334155; border-radius: 6px; padding: 3px 6px; font-size: 11px; }
.spatial-scene button { background: #1e293b; color: #e2e8f0; border: 1px solid #334155; border-radius: 6px; padding: 3px 8px; font-size: 11px; cursor: pointer; }
.spatial-scene button:hover:not(:disabled) { background: #334155; }
.spatial-scene button:disabled { opacity: 0.5; cursor: default; }
.spatial-scene .section { margin: 12px 0 6px; padding-top: 8px; border-top: 1px solid #1e293b; color: #38bdf8; font-size: 10px; text-transform: uppercase; letter-spacing: 0.06em; }
.spatial-scene .imglist { display: flex; flex-direction: column; gap: 3px; }
.spatial-scene .imgrow { display: flex; align-items: center; gap: 6px; padding: 2px 4px; border-radius: 5px; }
.spatial-scene .imgrow.sel { background: rgba(56,189,248,0.15); outline: 1px solid #38bdf8; }
.spatial-scene .imgname { flex: 1; text-align: left; background: none; border: none; padding: 2px 2px; color: #e2e8f0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.spatial-scene .rm { padding: 0 6px; line-height: 1.4; }
.spatial-scene .muted { color: #94a3b8; font-size: 11px; }
.spatial-scene .modes { display: flex; gap: 4px; margin: 6px 0; }
.spatial-scene .modes button.on { background: #38bdf8; color: #0b1020; border-color: #38bdf8; }
.spatial-scene .modes .reset { margin-left: auto; }
.spatial-scene .triplet { display: grid; grid-template-columns: 34px 1fr 1fr 1fr; gap: 4px; align-items: center; margin: 4px 0; }
.spatial-scene .tlabel { color: #94a3b8; font-size: 10px; }
.spatial-scene .triplet input { width: 100%; background: #0f172a; color: #cbd5e1; border: 1px solid #334155; border-radius: 5px; padding: 2px 4px; font-size: 11px; font-variant-numeric: tabular-nums; }
.spatial-scene .chrow { display: flex; align-items: center; gap: 6px; margin: 4px 0; }
.spatial-scene .chname { flex: 0 0 46px; font-size: 10px; color: #cbd5e1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.spatial-scene .chslider { width: 46px; }
.spatial-scene input[type=color] { width: 22px; height: 18px; padding: 0; border: 1px solid #334155; border-radius: 3px; background: none; }
.spatial-scene .err { color: #fca5a5; margin-top: 8px; white-space: pre-wrap; font-size: 11px; }
`;
