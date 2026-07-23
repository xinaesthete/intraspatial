// Slice-0 spike — does react-three-fiber carry the WebGPU backend well enough to host the
// serial-section scene editor's multi-viewport shell?
//
// LIFETIME: this page is a spike, kept only until something real is built on R3F (the viewer-layer
// promotion out of `playground/` — see `docs/packaging-and-consumers.md`). Delete it then; until
// then it is the only place R3F is exercised, so it doubles as a smoke-test of the WebGPU path.
// The findings below are its durable output and also live in
// `docs/serial-section-alignment-and-multi-viewport.md` §7.
//
// VERDICT: yes, with one shim. Measured in-browser on three r185 / R3F 9.6.1 / drei 10.7.7.
// The page checks the three revision at runtime and says so if it has moved — a spike that is not
// re-run acquires false authority, and the `WebGPURenderer` scissor Y-flip in particular is the kind
// of thing that may be fixed upstream, silently invalidating §3 below.
//
//   1. ASYNC RENDERER — PASS. R3F v9's `gl` prop accepts `(defaultProps) => Promise<Renderer>`;
//      `await renderer.init()` inside it works and reports `backend: WebGPU` (not a WebGL fallback).
//      Spread R3F's `defaultProps` into the constructor — they carry `alpha: true`, without which a
//      full-page fixed canvas clears opaque and hides the DOM under it.
//   2. TSL THROUGH THE RECONCILER — PASS TO MOUNT, FAILS TO SWAP. See `DeclarativeContent`: node
//      materials mount fine, but replacing `colorNode` on a mounted material does nothing. Drive
//      live changes through uniforms (which is what `ChannelComposite` already does).
//   3. TWO PANES, TWO CAMERAS, ONE CANVAS — PASS, AFTER THE Y-FLIP. drei `<View>`'s scissored
//      mid-frame `gl.clear(true, true)` does NOT wipe the frame on this backend (the main risk, given
//      that a mid-frame `clearDepth()` is known to). But viewport/scissor Y is inverted — see
//      `WebGpuScissorFlip`, which is the entire fix.
//   4a. IMPERATIVE SUBTREES — PASS. A hand-built `THREE.Group` on `<primitive>` renders and survives
//      Fast-Refresh (`rendererBuilds` stayed 1 across two edits; the canvas never blanked). This is
//      the wart that forces a hard reload in the imperative `SpatialScene.tsx`, and R3F removes it.
//   4b. SHARING ONE `Object3D` ACROSS VIEWS — NOT SUPPORTED, despite appearances. Mounting the same
//      group in both views makes it render in BOTH (proven: setting `group.visible = false` removed
//      it from both panes at once) — but a scene walk shows it parented at the view-scene ROOT,
//      outside either view's group. That is double-`<primitive>` attachment landing somewhere
//      unspecified, not a sharing feature, and it defeats per-viewport visibility control (which
//      onion-skin needs). Treat N viewports as N lightweight meshes per tile sharing one
//      geometry/material/texture — GPU memory lives in the textures, so it stays shared.
//
// HARNESS NOTE: a page whose tab is not visible runs no rendering steps, so `requestAnimationFrame`
// and `ResizeObserver` never fire — and R3F sizes itself entirely through `ResizeObserver`, so
// `<Canvas>` never configures and nothing renders. This looks exactly like a broken spike. Force a
// paint (take a screenshot) and it proceeds.
//
// The HMR criterion is measured, not eyeballed: `rendererBuilds` counts `WebGPURenderer`
// constructions and `frames` keeps ticking only while the canvas is alive. Edit the MARKER string
// below, save, and read the two numbers.

import { OrbitControls, PerspectiveCamera, View } from "@react-three/drei";
import { Canvas, extend, type ThreeElement, useFrame, useThree } from "@react-three/fiber";
import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { Fn, positionLocal, uniform, vec3, vec4 } from "three/tsl";
import { MeshBasicNodeMaterial, WebGPURenderer } from "three/webgpu";

// Node materials are not in the core THREE namespace R3F auto-extends, so register the one we use.
extend({ MeshBasicNodeMaterial });
declare module "@react-three/fiber" {
  interface ThreeElements {
    meshBasicNodeMaterial: ThreeElement<typeof MeshBasicNodeMaterial>;
  }
}

/** Edit this string and save to exercise Fast-Refresh — see criteria 2 and 4a. */
const MARKER = "edit me, save, and watch the counters (renderer builds must stay 1)";

/** The three revision the findings in the header were actually measured against. If the installed
 *  revision has moved past this, the page says so — see the LIFETIME note above. */
const VERIFIED_AGAINST_THREE_REVISION = "185";

// Counters live on globalThis so a Fast-Refresh module re-evaluation does NOT reset them; that is
// the whole measurement. A module-level `let` would zero itself and always report "one build".
interface SpikeCounters {
  rendererBuilds: number;
  groupBuilds: number;
  group?: THREE.Group;
  /** Set by `Probe`'s mount effect — see the diagnosis note there. */
  mounted?: boolean;
  gl?: unknown;
  scene?: unknown;
}
const spikeGlobal = globalThis as { __r3fSpike?: SpikeCounters };
spikeGlobal.__r3fSpike ??= { rendererBuilds: 0, groupBuilds: 0 };
const counters: SpikeCounters = spikeGlobal.__r3fSpike;

// ── The imperative subtree (criterion 4) ──────────────────────────────────────────────────────

/** A `THREE.Group` built the way `TileRenderer` builds one: outside React, mutated by hand, handed
 *  to the tree as an opaque object. Cached on `globalThis` so we can report whether a Fast-Refresh
 *  reused it or rebuilt it. */
function imperativeGroup(): THREE.Group {
  if (counters.group) return counters.group;
  counters.groupBuilds += 1;

  const group = new THREE.Group();
  // A TSL material built imperatively — the same shape `ChannelComposite` mints per tile.
  const tint = uniform(vec3(0.35, 0.8, 1));
  const mat = new MeshBasicNodeMaterial();
  const shade = Fn(() => vec4(tint.mul(positionLocal.y.add(1).mul(0.5).add(0.25)), 1));
  // biome-ignore lint/suspicious/noExplicitAny: TSL node output → colorNode (ADR-0009 §1 friction)
  mat.colorNode = shade() as any;

  const geo = new THREE.TorusKnotGeometry(0.6, 0.2, 96, 16);
  for (let i = 0; i < 3; i++) {
    const mesh = new THREE.Mesh(geo, mat); // geometry + material shared across meshes, as tiles do
    mesh.position.set((i - 1) * 2.2, 0, 0);
    group.add(mesh);
  }
  counters.group = group;
  return group;
}

// ── Scene content ─────────────────────────────────────────────────────────────────────────────

/** Declarative content: a node material through the reconciler (criterion 2). Duplicated per view,
 *  which is what N viewports would have to do for declarative objects.
 *
 *  FINDING: R3F *mounts* a TSL node material correctly, but **replacing `colorNode` on a mounted
 *  material is a no-op** — the compiled node graph is not rebuilt, so the surface keeps its original
 *  shading. (Measured: a Fast-Refresh that changed `hue` updated the DOM text in the same commit
 *  while the box's colour did not move.) The working pattern is the one this component now uses and
 *  the one `ChannelComposite` already uses: build the node graph ONCE and drive every live change
 *  through **uniforms**, which mutate in place and need no reconciliation. That is why the stain
 *  matrix and the contrast window must be uniforms rather than a rebuilt graph. */
function DeclarativeContent({ hue }: { hue: number }): React.JSX.Element {
  // Built once. `hue` moves the uniform, never the graph.
  const { colorNode, tint } = useMemo(() => {
    const t = uniform(vec3(1, 1, 1));
    const shade = Fn(() => vec4(t.mul(positionLocal.y.add(1.2).mul(0.5)), 1));
    // biome-ignore lint/suspicious/noExplicitAny: TSL node output → colorNode (ADR-0009 §1 friction)
    return { colorNode: shade() as any, tint: t };
  }, []);

  useEffect(() => {
    const c = new THREE.Color().setHSL(hue, 0.7, 0.55);
    tint.value.set(c.r, c.g, c.b);
  }, [hue, tint]);

  const ref = useRef<THREE.Mesh>(null);
  useFrame((_, dt) => {
    if (ref.current) ref.current.rotation.y += dt * 0.6;
  });

  return (
    <mesh ref={ref} position={[0, 1.6, 0]}>
      <boxGeometry args={[1.4, 1.4, 1.4]} />
      <meshBasicNodeMaterial colorNode={colorNode} />
    </mesh>
  );
}

/** FINDING (three r185): `WebGPURenderer.setViewport`/`setScissor` take a **top-left** origin, but
 *  `WebGLRenderer` — and therefore drei's `<View>`, which is written against it — passes
 *  **bottom-left**. Measured: for a pane at CSS rect `left 12, top 90, 683×799` on a 1400×900 canvas,
 *  `<View>` correctly computes `y = 900 − (90 + 799) = 11`, and the backend then draws that rect 11px
 *  from the TOP. Every view lands too high by `canvasH − rect.h − 2·rect.top`.
 *
 *  `y' = height − y − h` is the whole fix. Patching the renderer (rather than forking `<View>`) keeps
 *  it in one place and makes it trivial to delete if three changes the convention. Sizes are read per
 *  call in CSS pixels, so it stays correct across resize and pixel-ratio changes. */
function WebGpuScissorFlip(): null {
  const gl = useThree((s) => s.gl);

  useEffect(() => {
    const r = gl as unknown as {
      isWebGPURenderer?: boolean;
      __flipPatched?: boolean;
      getSize: (v: THREE.Vector2) => THREE.Vector2;
      setScissor: (...a: unknown[]) => void;
      setViewport: (...a: unknown[]) => void;
    };
    if (!r.isWebGPURenderer || r.__flipPatched) return;
    r.__flipPatched = true;

    const size = new THREE.Vector2();
    const setScissor0 = r.setScissor.bind(r);
    const setViewport0 = r.setViewport.bind(r);
    const flipY = (y: number, h: number): number => r.getSize(size).y - y - h;
    // The (x: Vector4|Box2) overloads are passed straight through — only the 4-number form is flipped.
    r.setScissor = (x, y, w, h) => (typeof x === "number" ? setScissor0(x, flipY(y as number, h as number), w, h) : setScissor0(x));
    r.setViewport = (x, y, w, h) => (typeof x === "number" ? setViewport0(x, flipY(y as number, h as number), w, h) : setViewport0(x));
  }, [gl]);

  return null;
}

/** Reports renderer identity + a live frame count, so "is the canvas still alive" is a number on
 *  the page rather than a judgement about whether the image looks right. */
function Probe({ onReport }: { onReport: (r: { backend: string; frames: number }) => void }): null {
  const gl = useThree((s) => s.gl);
  const scene = useThree((s) => s.scene);
  const frames = useRef(0);
  const lastPublished = useRef(0);

  // Mount-time breadcrumb: distinguishes "the canvas subtree never mounted" (suspended / threw)
  // from "it mounted but the frameloop never ran".
  useEffect(() => {
    counters.mounted = true;
    counters.gl = gl;
    counters.scene = scene;
  }, [gl, scene]);

  useFrame(() => {
    frames.current += 1;
    if (frames.current - lastPublished.current < 20) return;
    lastPublished.current = frames.current;
    const r = gl as unknown as { isWebGPURenderer?: boolean; backend?: { isWebGPUBackend?: boolean } };
    const backend = r.isWebGPURenderer ? (r.backend?.isWebGPUBackend ? "WebGPU" : "WebGL fallback") : "WebGLRenderer";
    onReport({ backend, frames: frames.current });
  });

  return null;
}

// ── Page ──────────────────────────────────────────────────────────────────────────────────────

const paneStyle: React.CSSProperties = {
  position: "relative",
  border: "1px solid #26314e",
  borderRadius: 6,
  overflow: "hidden",
  background: "#0e1526",
};

export default function R3fSpike(): React.JSX.Element {
  const container = useRef<HTMLDivElement>(null);
  const [report, setReport] = useState<{ backend: string; frames: number } | null>(null);
  // 4b: does the SAME imperative group render in both views, or only the last one to claim it?
  const [shareGroup, setShareGroup] = useState(false);
  // drei's `<View>` computes its scissor rect in CSS pixels. If `WebGPURenderer` applies the pixel
  // ratio to `setViewport`/`setScissor` differently from `WebGLRenderer`, the rects land at the
  // wrong scale — so make the ratio switchable and compare rather than reason about it.
  const [dpr, setDpr] = useState(1);
  const group = useMemo(() => imperativeGroup(), []);
  // `counters` is read during render; `Probe`'s periodic `setReport` re-renders often enough that
  // the async `gl` callback's increment shows up without its own subscription.

  return (
    <div ref={container} style={{ height: "100%", display: "flex", flexDirection: "column", gap: 10, padding: 12 }}>
      <header style={{ flex: "0 0 auto" }}>
        <h2 style={{ margin: "0 0 2px", fontSize: 15 }}>Spike — react-three-fiber on the WebGPU backend</h2>
        <p style={{ margin: 0, color: "#94a3b8", fontSize: 12 }}>{MARKER}</p>
        {THREE.REVISION === VERIFIED_AGAINST_THREE_REVISION ? (
          <p style={{ margin: "2px 0 0", color: "#64748b", fontSize: 11 }}>
            three r{THREE.REVISION} — findings in the source header were measured against this revision.
          </p>
        ) : (
          <p style={{ margin: "2px 0 0", color: "#fbbf24", fontSize: 11 }}>
            ⚠ three is now <b>r{THREE.REVISION}</b>, but the findings were measured against r{VERIFIED_AGAINST_THREE_REVISION}. Re-run this
            page before trusting them — the
            <code> WebGPURenderer</code> scissor Y-flip especially may have been fixed upstream.
          </p>
        )}
        <div style={{ marginTop: 6, display: "flex", gap: 14, flexWrap: "wrap", fontSize: 12, color: "#cbd5e1" }}>
          <span>
            backend: <b style={{ color: report?.backend === "WebGPU" ? "#4ade80" : "#f87171" }}>{report?.backend ?? "…"}</b>
          </span>
          <span>
            frames: <b>{report?.frames ?? 0}</b>
          </span>
          <span>
            renderer builds: <b>{counters.rendererBuilds}</b>
          </span>
          <span>
            imperative group builds: <b>{counters.groupBuilds}</b>
          </span>
          <label style={{ display: "flex", gap: 5, alignItems: "center" }}>
            <input type="checkbox" checked={shareGroup} onChange={(e) => setShareGroup(e.target.checked)} />
            share the imperative group with view B (4b)
          </label>
          <label style={{ display: "flex", gap: 5, alignItems: "center" }}>
            dpr
            <select value={dpr} onChange={(e) => setDpr(Number(e.target.value))}>
              <option value={1}>1</option>
              <option value={2}>2</option>
            </select>
          </label>
        </div>
      </header>

      <div style={{ flex: "1 1 auto", display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, minHeight: 0 }}>
        <View style={paneStyle}>
          <PerspectiveCamera makeDefault position={[4, 3.5, 7]} fov={45} />
          <OrbitControls makeDefault />
          <color attach="background" args={["#111a2e"]} />
          <primitive object={group} />
          <DeclarativeContent hue={0.55} />
        </View>

        <View style={paneStyle}>
          {/* Nadir: a genuinely different camera on the same pass — criterion 3. `PerspectiveCamera`
              looks down -Z unless told otherwise, so the pitch is explicit. */}
          <PerspectiveCamera makeDefault position={[0, 11, 0]} rotation={[-Math.PI / 2, 0, 0]} fov={45} />
          <color attach="background" args={["#151024"]} />
          {shareGroup ? <primitive object={group} /> : null}
          <DeclarativeContent hue={0.35} />
        </View>
      </div>

      <Canvas
        eventSource={container as React.RefObject<HTMLElement>}
        style={{ position: "fixed", inset: 0, pointerEvents: "none" }}
        dpr={dpr}
        gl={async (props) => {
          counters.rendererBuilds += 1;
          // Spread R3F's defaults rather than hand-picking: they carry `alpha: true`, without which
          // the canvas clears opaque and — being fixed over the whole page — hides the DOM beneath.
          const renderer = new WebGPURenderer({ ...props, canvas: props.canvas as HTMLCanvasElement });
          await renderer.init();
          return renderer;
        }}
      >
        <WebGpuScissorFlip />
        <Probe onReport={setReport} />
        <View.Port />
      </Canvas>
    </div>
  );
}
