// Spike — can three.js sample a GPUTexture WE allocated on ITS device?
//
// This is step 4 of docs/gpu-resident-loader.md, and the one unverified link in the device-resident
// Loader chain. Everything upstream of it is now tested headlessly (src/gpu/tiles/assemble.gpu.test.ts):
// integer planes upload, assemble, half-pack, and land in a texture with the right values. What that
// test cannot show is whether THREE's renderer will *sample* such a texture, because that needs three's
// own device and its own material pipeline.
//
// So: adopt `renderer.backend.device`, build the texture with the real assembly pass, wrap it in
// `THREE.ExternalTexture`, and render it — then read the pixels back and compare against the pattern
// we put in. Self-checking rather than eyeballed; the canvas at the bottom is only corroboration.
//
// Two cases, because the two render backends want different things:
//   A. 2-D rgba16float — what TileRenderer's channel-composite material samples.
//   B. 3-D r16float    — what the volume raymarch samples (texture3D), and the case that matters
//                        for the 3DxN store, whose chunk is a 32-slice slab.
//
// RESULT (three 0.185, Chrome/Dawn, 2026-08-10): both pass, to within fp16 + the rgba8 readback.
// The negative controls below say which adaptations are actually load-bearing:
//
//   ?noimage=1   → still 3/3. Supplying `texture.image = {width, height, depth}` is NOT required;
//                  three's ExternalTexture path returns before it needs the extent.
//   ?no3dflag=1  → case B FAILS. `isData3DTexture = true` IS required: three picks the bind-group
//                  view dimension from that flag alone (WebGPUTextureUtils._getDimension), so a 3-D
//                  source is otherwise bound as texture_2d and the generated WGSL fails to compile
//                  ("expected 'vec2<f32>', got 'vec3<f32>'"). The quad then renders BLACK — no
//                  exception, though three does log the pipeline error, so it is diagnosable if you
//                  are looking at the console.

import * as THREE from "three";
import { texture, texture3D, uv, vec3, vec4 } from "three/tsl";
import { MeshBasicNodeMaterial, WebGPURenderer } from "three/webgpu";

import { adoptDevice } from "../../src/gpu/interop";
import { assembleTile, copyAssembledToTexture, uploadPlane } from "../../src/gpu/tiles/assemble";

/** What a node material will accept as its colour. TSL nodes are assignable to this directly; it is
 *  wrapping them in `Fn(...)()` that produces an opaque FnNode the typings reject. */
type ColorNode = NonNullable<MeshBasicNodeMaterial["colorNode"]>;

// Negative controls. An ExternalTexture needs two things three does not give it — an `image` with
// the extent, and (for a 3-D source) the `isData3DTexture` flag three's view-dimension check keys on.
// A spike that only shows the happy path does not tell the next person which of those are load-bearing,
// so both are switchable: `?noimage=1` and `?no3dflag=1` re-run the same cases without them.
const params = new URLSearchParams(location.search);
const OMIT_IMAGE = params.get("noimage") === "1";
const OMIT_3D_FLAG = params.get("no3dflag") === "1";

const canvas = document.getElementById("canvas") as HTMLCanvasElement;
const verdictEl = document.getElementById("verdict") as HTMLDivElement;
const casesEl = document.getElementById("cases") as HTMLDivElement;

interface CaseResult {
  name: string;
  ok: boolean;
  detail: string;
}
const results: CaseResult[] = [];

function report(name: string, ok: boolean, detail: string): void {
  results.push({ name, ok, detail });
  const el = document.createElement("div");
  el.className = "case";
  el.innerHTML = `<h2>${name} <span class="tag ${ok ? "pass" : "fail"}">${ok ? "PASS" : "FAIL"}</span></h2><pre></pre>`;
  (el.querySelector("pre") as HTMLPreElement).textContent = detail;
  casesEl.append(el);
  console.log(`[spike] ${ok ? "PASS" : "FAIL"} — ${name}\n${detail}`);
}

/** The pattern under test, as a function of texel coordinate.
 *
 *  Every lane must vary along every axis it is meant to prove, or the comparison passes on a
 *  degenerate read. The first version of this spike had the single-lane (3-D) case carry `fx`, which
 *  does not depend on z at all — so "sampling z slice 8" would have passed identically had three
 *  bound the 3-D texture as a 2-D one and ignored the z coordinate entirely. Lane 0 now mixes all
 *  three axes with distinguishable weights, so a dropped or transposed axis moves the value. */
const pattern = (x: number, y: number, z: number, lane: number, w: number, h: number, d: number): number => {
  const fx = x / Math.max(1, w - 1);
  const fy = y / Math.max(1, h - 1);
  const fz = d > 1 ? z / Math.max(1, d - 1) : 0;
  if (d > 1) return [0.15 * fx + 0.25 * fy + 0.6 * fz, fy, fx, 1][lane] ?? 0;
  return [fx, fy, 0.25, 1][lane] ?? 0;
};

/** Build one uint16 plane of `pattern`, ready for `uploadPlane`. */
function planeFor(lane: number, w: number, h: number, d: number): Uint16Array {
  const out = new Uint16Array(w * h * d);
  for (let z = 0; z < d; z++) {
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) out[(z * h + y) * w + x] = Math.round(pattern(x, y, z, lane, w, h, d) * 65535);
    }
  }
  return out;
}

/** Fill a GPUTexture through the real assembly pass — integer planes in, texels out, no host loop. */
async function fillTexture(device: GPUDevice, texture: GPUTexture, w: number, h: number, d: number, lanes: number): Promise<void> {
  const planes = Array.from({ length: lanes }, (_, lane) => uploadPlane(device, planeFor(lane, w, h, d)));
  const assembled = await assembleTile({ device, planes, width: w, height: h, depth: d, bits: 16, scale: 1 / 65535, out: "f16" });
  copyAssembledToTexture(device, assembled, texture, { width: w, height: h, depthOrArrayLayers: d });
}

/** Render one quad with `colorNode` into an offscreen target and read the pixels back. The target is
 *  rgba8, so comparisons carry a 1/255 quantisation on top of fp16 — the tolerance below allows for
 *  both, and is still far tighter than any real failure (a wrong axis or a dropped channel). */
async function renderAndRead(renderer: WebGPURenderer, colorNode: ColorNode, size: number): Promise<Uint8Array> {
  const target = new THREE.RenderTarget(size, size, { depthBuffer: false });
  target.texture.colorSpace = THREE.NoColorSpace;
  const scene = new THREE.Scene();
  const camera = new THREE.OrthographicCamera(-0.5, 0.5, 0.5, -0.5, 0, 1);
  const material = new MeshBasicNodeMaterial();
  material.colorNode = colorNode;
  scene.add(new THREE.Mesh(new THREE.PlaneGeometry(1, 1), material));
  renderer.setRenderTarget(target);
  renderer.render(scene, camera);
  const px = await renderer.readRenderTargetPixelsAsync(target, 0, 0, size, size);
  renderer.setRenderTarget(null);
  return new Uint8Array(px.buffer ?? px);
}

/** Compare a read-back rgba8 buffer against `pattern`, sampling texel centres. `flipY` because a
 *  render-target read is bottom-up while our texel grid is top-down. */
function compare(
  px: Uint8Array,
  size: number,
  w: number,
  h: number,
  d: number,
  z: number,
  lanes: number,
): { maxErr: number; worst: string } {
  let maxErr = 0;
  let worst = "";
  const probes = [0.5, 1.5, 2.5, 3.5].flatMap((a) => [0.5, 1.5, 2.5, 3.5].map((b) => [a, b] as const));
  for (const [px0, py0] of probes) {
    const sx = Math.min(size - 1, Math.floor((px0 / 4) * size));
    const sy = Math.min(size - 1, Math.floor((py0 / 4) * size));
    const u = (sx + 0.5) / size;
    const v = (sy + 0.5) / size;
    const tx = Math.min(w - 1, Math.floor(u * w));
    const ty = Math.min(h - 1, Math.floor((1 - v) * h)); // flipY
    const base = (sy * size + sx) * 4;
    for (let lane = 0; lane < lanes; lane++) {
      const want = pattern(tx, ty, z, lane, w, h, d);
      const got = (px[base + lane] ?? 0) / 255;
      const err = Math.abs(got - want);
      if (err > maxErr) {
        maxErr = err;
        worst = `texel (${tx},${ty},${z}) lane ${lane}: want ${want.toFixed(4)}, got ${got.toFixed(4)}`;
      }
    }
  }
  return { maxErr, worst };
}

// Nearest sampling throughout: this spike is testing whether the texels arrive, and a linear filter
// would blur the very texel boundaries the comparison keys on.
const TOLERANCE = 0.02;

async function main(): Promise<void> {
  if (!("gpu" in navigator)) {
    verdictEl.className = "verdict fail";
    verdictEl.textContent = "WebGPU is not available in this browser.";
    return;
  }

  const renderer = new WebGPURenderer({ canvas, antialias: false });
  await renderer.init();
  const device = (renderer as unknown as { backend?: { device?: GPUDevice } }).backend?.device;
  if (!device) throw new Error("WebGPURenderer exposed no backend device after init()");

  // The whole point: our compute runs on the renderer's device, not one of our own.
  const backend = adoptDevice(device, "three");
  report(
    "0. Adopt the renderer's device",
    (await backend.getDevice()) === device,
    `renderer.backend.device adopted as a GpuBackend (kind "three"). Same GPUDevice object: ${(await backend.getDevice()) === device}`,
  );

  // ---- Case A: 2-D rgba16float, the tile-renderer shape --------------------------------------
  {
    const [w, h] = [64, 64];
    const gpuTex = device.createTexture({
      size: { width: w, height: h },
      format: "rgba16float",
      usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.TEXTURE_BINDING,
    });
    await fillTexture(device, gpuTex, w, h, 1, 4);

    const ext = new THREE.ExternalTexture(gpuTex);
    // ExternalTexture carries no `image`, and three reads width/height from it when sizing the
    // texture. Supplying it is the one adaptation this spike is really testing for.
    if (!OMIT_IMAGE) {
      (ext as unknown as { image: { width: number; height: number; depth: number } }).image = { width: w, height: h, depth: 1 };
    }
    ext.colorSpace = THREE.NoColorSpace;
    ext.minFilter = THREE.NearestFilter;
    ext.magFilter = THREE.NearestFilter;
    // Without these three warns `Unsupported texture wrap type "undefined"` — ExternalTexture's
    // defaults do not survive into the sampler descriptor.
    ext.wrapS = THREE.ClampToEdgeWrapping;
    ext.wrapT = THREE.ClampToEdgeWrapping;
    (ext as unknown as { wrapR: THREE.Wrapping }).wrapR = THREE.ClampToEdgeWrapping;

    try {
      const px = await renderAndRead(renderer, texture(ext, uv()), 128);
      const { maxErr, worst } = compare(px, 128, w, h, 1, 0, 4);
      report(
        "A. 2-D rgba16float sampled by a node material",
        maxErr <= TOLERANCE,
        `${w}×${h} rgba16float, filled by assembleTile("f16") + copyBufferToTexture, wrapped in\n` +
          `THREE.ExternalTexture and sampled with texture(ext, uv()).\n` +
          `max error ${maxErr.toFixed(4)} (tolerance ${TOLERANCE})${worst ? `\nworst — ${worst}` : ""}`,
      );
    } catch (e) {
      report("A. 2-D rgba16float sampled by a node material", false, `threw: ${e instanceof Error ? e.stack : String(e)}`);
    }
  }

  // ---- Case B: 3-D r16float, the volume-raymarch shape ----------------------------------------
  {
    const [w, h, d] = [32, 32, 16];
    const zCoordOf = (z: number): number => (z + 0.5) / d;
    /** Greyscale read of one z slice — the volume raymarch's sampling, reduced to one slice. */
    const sliceNode = (z: number): ColorNode => {
      const c = texture3D(ext, vec3(uv().x, uv().y, zCoordOf(z))).r;
      return vec4(c, c, c, 1) as unknown as ColorNode;
    };
    const gpuTex = device.createTexture({
      size: { width: w, height: h, depthOrArrayLayers: d },
      dimension: "3d",
      format: "r16float",
      usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.TEXTURE_BINDING,
    });
    await fillTexture(device, gpuTex, w, h, d, 1);

    const ext = new THREE.ExternalTexture(gpuTex);
    if (!OMIT_IMAGE) {
      (ext as unknown as { image: { width: number; height: number; depth: number } }).image = { width: w, height: h, depth: d };
    }
    // three picks the view dimension from `isData3DTexture`; an ExternalTexture is not one, so a
    // 3-D source would otherwise be bound as 2-D. Whether this flag is enough is the question.
    if (!OMIT_3D_FLAG) (ext as unknown as { isData3DTexture: boolean }).isData3DTexture = true;
    ext.colorSpace = THREE.NoColorSpace;
    ext.minFilter = THREE.NearestFilter;
    ext.magFilter = THREE.NearestFilter;
    // Without these three warns `Unsupported texture wrap type "undefined"` — ExternalTexture's
    // defaults do not survive into the sampler descriptor.
    ext.wrapS = THREE.ClampToEdgeWrapping;
    ext.wrapT = THREE.ClampToEdgeWrapping;
    (ext as unknown as { wrapR: THREE.Wrapping }).wrapR = THREE.ClampToEdgeWrapping;

    // TWO slices, deliberately. One slice can be matched by accident; two slices whose expected
    // values differ by 0.6·Δfz can only both match if the z coordinate is really selecting.
    try {
      const readSlice = async (z: number): Promise<Uint8Array> => renderAndRead(renderer, sliceNode(z), 128);

      let maxErr = 0;
      let worst = "";
      const seen: string[] = [];
      for (const zSlice of [2, 13]) {
        const px = await readSlice(zSlice);
        for (const [a, b] of [
          [0.5, 0.5],
          [1.5, 2.5],
          [2.5, 1.5],
          [3.5, 3.5],
        ] as const) {
          const sx = Math.floor((a / 4) * 128);
          const sy = Math.floor((b / 4) * 128);
          const tx = Math.min(w - 1, Math.floor(((sx + 0.5) / 128) * w));
          const ty = Math.min(h - 1, Math.floor((1 - (sy + 0.5) / 128) * h)); // flipY
          const want = pattern(tx, ty, zSlice, 0, w, h, d);
          const got = (px[(sy * 128 + sx) * 4] ?? 0) / 255;
          if (Math.abs(got - want) > maxErr) {
            maxErr = Math.abs(got - want);
            worst = `texel (${tx},${ty},${zSlice}): want ${want.toFixed(4)}, got ${got.toFixed(4)}`;
          }
        }
        seen.push(`z=${zSlice} centre ${((px[(64 * 128 + 64) * 4] ?? 0) / 255).toFixed(4)}`);
      }
      report(
        "B. 3-D r16float sampled by texture3D(), two z slices",
        maxErr <= TOLERANCE,
        `${w}×${h}×${d} r16float (a 3-D slab like the 3DxN store's chunk).\n` +
          `Lane 0 = 0.15·fx + 0.25·fy + 0.6·fz, so a collapsed z axis cannot pass both slices: ${seen.join(", ")}.\n` +
          `max error ${maxErr.toFixed(4)} (tolerance ${TOLERANCE})${worst ? `\nworst — ${worst}` : ""}`,
      );
    } catch (e) {
      report("B. 3-D r16float sampled by texture3D(), two z slices", false, `threw: ${e instanceof Error ? e.stack : String(e)}`);
    }

    // Corroboration for the eye: draw both cases side by side on the visible canvas.
    const scene = new THREE.Scene();
    const camera = new THREE.OrthographicCamera(-1, 1, 0.5, -0.5, 0, 1);
    const volMat = new MeshBasicNodeMaterial();
    volMat.colorNode = sliceNode(8);
    const volMesh = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), volMat);
    volMesh.position.x = 0.5;
    scene.add(volMesh);
    const twoDTex = new THREE.ExternalTexture(
      device.createTexture({
        size: { width: 64, height: 64 },
        format: "rgba16float",
        usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.TEXTURE_BINDING,
      }),
    );
    (twoDTex as unknown as { image: { width: number; height: number; depth: number } }).image = { width: 64, height: 64, depth: 1 };
    twoDTex.colorSpace = THREE.NoColorSpace;
    twoDTex.minFilter = THREE.NearestFilter;
    twoDTex.magFilter = THREE.NearestFilter;
    twoDTex.wrapS = THREE.ClampToEdgeWrapping;
    twoDTex.wrapT = THREE.ClampToEdgeWrapping;
    await fillTexture(device, twoDTex.sourceTexture as GPUTexture, 64, 64, 1, 4);
    const rgbMat = new MeshBasicNodeMaterial();
    rgbMat.colorNode = texture(twoDTex, uv());
    const rgbMesh = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), rgbMat);
    rgbMesh.position.x = -0.5;
    scene.add(rgbMesh);
    renderer.setSize(512, 256, false);
    renderer.render(scene, camera);
  }

  const flags = [OMIT_IMAGE ? "noimage" : null, OMIT_3D_FLAG ? "no3dflag" : null].filter(Boolean);
  if (flags.length) {
    const el = document.createElement("div");
    el.className = "case";
    el.innerHTML = `<h2>Negative control <span class="tag">${flags.join(" + ")}</span></h2><pre>Shim(s) deliberately omitted. A PASS here means the shim is NOT required; a FAIL means it is.</pre>`;
    casesEl.prepend(el);
  }

  const allOk = results.every((r) => r.ok);
  verdictEl.className = `verdict ${allOk ? "pass" : "fail"}`;
  verdictEl.textContent = allOk
    ? `PASS — ${results.length}/${results.length} cases. three samples our device-allocated textures; the Loader can return one.`
    : `FAIL — ${results.filter((r) => r.ok).length}/${results.length} cases passed. See details below.`;
  console.log(`[spike] verdict: ${verdictEl.textContent}`);
}

main().catch((e) => {
  verdictEl.className = "verdict fail";
  verdictEl.textContent = `threw: ${e instanceof Error ? e.message : String(e)}`;
  console.error("[spike] threw", e);
});
