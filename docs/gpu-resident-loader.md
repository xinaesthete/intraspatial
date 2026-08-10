# A device-resident Loader — where the seam goes, and what zarrextra would have to grow

The `Loader` (ADR-0008 §7) hands back a `Tile` whose samples are a host `Float32Array`. That is the
right shape for a CPU consumer and the wrong shape for every consumer we actually have: the tile
renderer uploads it to a texture, and the op-graph uploads it to a buffer. A GPU codec makes it worse
still — it would decode *on the device* and then be asked to read the result back so the caller could
upload it again.

This note works out where a "return a reference to a device-side buffer" seam belongs, what the
generic fallback looks like, and what a client with its own decoder hooks into. Companion to
[ADR-0008](./decisions/0008-view-driven-multiscale-datasource.md) (the Loader seam and the
CPU-decode decision in §9), [ADR-0017](./decisions/0017-tier2-resident-buffer-edges.md) (the resident
payload vocabulary this reuses), and [`zarrextra-worker-decode.md`](./zarrextra-worker-decode.md)
(what it took to get the current CPU path working).

## 1. What the host actually does per tile today

The steps below are what the code does
([`spatialDataLoader.ts:247`](../playground/src/datasource/spatialDataLoader.ts),
[`tileRenderer.ts:168`](../playground/src/datasource/tileRenderer.ts)); the figures are arithmetic for
one worked example — a 4-channel 1024² tile — not a measurement.

| Step | Where | Cost |
| :--- | :--- | :--- |
| 4× fetch + HTJ2K decode | zarrextra worker pool | off-main-thread ✅ |
| 4× `Uint16Array` transferred to main thread | postMessage | 8 MB |
| normalise `/norm` + interleave 4 planes → one `Float32Array` | **main-thread JS loop** | 4.2 M iterations, 16 MB allocated |
| f32 → half-float via `DataUtils.toHalfFloat` | **main-thread JS loop** | 4.2 M iterations, 8 MB allocated |
| `DataTexture` upload | queue | 8 MB |

Two full passes over every sample on the main thread, and 24 MB of intermediates, to move data that
the worker already decoded into a texture. That is the cost the current interface *forces*: `Tile.data`
is a host array, so the only place to do this work is the host.

Note what this implies for sequencing. **Both host loops are pure format conversion — they are not the
codec.** They disappear the moment the Loader is allowed to hand back a device payload, whether or not
we ever decode a codestream on the GPU. That is the first slice, and it is independent of the codec work
(§7).

### 1a. Measured, on the 3DxN volume

Steps 1 and 2 of §9 are now built (`src/gpu/tiles/assemble.ts`, `pnpm bench:assembly`), so the rest of
this section is measurement rather than arithmetic. The subject is
`/Volumes/CrucialOx9/3DxN/8090_13_Punch1_fused_htj2k.zarr`, which is the interesting case because it is
volumetric:

| | |
| :--- | :--- |
| axes / shape | `(t, c, z, y, x)` = `1 × 2 × 6822 × 2376 × 3101`, `uint16`, 6 levels |
| chunk | `[1, 1, 32, 512, 512]` — a **32-slice slab**, 8.39 M voxels, **16 MB decoded** |
| codecs | `[experimental.openjph_htj2k]` — one codec, **no sharding**, so a chunk file *is* the codestream |
| compressed | 5.1–5.3 MB per chunk (≈3.1×); 82 GB for the lossless variant, 125 GB for the store |
| level-0 chunk count | 14 980 (214 z × 5 y × 7 x × 2 channels) |
| decode | HTJ2K encodes the 32 z-slices as **32 components of one codestream** |

Per level-0 brick, medians over three chunks (`pnpm bench:assembly`, Dawn, M-series):

| Stage | Today (host) | With the assembly pass | |
| :--- | ---: | ---: | :--- |
| HTJ2K decode (openjph-wasm) | 89–96 ms | 89–96 ms | unchanged — this is the codec, not the conversion |
| normalise `/65535` → `Float32Array` | 8–10 ms | — | |
| quantise → `Uint8Array` (R8) | 14–26 ms | — | |
| upload + assemble + fill texture | — | **5.6–6.0 ms** | one run paid 15.8 ms on the first brick (pipeline compile); not reproduced |
| **conversion subtotal** | **24–35 ms** | **5.6–6.0 ms** | **4–6×** |
| transient host allocation | 40 MB | 0 | 32 MB f32 + 8 MB R8, per brick, straight to the GC |
| host loop iterations | 16.8 M | 0 | |

The host-side variance is itself telling: the quantise loop ranges 14–26 ms for identical work, which
is GC pressure from the 40 MB it allocates each time. The device path has no such spread.

Three things worth reading off that:

- **The conversion is ~25 % of per-brick wall-clock, and 100 % of the part we control.** Decode dominates
  at ~92 ms and the assembly pass does not touch it. End-to-end a brick goes ~120 ms → ~98 ms.
- **On the volume page that 25 % is all main-thread.** `spatialDataVolume.ts` deliberately does *not*
  call `enableWorkerChunkDecode()` — the shipped worker bundle has cornerstone baked in and would bypass
  its openjph-wasm shim — so decode is on the main thread there already. The thread-affinity objection in
  §7 therefore **does not apply to the volumetric path**: there is no off-main-thread win to lose, which
  makes the volume the better first target for GPU decode, not the harder one.
- **It buys precision back for free.** The volume path quantises 16-bit samples to R8 today
  (`naiveVolumeRenderer.ts:122`); the device path lands them in `r16float` at the same cost. That doubles
  the per-brick VRAM (8 → 16 MB), so it wants to be a knob rather than a silent change — but the choice
  currently costs a whole extra host pass, and after this it costs nothing.

## 2. The constraint that decides everything: the codec seam cannot be the GPU seam

The obvious place to hook a GPU decoder is zarrextra's codec registration, since that is where the
HTJ2K decoder already plugs in. It cannot work, for two independent reasons:

```ts
// zarrextra/dist/codecs.d.ts
export type DecodedImageBytes = ArrayBuffer | ArrayBufferView;
export type ImageCodecDecoder =
  (encoded: Uint8Array, meta: ChunkMetadata, config: unknown) => Promise<DecodedImageBytes> | DecodedImageBytes;
```

1. **The return type is host bytes**, by construction — it feeds zarrita's chunk-assembly path, which
   slices, fills, and concatenates chunks into a typed array. A `GPUBuffer` cannot travel that path.
2. **More fundamental: decode runs in a worker.** `enableWorkerChunkDecode()` routes the whole decode
   through the `@fideus-labs/worker-pool` (that is the point of it, and we argued for it in
   `zarrextra-worker-decode.md`). WebGPU objects are neither transferable nor structured-cloneable, and
   a device created *in* a worker is a different device from the renderer's — a buffer allocated there
   is unusable by the thing that wants to draw it. Thread affinity is not negotiable: **whatever produces
   a device-side buffer must run on the thread that owns the device.**

So the device-side seam sits one layer **above** the codec, at the chunk accessor, and zarrextra's codec
registry stays exactly as it is. This is also why the hook must take **encoded bytes** as input, not
decoded pixels: a hook that receives pixels has already lost, because someone has already paid for a
full-resolution host plane. Give the decoder the codestream and let it decide where the work happens.

## 3. The split: what each side owns

| Concern | Home | New surface |
| :--- | :--- | :--- |
| Fetch, sharding, chunk-key algebra, codec metadata, CPU decode + worker pool | **zarrextra** | one raw-chunk accessor (§6). No WebGPU types, ever. |
| Device payloads, decoder registry, sample assembly, LOD/cache/render | **this repo** | `Tile` widens; `GpuChunkDecoder`; one assembly pass |
| Domain model, coordinate systems, element naming | **spatialdata.js** | none |

The upstream ask is deliberately tiny and dependency-free: zarrextra should not learn what a `GPUDevice`
is. It only has to stop insisting that decode has happened by the time we get the chunk.

## 4. The interface here: `Tile` widens, `Loader` does not

`FieldValue` already solved this exact problem for graph edges (ADR-0017): one value type that carries
*either* host data or a device payload, with the executor bridging on demand
([`handle.ts:262`](../src/gpu/graph/handle.ts)). A `Tile` is the same kind of thing, so it should
borrow the same vocabulary rather than invent a second one:

```ts
// src/datasource/types.ts
export interface Tile {
  readonly id: ChunkId;
  readonly dims: readonly [number, number, number];
  readonly element: ElementType;
  readonly dtype: Dtype;
  /** Host samples — lane-major interleaved, as today. */
  readonly data?: Float32Array;
  /** Device-resident samples: same logical contents and layout as `data` (ADR-0017). */
  readonly buffer?: ResidentBuffer;
  /** Device-resident as a sampled texture — what the render backend wants, and what a
   *  decoder that writes a storage texture directly can produce with no copy at all. */
  readonly texture?: ResidentTexture;
  // INVARIANT: at least one of data / buffer / texture is present.
}
```

`ResidentBuffer` / `ResidentTexture` already exist and `types.ts` already imports from
`gpu/graph/handle`, so this adds no new dependency direction and no new concepts.

**`Loader` itself is unchanged** — still `getChunk(id): Promise<Tile>`. Which residency you get is a
property of *how the loader was constructed*, not of the call:

```ts
const loader = await openSpatialDataImage(url, "morphology_focus", { device });   // device-resident tiles
const loader = await openSpatialDataImage(url, "morphology_focus");               // host tiles, as today
```

Construction-time rather than call-time because a loader serves exactly one device in every arrangement
we have, and because it keeps every existing consumer compiling. The call-time variant
(`getChunk(id, target)`) is the thing to reach for only if one loader ever has to feed two devices.

Consumers that genuinely need host samples get two accessors rather than one, mirroring the executor's
bridge (this is what was built):

```ts
/** Samples of a host-resident tile. THROWS on a device-resident one — the overwhelming majority of
 *  call sites want exactly this, and making them say so is the point. */
export function hostSamples(tile: Tile): Float32Array;

/** The readback bridge. Deliberately explicit and async: a readback is the thing this whole seam
 *  exists to avoid, so it belongs at the call site rather than behind a getter. */
export async function tileToHost(tile: Tile, backend: GpuBackend): Promise<Tile>;
```

Splitting them this way is what makes the migration mechanical *and* honest: `tsc` names every site
that assumed host data (there were seven), and each one either states the assumption or takes a
visible readback. A single async accessor would have silently made every one of them a potential
device→host round-trip.

## 5. The decoder hook, and the fallback

Registration mirrors zarrextra's own `registerExperimentalHtj2kCodec()` shape, so it reads familiarly:

```ts
export interface EncodedChunk {
  readonly bytes: Uint8Array;
  readonly codecId: string;        // e.g. "experimental.openjph_htj2k"
  readonly config: unknown;        // the codec's zarr config blob
  readonly dims: readonly [number, number, number];
  readonly storeDtype: string;     // "uint16", "uint8", …
}

export interface DecodeContext {
  readonly device: GPUDevice;
  readonly backend: GpuBackend;    // lease / upload / release (interop/adoptDevice)
  readonly signal?: AbortSignal;
}

export interface GpuChunkDecoder {
  readonly name: string;
  /** Cheap and synchronous. `false` ⇒ this chunk falls back to the CPU path. */
  accepts(meta: Omit<EncodedChunk, "bytes">, device: GPUDevice): boolean;
  /** One decoded plane, device-resident, in the tile's `dtype`. */
  decode(chunk: EncodedChunk, ctx: DecodeContext): Promise<ResidentBuffer>;
}

export function registerGpuChunkDecoder(d: GpuChunkDecoder): void;
```

**A decoder returns one plane, not a tile.** Normalisation, interleaving N channel planes into lanes,
and conversion to the render texture's format are *not* the decoder's job — they are one shared compute
pass we own, because:

- WebGPU storage textures are written per-texel as a whole `vec4`; you cannot have four decoders each
  write "their" channel of one RGBA texture. Planes must land in buffers and be packed by a pass that
  sees all of them.
- It is the same pass for every decoder, including the fallback, so writing it once is what actually
  deletes the host loops in §1.

The fallback is then not a special case, just the decoder of last resort: no registered decoder accepts
the chunk ⇒ call zarrextra's ordinary `getTile` (worker, CPU, unchanged), `queue.writeBuffer` the raw
`Uint16Array` plane, and hand it to the same assembly pass. **The fallback still wins**, because the
`u16 → normalise → f32 → interleave → f16` chain moves to the GPU and the upload halves (u16 planes
instead of an f32 interleave).

## 6. The upstream ask (zarrextra)

One accessor, no new dependency:

```ts
interface VivCompatiblePixelSource {
  // …existing getTile / getRaster…
  /** The chunk's stored bytes and codec metadata, without decoding. */
  getTileEncoded?(props: { x: number; y: number; selection: RasterSelection; signal?: AbortSignal }):
    Promise<{ bytes: Uint8Array; codecId: string; config: unknown; width: number; height: number; dtype: string }>;
}
```

Why upstream rather than reading the store ourselves: for a plain unsharded v3 array we *can* just read
the chunk file and skip zarrextra entirely. That is not speculative — `scripts/bench-tile-assembly.ts`
does exactly this against the 3DxN store today: the array declares one codec and no sharding, so
`{array}/c/{t}/{c}/{z}/{y}/{x}` **is** the HTJ2K codestream, and openjph-wasm decodes it directly. Zero
upstream change.

**But that escape hatch is precisely what sharding closes**, which is why it is worth reviewing
zarrextra soon rather than later. Once a "chunk" is a slice inside a shard, getting its bytes means
parsing the shard index — and chunk-key construction, and the codec chain that may wrap a
byte-shuffle or compressor *around* the image codec, are all zarrextra's job. Reimplementing them here
is exactly the duplication the layering table exists to prevent. The pressure toward sharding is real
for this data: level 0 of one variant is already **14 980 chunk files**, and the full store is ~50 000.
So today the accessor is a convenience; the first sharded store makes it load-bearing, and a GPU
decoder without it would simply be blocked.

Optional method, feature-detected — absent ⇒ we take the CPU fallback, which is where we already are.
Worth pairing with the items already listed in
[`zarrextra-worker-decode.md`](./zarrextra-worker-decode.md) when we file upstream.

## 7. What this buys the GPU codec — and the honest catch

The endpoint ADR-0008 §9 deferred ("the coefficient-domain decode hook reopens once our codec overtakes
OpenJPH") becomes reachable without a second redesign: upload the compressed codestream (small), run the
HT block decoder and the IDWT on the device (`idwt53.ts` / `idwt97.ts` exist), never materialise a
full-resolution plane on the host at all.

**The catch is thread affinity — but only on the 2-D path.** GPU decode must run on the device-owning
thread, which is the main thread, so a naive "our decoder" would claw back the off-main-thread win that
`enableWorkerChunkDecode()` bought the *image* viewer. The volume viewer never had that win (§1a), so
there is nothing to lose there — and it is the volume where decode is 92 ms per brick and hurts most.

The entropy decode (the HT block coder, `block-decoder-port.md`) is the expensive CPU part, and it is
expensive per code-block. The arrangement that keeps both wins on the 2-D path:

- **worker:** bitstream → *coefficients* (wasm block decode; the result is a transferable `ArrayBuffer`);
- **main thread:** coefficients → upload (i16, half the bytes of f32 pixels) → dequant + IDWT + assemble
  on the GPU.

so the `GpuChunkDecoder` for HTJ2K is itself two-stage, and its `decode` is mostly `await worker` plus a
submit. Only when the block decoder itself runs on the GPU (which is what HT is designed for — the
cleanup pass is per-code-block parallel) does the worker stage disappear.

The volume store is unusually well shaped for that endpoint: a chunk is **one codestream with 32
components** (one per z slice), so a single chunk already carries 32 independent 512×512 subimages'
worth of parallelism, and 5.2 MB of compressed bytes stand in for 16 MB of samples on the upload.

## 8. Ownership, lifetime, and the cache

- The Tier-2 `BufferPool`/`TexturePool` recycle by liveness; a tile texture lives in the LRU
  `TileCache` for as long as the camera cares about it. **Tile payloads are owned, not leased** —
  leasing a transient-pool buffer for a long-lived cache entry would defeat the pool. `TileCache`'s
  existing `dispose` hook is the free.
- `SelectedChunk.approxBytes` keeps meaning what it says (decoded sample bytes, the working-set
  estimate); the resident texture is typically *smaller* (fp16), as the comment there already notes.
- The device must come from the **host renderer**, never `navigator.gpu` — `adoptDevice`
  ([`src/gpu/interop/adoptDevice.ts`](../src/gpu/interop/adoptDevice.ts)) is that seam, and the
  playground already adopts three's device in `datasourceMain.ts`.
- three.js side: **verified** (`playground/externaltexture.html`, three 0.185, Chrome/Dawn). A
  `GPUTexture` we allocate on `renderer.backend.device` and fill with the assembly pass samples
  correctly through `THREE.ExternalTexture`, in both shapes that matter: 2-D `rgba16float` via
  `texture(ext, uv())` and 3-D `r16float` via `texture3D()`. Max error 0.0015 against the pattern
  written in, which is fp16 plus the rgba8 readback. Details in §8a.

### 8a. What the ExternalTexture spike actually found

`playground/externaltexture.html` is self-checking rather than eyeballed: it fills a texture through
the real assembly pass, renders it, reads the pixels back, and compares against the pattern it wrote.
Both cases pass. Two things are worth carrying forward, and neither was visible from reading three's
source:

- **`texture.image` is NOT required.** The obvious adaptation — telling three the extent, since an
  `ExternalTexture` has no image — turns out to be unnecessary; `createTexture` returns early for
  external textures, before anything needs it. (`?noimage=1` on the spike page still passes 3/3.)
- **`isData3DTexture = true` IS required for a 3-D source, and omitting it fails silently-ish.**
  Three picks the bind-group view dimension from that flag alone (`WebGPUTextureUtils._getDimension`),
  so without it a 3-D texture is bound as `texture_2d`, the generated WGSL fails to compile
  (`expected 'vec2<f32>', got 'vec3<f32>'`), and **the quad renders black with no exception thrown**.
  Three does log the pipeline error, so it is diagnosable — but the visible symptom is "no data",
  which is exactly what a loading bug looks like. (`?no3dflag=1` reproduces it.)

Setting `wrapS`/`wrapT`/`wrapR` explicitly is also needed to avoid three warning
`Unsupported texture wrap type "undefined"` — cosmetic, but it is noise in a console you will be
reading for the errors above.

The spike also caught a defect in its own first draft, which is the reason to write assertions rather
than look at a picture: the 3-D case originally filled the single-lane texture with `fx`, which does
not vary with z at all, so "sampling z slice 8" would have passed identically had three ignored the
z coordinate entirely. Lane 0 now mixes all three axes (`0.15·fx + 0.25·fy + 0.6·fz`) and two slices
are compared, so a collapsed z axis cannot pass.

## 9. Slices, and where we are

1. ✅ **`Tile` widened** to `data?` / `buffer?` / `texture?`, with `hostSamples` (sync, throws on a
   device-resident tile), `tileToHost` (the explicit readback bridge), and `tileBytes` for the cache
   ceiling. Every existing consumer now says which residency it assumes, at the call site. No behaviour
   change: 732 CPU tests and both typechecks green.
2. ✅ **The assembly pass** — `src/gpu/tiles/assemble.ts`, verified against the host loops it replaces
   in `assemble.gpu.test.ts` (8/16/32-bit unpack, half-pack with row padding, and the full chain into an
   `rgba16float` texture read back through `textureLoad`). Benched on real chunks: §1a.
3. ✅ **`THREE.ExternalTexture` spike** — `playground/externaltexture.html`, self-checking, 2-D and
   3-D both correct on the adopted device (§8a). The last unverified link is closed: nothing now
   stands between a device-resident `Tile` and a rendered pixel.
4. 🟡 **Wire it into a Loader** — the **volume half is done**: `spatialDataVolume` takes `{ device }`
   and returns a texture-resident `Tile`, `NaiveVolumeRenderer` consumes it via `ExternalTexture`,
   and the whole path is verified against the real 3DxN store (§9a). The **2-D half**
   (`spatialDataLoader` + `TileRenderer`) is not started; it is the more involved one, because it is
   multi-channel (so it exercises the 3-lane RGBA padding the volume never does) and it interacts
   with the zarrextra worker decode the volume path does not use.
5. ⬜ **`registerGpuChunkDecoder`**, the raw-bytes accessor (or the escape hatch), and the two-stage
   HTJ2K decoder. Deliberately last: it is the part that needs the codec to be good.

Two findings from building 1–2, both of the "silent failure" family this repo keeps collecting:

- **`writeBuffer` validates the SOURCE size, not just the destination**, and rejects a view whose
  byteLength is not a multiple of 4 — which an odd-length 8/16-bit edge chunk is. One padded copy on
  that branch only.
- **The GPU and host paths are not bit-identical**, and asserting that they are fails. JS computes
  `v * scale` in f64 and rounds once on the store to `Float32Array`; WGSL multiplies two f32s. For
  8-bit that lands on opposite sides of the last ulp (13/255). Agreement to f32 epsilon is the honest
  contract, and that is what the test asserts.

### 9a. Verifying the volume path against the real store

The pane in this environment renders at 0×0, so the canvas proves nothing — the verification is
numeric instead, and is stronger for it. Two `SpatialDataVolume`s were opened on the same store, one
with `{ device }` and one without, and the same chunk pulled through both:

| | |
| :--- | :--- |
| chunk | level 5, `(0,0,3)` → dims `97×75×32`, 232 800 voxels |
| device tile | `texture`, `r16float`, depth 32, no `data` |
| host tile | `data`, 232 800 f32, no texture |
| max \|device − host\| | **1.5 × 10⁻⁵** over every voxel |

That error is exactly fp16 quantisation at these magnitudes (samples ≈ 0.03, fp16 relative precision
≈ 4.9 × 10⁻⁴), and it exercises the parts most likely to be wrong: the strided sub-box gather, the
channel offset, and the normalisation — all against the host loop that was already known good.

Live, the page reports **118 MiB resident for 19 bricks** where the host path reports **59 MiB** for
the same selection. The factor of two is not a bug: it is `r16float` against `R8`, i.e. the 16-bit
precision the host path was throwing away, now kept for free. It also doubles as proof the device
path is genuinely in use, since the host path cannot produce that number.

## Open questions

- ~~Does the fallback actually win?~~ **Yes, measured: ~5–6× on the conversion, 40 MB/brick of transient
  allocation deleted (§1a).** The remaining unknown is the multi-channel 2-D case, where the upload is
  N separate u16 planes rather than one interleaved buffer — N `writeBuffer` calls per tile. The volume
  is single-plane, so this bench does not exercise it.
- **fp16 everywhere?** The 2-D renderer stores fp16 and the volume path quantises to R8; the device path
  makes `r16float` free in time but doubles VRAM against R8. It wants to be an explicit precision knob,
  which `tileRenderer.ts` already flagged as a future item — now there is a reason to build it.
- **Does the same interface hold for the points/AnnData datasource?** ADR-0008 flagged grid over-fit as
  the standing risk. A device-resident points tile is arguably *more* natural than a grid one, but it
  has not been designed against.
- **Sharding.** Not used by this store, but its ~50 000 chunk files are exactly the pressure that leads
  to it. Reviewing zarrextra's sharding support is worth doing *before* the decoder work, since sharding
  closes the escape hatch that step 5 would otherwise start from (§6).
- **Does the brick shape want to change?** A `[32, 512, 512]` chunk is a slab, and the brick-atlas
  `VolumeRenderer` assumes a cube (`B = ms.chunkShape[0]`, i.e. 512 → a 512³ = 134 MB atlas entry),
  which is why the real store goes through `NaiveVolumeRenderer` instead. Unrelated to this seam, but it
  is the next thing that limits the volume path.
