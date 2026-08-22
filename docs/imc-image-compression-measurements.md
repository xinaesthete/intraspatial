# Compressing the COVID IMC stack: measured

**Status: IMPLEMENTED (2026-08-01) — `docs/covid-spatialdata-migration.md`.** The recommendation
below was carried out unchanged and its numbers reproduced exactly: 131.1 MB per 2000² ROI at
zstd-19 for level 0, and 519.8 MB of HTJ2K against 681.2 MB of PNG (1.31×) for the H&E. Three things
to carry forward:

- **The stacks are already 2-level pyramids.** `tf.pages` shows 49 planes, `tf.series[0].levels`
  shows two. That is the 197.5 MB / 247.4 MB discrepancy between "LZW" here and the file on disk, and
  it is why the dataset total is 6.43 GB rather than 6.0. The stored store carries the level too:
  4.36 GB against 6.43 GB, like for like.
- **The source's level 1 is an exact 2× subsample, and reproducing it beats a block mean by 1.67×**
  (33.8 MB against 56.4 MB per ROI). Averaging float32s manufactures bit patterns that were not in
  the data — the same mechanism as finding 2, from the other direction.
- **The HTJ2K choice for the H&E gates the whole store**: `spatialdata.read_zarr` fails with
  `UnknownCodecError` without `spatialdata-js-util` installed, for every element, not just the 30
  H&E. Clean error, total effect, and not noted when the codec was chosen.

Companion to
`docs/covid-imagery-to-spatialdata-plan.md`, which left "lossless or lossy for the IMC stack?" as
open question 1 and guessed that "LZW on float32 is close to the worst case, so the headroom may be
large." **That guess is wrong**, and the measurements below are the reason it is worth writing down
rather than quietly correcting: LZW turns out to be a strong baseline on this data, and the obvious
upgrade path makes the files *bigger*.

Prompted by considering `~/code/www/SpatialData.ts/python/spatialdata-js-util` for the job. The
assessment of that tool is in the last section; it is a good fit for exactly one of the three image
kinds here and a trap for the other two.

Everything below is measured, not cited, and reproducible:

```bash
uv run --with tifffile --with numpy --with imagecodecs --with numcodecs \
  python scripts/imc-compression-measure.py stack "/Volumes/Crucial X8/covid/images/uDFaO.ome.png"
```

`stack` regenerates findings 1–2, `crossover` finding 3, `png` finding 5. It is Python rather than
TypeScript on purpose: the point is to measure the codec `spatialdata-js-util` actually calls,
through the same binding it calls it with.

## What was measured

- **Stack**: `uDFaO.ome.png` (COVID_SAMPLE_16_ROI_3), 49 channels × 2000² float32, 784.0 MB raw,
  LZW-compressed in the file. Two further stacks (`AaSUN`, `zPALd`) used to confirm the lossless
  result generalises.
- **H&E**: `KroHka.png`, the same ROI's morphology image.
- **cellmask**: `q9Qtix.png`, the same ROI's binary mask.
- **Codec**: OpenJPH via `imagecodecs` 2026.6.26 — the same backend `spatialdata-js-util` prefers,
  called the same way its `ImagecodecsHtj2kBackend.encode` calls it.
- **Error metric**: relative error of **10 × 10 block means**, p99 over blocks carrying signal. At
  1 µm/px a cell is ~10 px across, so a block mean is a proxy for the per-cell mean intensity that
  the downstream statistics actually consume. Per-pixel max error is reported alongside where it
  changes the reading. Scoring on raw per-pixel MSE would have flattered every scheme here, because
  most of every channel is empty background.

Corrections to the plan doc while in there: the 32 stacks are **not** all 2000² — eight have other
extents, from 930 × 2750 to 1250 × 3500 — so the true raw total is **22.3 GB**, not the 25 GB
extrapolated from one file. All 32 do have 49 channels.

## The blocking finding

`spatialdata-js-util` refuses float32 outright:

```python
SUPPORTED_BROWSER_JP2K_DTYPES = {uint8, int8, uint16, int16}   # images.py:42
```

> Browser image codecs support only <=16-bit integer dtypes …; use Blosc for labels or skip this
> raster.

This is not a limitation of the tool so much as of the browser codecs it targets, and it is correct
to enforce. But it means **using the tool on the IMC stack at all requires choosing a quantisation
first** — which is a scientific decision, not a packaging one, and everything else follows from it.

## Two dead hypotheses, tested before assuming

If the float32 were really integer ion counts, uint16 would be free and there would be no decision to
make. It isn't:

1. **Are the values integers?** No. Integer-valued fractions per channel run from 0.01% (`80ArAr`) to
   98.7% (`Va7-2`), and the fraction tracks the zero fraction almost exactly — i.e. the zeros are
   integers and the *non-zeros mostly are not*. The dense channels are the least integral: `DNA3` is
   5.1% integer, `DNA1` 11.4%.
2. **Are they a lattice `{k·c}`** — counts times one per-channel compensation factor? Also no. Taking
   `c` as the smallest non-zero value, the fraction of distinct values landing on the lattice peaks
   at 62% and is usually 5–25%. Zero channels of 49 are exactly representable as `uint16 × scale`.

So the float32 carries real non-integer structure, and any conversion to 16-bit is lossy. Worth the
two probes: had either come back yes, the whole question would have collapsed.

*(Some suggestive residue, not chased: the minimum non-zero value drifts smoothly with channel index
— 0.031080, 0.031158, 0.031237, 0.031314 for channels 7–10 — which looks like a per-channel
correction applied in acquisition order. It doesn't change any decision here.)*

## Finding 1 — LZW is a strong baseline, and the natural replacement is worse

The plan assumed LZW on float32 was near worst-case. It isn't: this data is *repetitive-symbolic*
(long runs of exact zeros, few distinct bit patterns per channel), which is precisely what a
dictionary coder eats. The standard scientific-array reflex — byte-shuffle before an entropy coder —
is the **wrong pre-filter** here, because shuffle assumes smooth-continuous values:

| lossless, float32 preserved | per ROI | vs raw | vs LZW |
|---|---|---|---|
| raw float32 | 784.0 MB | 1.00× | — |
| **LZW (today)** | **197.5 MB** | 3.97× | 1.00× |
| zstd-5 + byteshuffle | 232.9 MB | 3.37× | **0.85×** |
| zstd-5 + bitshuffle | 284.5 MB | 2.76× | **0.69×** |
| zstd-5, no shuffle | 161.7 MB | 4.85× | 1.22× |
| **zstd-9, no shuffle** | **140.0 MB** | 5.60× | **1.41×** |

Confirmed on two further stacks: zstd-9/no-shuffle gives 1.37× and 1.40×; zstd-5/shuffle gives 0.96×
and 0.86×. **Blosc's shuffle is on by default**, so a store written without thinking about it lands
in the row that is worse than the LZW we already have.

That is a 1.4× win available today, losslessly, with no codec extension and no browser involvement.

## Finding 2 — quantising to 16-bit costs more than the codec wins back

Using `arcsinh(x/5)` as the quantiser (justified in finding 4), across all 49 channels:

| browser-decodable | per ROI | vs raw | vs LZW | block p99 error |
|---|---|---|---|---|
| arcsinh u16, HTJ2K reversible | 301.9 MB | 2.60× | **0.65×** | 0.001% (quantiser only) |
| arcsinh u16, HTJ2K `balanced` | 244.0 MB | 3.21× | **0.81×** | median 0.13%, worst 1.6% |
| arcsinh u12, HTJ2K reversible | 218.4 MB | 3.59× | **0.90×** | median 0.061%, worst 0.364% |
| arcsinh u8, HTJ2K reversible | **130.1 MB** | 6.03× | **1.52×** | median 1.04%, worst 5.70% |

**Lossless HTJ2K on 16-bit is 53% larger than doing nothing.** The reason is visible per channel: the
sparse channels dominate the loss. `Ki67` is 0.47 MB under LZW and 4.46 MB under HTJ2K — a 9×
expansion — because quantising a 95%-empty channel with a tiny value range spreads its signal across
the full 16-bit range and turns quantisation noise into genuine entropy for the wavelet coder to
carry. The dense channels go the other way (`80ArAr` 16.70 MB → 5.97 MB), but there are fewer of
them.

So bit depth, not codec, is the knob that matters — and only at 8 bits does anything browser-
decodable beat the status quo.

## Finding 3 — both shipped presets are below the lossy/lossless crossover for ≤8-bit input

This one is a defect rather than a trade-off, and it is not specific to the arcsinh choice.

`HTJ2K_PRESETS` (images.py:54) sets `balanced` → `quality=0.0002` and `small` → `quality=0.001`.
`quality` is the OpenJPH quantisation step, and the README is careful that lower means *finer*. For
8-bit input both values are finer than the data's own resolution, so the irreversible 9/7 path spends
more bits than the reversible 5/3 path **and returns the identical image**:

On the H&E (uint8 RGBA, PNG 29.77 MB):

| encode | size | vs PNG | max error |
|---|---|---|---|
| HTJ2K `lossless` | 22.71 MB | 1.31× | 0 |
| HTJ2K `balanced` (0.0002) | 43.96 MB | 0.68× | **0** |
| HTJ2K `small` (0.001) | 31.41 MB | 0.95× | **0** |
| HTJ2K level 0.005 | 17.82 MB | 1.67× | 255 (mean 0.24) |
| HTJ2K level 0.02 | 7.75 MB | 3.84× | 255 (mean 1.17) |

`balanced` is **94% larger than `lossless` and bit-identical to it**. Both lossy presets are strictly
dominated: a user asking for a smaller file gets a bigger one, with no quality cost to show for it,
and nothing in the output says so. The same holds on every 8-bit array measured, and a sweep locates
the crossover at level ≈ 0.005 for 8-bit and below 0.0002 for 16-bit.

A related consequence, from the same sweep: **at a fixed `level`, codestream size is essentially
independent of input bit depth** (`80ArAr` at level 0.005: 1.34 MB from uint8, 1.33 MB from uint16),
while the error is not (0.482% vs 0.393%). So for a lossy encode, **never pre-quantise to 8 bits** —
feed uint16 and let the codec quantise. Pre-quantising pays the full error and saves nothing.

## Finding 4 — arcsinh is the only quantiser that survives the dynamic range

IMC channels are extraordinarily heavy-tailed: `CD10` has p99.9 = 3.5 and max = 20,295, a ratio of
5,800. Measured on six representative channels:

| quantiser | what it does | worst block p99 error |
|---|---|---|
| linear to max | step = max/65535 | **7.3%** (`CD10`) — the real 0–3.5 signal gets ~11 levels |
| clip at p99.99 | rescale after clipping | **2.6%**, and per-pixel error is the *entire* outlier (2 × 10⁴) |
| **arcsinh(x/5)** | the transform IMC is displayed in anyway | **0.028%** |

Naive max-scaling is the trap the dynamic range sets, and it fails silently: the image still looks
plausible. Percentile clipping fails differently — block error looks fine at the median and reaches
99.5% on the blocks containing hot pixels, which for some markers *are* the signal. arcsinh is
invertible, so it is a non-uniform quantiser rather than a transform of the data, and it is what
every IMC viewer displays through regardless.

## Finding 5 — the two 8-bit PNGs, and where the tool does fit

**H&E is the clean case.** 8-bit RGB, perceptual, no quantitative downstream use, inside the
supported dtypes. HTJ2K beats PNG losslessly (1.31×) and by 3.8× at level 0.02 with mean error 1.2/255.
Two incidental findings: the alpha channel is **constant 255** across the image (drop it — the codec
already squeezes it to 15 KB, but it is noise in the schema), and the 30 H&E images have **30
different extents** and per-image offsets — `he` for this ROI is 3630 × 3630 px mapped to
1857.2 × 1857.2 µm at offset (26.5, 79.5), i.e. ~0.51 µm/px against the IMC grid's 1.0. Each needs
its own affine, which is an argument *for* SpatialData rather than a complication.

**cellmask must never go through the image codec.** Binary, two distinct values, and HTJ2K expands it
**8×** (0.39 MB PNG → 3.21 MB reversible). zstd-9 gives 0.29 MB. The tool gets this right by routing
labels to Blosc — the trap is only that `cellmask` is not currently a labels element, so a careless
config would hand it to the image path.

## Finding 6 — lossy does change the picture, but not in the direction of "use HTJ2K"

Findings 1–2 compared lossless against lossless, which is not the comparison anyone would actually
make: zstd has no lossy mode, so if error is acceptable the codec should win. Sweeping the whole
49-channel stack at real lossy operating points (not the broken presets of finding 3):

| whole stack, per ROI | size | vs zstd-9 lossless | block p99 error: median | p90 | worst channel |
|---|---|---|---|---|---|
| zstd-9 float32 | 140.0 MB | 1.00× | exact | exact | — |
| arcsinh u16, HTJ2K level 0.0005 | 209.6 MB | **0.67×** | 0.395% | 1.72% | `CD10` 4.2% |
| arcsinh u16, HTJ2K level 0.002 | 155.5 MB | **0.90×** | 1.68% | 7.61% | `CD10` 18.4% |
| arcsinh u16, HTJ2K level 0.005 | 118.6 MB | 1.18× | 4.55% | 20.5% | `CD10` 45.4% |
| arcsinh u16, HTJ2K level 0.02 | 62.3 MB | 2.25× | 20.0% | 68.9% | `CD10` 97.7% |

**Lossy HTJ2K does not overtake *lossless* zstd until the median error is already ~3%, and the p90
channel is then ~15%.** That is an unusual result and it is the sparsity again: the wavelet spends
its bit budget on discontinuities, and these channels are nearly all discontinuity.

## Finding 7 — the right answer is per channel, and it is mostly not HTJ2K

The stack-wide numbers hide a composition effect: the six dense channels do brilliantly under lossy
(`80ArAr` 16.70 MB → 1.33 MB at 0.39% error) while forty-odd sparse ones do terribly. So the fair
test is an **oracle**: let every channel pick the smallest encoding meeting an error budget. Nothing
per-channel can beat this, so if it does not win, nothing will.

| error budget | stack total | vs zstd-9 | what the 49 channels pick |
|---|---|---|---|
| ≤ 0.5% | 84.9 MB | 1.65× | **38 zstd**, 7 HTJ2K lossy, 4 HTJ2K u8 |
| **≤ 1%** | **79.6 MB** | **1.76×** | **36 zstd**, 13 HTJ2K lossy |
| ≤ 2% | 73.4 MB | 1.91× | **35 zstd**, 14 HTJ2K lossy |
| ≤ 5% | 63.9 MB | 2.19× | **34 zstd**, 15 HTJ2K lossy |
| ≤ 10% | 52.1 MB | 2.69× | **29 zstd**, 20 HTJ2K lossy |

So lossy *is* worth having — 79.6 MB against 197.5 MB today is 2.5×, at ≤1% per-cell error — but the
win comes from applying it to a **minority** of channels. At every budget up to 10%, most channels
still prefer lossless generic compression. There is no error budget at which HTJ2K wins the stack.

**The structural catch:** zarr assigns one codec per *array*, so per-channel codecs mean splitting the
`c` axis across two image elements (one lossless, one lossy) sharing a coordinate system. Legal, and
awkward for a viewer that expects one 49-channel stack. The 1.76× is the price of that awkwardness —
worth knowing before paying it.

## Finding 8 — chunking channels together buys nothing here

Worth testing, because the obvious way to exploit a 49-channel stack is to let the coder see several
channels at once, and the tool's `--chunks auto` gives one channel per chunk.

The prior that motivated it turned out to be **false**: off-tissue background is *not* zero in every
channel. Four channels (`80ArAr`, `127I`, `131Xe`, `134Xe`) have no zeros at all — they carry a
non-zero floor everywhere — so there is no shared sparsity pattern to exploit. (The zero-mask
correlation comes back `NaN` for exactly that reason, which is how the assumption got caught.)

Measured inter-channel correlation is essentially nil: **median 0.016**, p90 0.093. The one real pair
is `DNA1`/`DNA3` at r=0.981, which is two DNA channels of the same stain and so not news. Everything
else is below 0.62.

| channels per chunk | file order | correlation-sorted order |
|---|---|---|
| 1 | 140.0 MB | 140.0 MB |
| 4 | 140.1 MB | 140.0 MB |
| 49 | 140.1 MB | 140.0 MB |

Flat to three digits, including after reordering channels so the correlated ones are adjacent, and
HTJ2K encoding eight channels as one multi-component codestream matches eight separate ones to the
same precision.

That table alone was not proof, though: **Blosc splits its input into independent blocks**, so
cross-channel redundancy could have been invisible by construction rather than absent. Repeating it
with plain (unblocked) zstd, which sees one window over the whole buffer:

| channels per chunk | plain zstd-9 | plain zstd-19 |
|---|---|---|
| 1 | 157.9 MB | 131.1 MB |
| 4 | 158.0 MB | 131.1 MB |
| 49 | 158.0 MB | 131.1 MB |

Flat there too, so the absence is real. **Keep one channel per chunk** — it costs nothing and it is
what preserves single-channel random access in a browser. If a future panel *does* have correlated
channels the measurement is cheap to redo; on this one there is nothing to collect.

## Finding 9 — plain zstd at a high level beats Blosc-zstd, which the confound check turned up

Incidental to finding 8, and it changes the recommended codec. Blosc's `clevel` is not a zstd level —
it maps onto one internally, and for zstd it lands around 16–18. So `clevel=9` is *not* the top of the
range, and going past it needs the plain zstd codec rather than Blosc:

| whole stack, per ROI | size | vs Blosc-9 | encode (extrapolated, 49 ch) |
|---|---|---|---|
| Blosc zstd `clevel=9` | 140.0 MB | 1.00× | ~30 s |
| plain zstd level 12 | — | 0.945× | ~23 s |
| plain zstd level 15 | — | 0.962× | ~83 s |
| **plain zstd level 19** | **131.1 MB** | **1.068×** | ~280 s |
| plain zstd level 22 | — | 1.058× | ~285 s |

Level 19 is the knee — 22 costs the same time for 0.2% less. Note levels 12 and 15 are *worse* than
Blosc's `clevel=9`, which is what gives away where Blosc's mapping sits. ~5 min/ROI puts a full
conversion at ~2.5 h, one-time, which is fine.

Plain zstd is also the more standard choice: it is a **core codec in the zarr v3 spec**, where Blosc
is an extension.

## Recommendation

**Decided 2026-08-01 (user's call): one array, float32, plain zstd level 19, one channel per chunk.**
131 MB/ROI, **~3.7 GB against 6.0 GB today** (1.6×), lossless, no codec extension, a core zarr v3
codec, and the stack stays a single 49-channel element that a viewer can treat normally.

*(The decision was taken against Blosc-zstd at 140 MB/ROI; finding 9 then found plain zstd-19 is 6.8%
better and more standard. Same shape of answer, better constant — no re-decision needed. **Do not use
byte-shuffle**, whichever codec: finding 1.)*

This deliberately declines the smaller option below — taking the hit on size rather than splitting
the `c` axis by which codec suits each channel.

<details>
<summary>The per-channel split, not taken (~2.3 GB)</summary>

| | channels | codec | per ROI | whole dataset |
|---|---|---|---|---|
| sparse / spiky | ~36 | float32, zstd-9 **shuffle off** | ~66 MB | ~1.9 GB |
| dense | ~13 | arcsinh(x/5) → uint16, HTJ2K lossy | ~14 MB | ~0.4 GB |

~80 MB/ROI at ≤1% per-cell error on the lossy channels and exact on the rest — 1.76× smaller than the
decision above. Rejected because zarr binds one codec per *array*, so it means splitting one
49-channel stack into two image elements purely as a storage artefact. Channel assignment would also
have to be measured per ROI rather than fixed from one file.

*(This also supersedes an earlier draft that split by **job** — a lossless source plus a separate
uint8 viewing layer, 270 MB/ROI. Finding 7 does that work better, because the channels tolerating
lossy are precisely the ones a viewing layer existed for.)*

</details>

**The interesting part is upstream, not here.** What the rejected option really wants is *per-channel
codec selection within one array* — a thing zarr cannot express and OME-Zarr therefore cannot either.
The measurements say the payoff is real (1.76× on this dataset, and it would grow with panel size,
since sparse and dense channels coexist in every IMC panel). That is worth raising alongside
`docs/proposals/instance-views.md` rather than working around locally; sharding in zarr v3 gets
close on layout but still binds one codec per array.

Two caveats to carry:

- **`CD10` breaks every lossy configuration** — 45% block error at level 0.005 and *larger* than its
  LZW baseline. Sparse-plus-huge-outlier is the shape to watch for; it should land in the lossless
  group and the budget check is what puts it there.
- Anything lossy must be named so it cannot be mistaken for the source — a `:lossy` suffix or a
  separate element name, never a resolution level of the same element.

## Would float sample support change any of this? No.

Worth settling because it is the obvious "what if the dtype gate just went away" question, and
because a first pass at this doc got the format facts wrong.

**HTJ2K really can carry floats, and it stays HTJ2K.** The
[HTJ2K white paper](https://htj2k.com/wp-content/uploads/white-paper.pdf) is explicit — "HTJ2K can
be used to losslessly encode half-float or even full IEEE floating point representations of scene
radiance" (p9) — via the custom floating-point mappings defined in **JPEG 2000 Part-2** (p3). So this
is a standardised mapping, not a private extension, and a Part-2-aware decoder reads it. Any claim
that float support would cost interoperability is wrong.

**But the same paper measures it, and the number is the answer.** Its §7.5.3 benchmark encodes ACES
OpenEXR 16-bit half-float — smooth, dense, professionally graded HDR film frames, about the friendliest
possible float data for a wavelet:

| lossless, half-float | avg ratio | median ratio | decode fps |
|---|---|---|---|
| HTJ2K Part-15 | 1.462 | 1.529 | 38 |
| J2K Part-1 | 1.501 | 1.573 | 1.4 |
| **OpenEXR ZIP** (plain zlib) | **1.454** | 1.450 | 20 |

On its own showcase data, lossless float HTJ2K ties with **zlib** on ratio. What it wins is decode
throughput — 27× over Part-1, and nearly 2× over ZIP. That is a real and valuable thing, and it is
not compression.

Ours is the harder case (sparse and spiky, where zstd gets 5.6× precisely because a dictionary coder
eats runs of zeros), so lossless float HTJ2K would land further behind, not closer. And finding 7
closes it from the other side: the 13 channels that do want HTJ2K want it **lossy**, at a codec error
of 1–5% — hundreds of times larger than the arcsinh 16-bit quantisation it would replace. Float
samples would remove a rounding error that is already invisible under the error we are choosing to
accept.

So float support is a genuine format capability, correctly described by the paper, that this dataset
has no use for.

## Assessment of `spatialdata-js-util`

Genuinely useful, and the parts of it that are not about presets are the parts worth having:

- **Use it for the H&E** — this is exactly its case, and `--pyramid` matters more than the codec
  choice, since nothing here is pyramidal and a zoomed-out view otherwise pulls full-resolution
  chunks.
- **Use `--sibling`** for the viewing layer. It already implements the keep-the-original-and-add-a-
  compressed-copy shape that the recommendation above needs.
- **`--chunks auto`** resolves to `(1, …, 1024, 1024)` (images.py:197), so channels are chunked one
  per chunk and a single channel is fetchable. Right for a 49-channel stack.
- The **backend probe** is the right instinct — it refuses to trust `imagecodecs` on reputation and
  gates it on decoding a committed multi-component fixture, because the failure mode it guards
  against (silently decoding every component as component 0) is invisible.
- **Do not use it for the float32 stack** without deciding the quantisation deliberately; the dtype
  gate will stop you, which is the correct behaviour.
- **Do not use `--preset balanced` or `--preset small` on ≤8-bit data** until finding 3 is resolved.

Worth reporting upstream, in order: the preset crossover (finding 3) is a concrete defect with a
one-line reproduction; the "never pre-quantise for a lossy encode" result (finding 3, second half) is
worth a note in the README beside the existing `--quality` warning; and a `--dry-run` that reports
predicted sizes against the input would have surfaced all of this without anyone measuring anything.
