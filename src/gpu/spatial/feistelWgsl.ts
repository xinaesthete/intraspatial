// A format-preserving Feistel permutation of [0, n), shared by the GPU permutation nulls
// (`quadratCorrelationGpu`, `anchorScreenGpu`).
//
// π(i) is computed independently per thread in O(1), with no scratch and no sort, so a shuffle
// becomes one thread per element instead of a sequential Fisher–Yates with an n-element buffer per
// shuffle. Six rounds of a balanced Feistel network over the two halves of the index make a
// bijection on [0, 2^(2·halfBits)) for ANY round function; cycle-walking (re-encrypting until the
// image lands below n) folds that onto [0, n) and keeps it a bijection, because each out-of-range
// point is walked along its own closed cycle. The domain is under 4n, so the expected number of laps
// is below 4.
//
// It is a *pseudo-random* permutation where the CPU nulls draw a uniform one, so GPU and CPU nulls
// agree statistically, never draw for draw. `feistelPermuteCpu` is the exact host twin, for tests.
//
// The snippet is self-contained: every parameter is an argument, so a host shader declares no
// uniform names for it. It defines `mix32`, `feistel`, `feistelPermute` and `feistelKey`.

export const FEISTEL_WGSL = /* wgsl */ `
fn mix32(x: u32) -> u32 {
  var v = x;
  v = v ^ (v >> 16u); v = v * 0x7feb352du;
  v = v ^ (v >> 15u); v = v * 0x846ca68bu;
  v = v ^ (v >> 16u);
  return v;
}

// One pass of a balanced Feistel network on 2*halfBits bits. Invertible for ANY round function,
// which is what makes the composition a permutation rather than a hash.
fn feistel(idx: u32, key: u32, halfBits: u32, halfMask: u32) -> u32 {
  var l = idx & halfMask;
  var r = (idx >> halfBits) & halfMask;
  for (var round = 0u; round < 6u; round = round + 1u) {
    let f = mix32(r ^ key ^ (round * 0x9e3779b9u)) & halfMask;
    let nl = r;
    r = l ^ f;
    l = nl;
  }
  return (r << halfBits) | l;
}

// Cycle-walking onto [0, n). The guard only bounds a loop the domain size already bounds.
fn feistelPermute(i: u32, key: u32, n: u32, halfBits: u32, halfMask: u32) -> u32 {
  var x = i;
  for (var guard = 0u; guard < 64u; guard = guard + 1u) {
    x = feistel(x, key, halfBits, halfMask);
    if (x < n) { return x; }
  }
  return i;
}

// The key of shuffle s of a run seeded with seedBase.
fn feistelKey(seedBase: u32, s: u32) -> u32 { return mix32(seedBase + s * 0x9e3779b9u + 1u); }
`;

/** The Feistel domain for [0, n): the smallest balanced power of two that covers n. */
export function feistelDomain(n: number): { halfBits: number; halfMask: number } {
  const bits = Math.max(2, Math.ceil(Math.log2(Math.max(n, 2))));
  const halfBits = Math.ceil(bits / 2);
  return { halfBits, halfMask: (1 << halfBits) - 1 };
}

function mix32(x: number): number {
  let v = x >>> 0;
  v = Math.imul(v ^ (v >>> 16), 0x7feb352d) >>> 0;
  v = Math.imul(v ^ (v >>> 15), 0x846ca68b) >>> 0;
  return (v ^ (v >>> 16)) >>> 0;
}

/** Host twin of WGSL `feistelKey`. */
export function feistelKeyCpu(seedBase: number, s: number): number {
  return mix32((seedBase + Math.imul(s, 0x9e3779b9) + 1) >>> 0);
}

/** Host twin of WGSL `feistelPermute`: the same bits, for tests. */
export function feistelPermuteCpu(i: number, key: number, n: number, halfBits: number, halfMask: number): number {
  let x = i >>> 0;
  for (let guard = 0; guard < 64; guard++) {
    let l = x & halfMask;
    let r = (x >>> halfBits) & halfMask;
    for (let round = 0; round < 6; round++) {
      const f = mix32((r ^ key ^ Math.imul(round, 0x9e3779b9)) >>> 0) & halfMask;
      const nl = r;
      r = l ^ f;
      l = nl;
    }
    x = ((r << halfBits) | l) >>> 0;
    if (x < n) return x;
  }
  return i;
}
