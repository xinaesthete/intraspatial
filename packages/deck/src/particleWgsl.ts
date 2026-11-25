// The decoding both particle layers and their tests share: which particles and bonds are drawn.
// Plain WGSL with no deck dependency, so a compute pass can run it headless.

/** An integer attribute stored as a u32 word, or as an f32 holding a whole number. */
export const CATEGORY_WGSL = /* wgsl */ `
fn particle_category(word: u32, isFloat: bool) -> u32 {
  return select(word, u32(bitcast<f32>(word)), isFloat);
}
`;

/**
 * A bond-slot table: `K` slots per particle, word 0 of each slot being `partner | kind << 24`, or
 * all ones when empty. A bond sits in both ends' rows; it is drawn from the lower-numbered end
 * only, so once.
 */
export const BOND_WGSL = /* wgsl */ `
const BOND_EMPTY: u32 = 0xFFFFFFFFu;

struct BondEnds {
  a: u32,
  b: u32,
  kind: u32,
  draw: bool,
};

fn bond_ends(word: u32, slot: u32, K: u32) -> BondEnds {
  var e: BondEnds;
  e.a = slot / K;
  e.b = word & 0xFFFFFFu;
  e.kind = word >> 24u;
  e.draw = word != BOND_EMPTY && e.a < e.b;
  return e;
}
`;
