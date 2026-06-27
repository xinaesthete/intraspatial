# HT block decoder — port plan (Phase 2)

Faithful Rust port of OpenJPH's HT **cleanup-pass** decoder
(`src/core/coding/ojph_block_decoder32.cpp` + table generation in
`ojph_block_common.cpp` / `table0.h` / `table1.h`), BSD-2-Clause.

## Scope for v1 (validated)

Probing real OpenJPH lossless output (`parse_packets_summary.max_num_passes`)
shows **every code-block is cleanup-pass only (`num_passes == 1`)**. So v1 needs
only the cleanup pass — **not** SigProp/MagRef. `lengths2 == 0`.

Decoder entry (from OpenJPH):
```
ojph_decode_codeblock32(coded_data, decoded_data, missing_msbs, num_passes,
                        lengths1, lengths2, width, height, stride, stripe_causal)
```
All inputs are now produced by the parser per code-block: `offset`,
`length_cleanup` (= `lengths1`), `length_refinement` (= `lengths2`),
`missing_msbs`, `num_passes`. (`packet::CodeBlock`).

## Validation strategy

A **0-decomposition** codestream has no DWT, so resolution 0's single LL subband
*is* the image, tiled into code-blocks. The decoded code-block coefficients are
therefore the (DC-level-shifted) pixels. Validate end-to-end:
encode pixels with `openjph-wasm` (`decompositions: 0`, reversible) → parse →
decode each code-block → reassemble → undo level shift → compare to the original
pixels (bit-exact). No DWT or intermediate reference needed.

## Algorithm structure (three sub-readers + two steps)

1. **MEL reader** (`dec_mel_st`): adaptive run code, MSB-first, unstuffing removes
   the MSB of the byte after a 0xFF. Decodes runs of zero-events. `mel_get_run`.
2. **Reverse VLC reader** (`rev_struct`): backward-growing, LSB-first; unstuff when
   the (higher-address) previous byte > 0x8F and current is 0x7F. `rev_init`
   discards the first 12 bits (the MEL+VLC length / `scup`). `rev_fetch`/`rev_advance`.
3. **Forward MagSgn reader** (`frwd_struct32`): forward, LSB-first; unstuff after
   0xFF; feeds 0xFF past end (`X = 0xFF`). `frwd_fetch`/`frwd_advance`.

`scup = (coded[lcup-1] << 4) + (coded[lcup-2] & 0xF)` (length of MEL+VLC).
`p = 30 - missing_msbs` (least-significant bitplane).

**Step 1 — VLC + MEL → scratch (2 bytes/quad):** scan quads in pairs across each
quad-row. Per quad pair use `vlc_tbl0` (initial row) / `vlc_tbl1` (others) keyed by
context `c_q` + 7 VLC bits → `rho`, `e_k`, `e_1`, code length. Zero-context quads
consume a MEL run event. `uvlc_tbl0`/`uvlc_tbl1` decode `u_q` (magnitude exponents)
per quad pair. Store `inf` (rho/e_k/e_1) and `U_q` per quad. Context for next quad:
`c_q = ((t & 0x10)<<3) | ((t & 0xE0)<<2)`; for non-initial rows OR in north
neighbours' sigma.

**Step 2 — MagSgn → coefficients:** for each quad sample, if significant
(`inf & (1<<(4+bit))`), fetch `m_n = U_q - e_k` bits from MagSgn; build
`v_n = bits | (e_1 << m_n) | 1`; `val = (sign<<31) | ((v_n + 2) << (p-1))`.
For non-initial rows, `U_q = u_q + kappa`, `kappa = gamma ? emax : 1`, where
`emax` comes from the previous row's `v_n` values (the `v_n_scratch` row).

Output `decoded_data[i]` is sign-magnitude: bit 31 = sign, magnitude in low bits.
The reversible coefficient is recovered from the magnitude (see how
`ojph_subband`/`ojph_codeblock` consume `decoded_data`); finish by inverting the
DC level shift for unsigned components.

## Implementation notes

- **Byte-at-a-time readers are sufficient.** OpenJPH's word reads + `intptr_t & 3`
  alignment dance are an optimization; reading byte-by-byte with identical
  per-byte unstuffing yields the same bit order. Simpler and safe in Rust.
- **Padding.** Backward (rev) and forward (frwd) readers over-read by several
  bytes. Copy the code-block bytes into a buffer with leading/trailing zero
  padding (OpenJPH uses prefix/suffix bufs) so all reads stay in bounds.
- **Tables.** Port `vlc_tbl0/1` (1024 × u16) and `uvlc_tbl0/1` from the init code
  in `ojph_block_common.cpp` (which expands `tbl0`/`tbl1` from `table0.h`/
  `table1.h`). Generate once at module load (lazy `OnceLock`).
- Defer SigProp/MagRef (`num_passes > 1`) and 9/7 dequant to later phases.

## Next steps

1. Port the VLC/UVLC table generation → `block_tables.rs` (mechanical).
2. Port the three readers → `block_decoder.rs` (byte-at-a-time).
3. Port step 1 + step 2 (cleanup only).
4. Wire parser → decoder; validate bit-exact on 0-decomposition fixtures.
5. Then Phase 3: GPU inverse 5/3 DWT, and revisit multi-pass / 9/7.
