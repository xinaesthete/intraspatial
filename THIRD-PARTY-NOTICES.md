# Third-party notices

Most of this project is licensed under the MIT License (see [`LICENSE`](LICENSE)).

The portions listed below are **derived from OpenJPH**
(https://github.com/aous72/OpenJPH) and are licensed under the **BSD 2-Clause
License**, not MIT. As required by that license, the OpenJPH copyright notice and
license text are retained here. These portions remain BSD-2-Clause when
redistributed or repackaged.

## Affected code

- **`rust/htj2k-core/`** — the HTJ2K codestream parser, HT cleanup-pass block
  decoder, and inverse 5/3 & 9/7 DWT are a faithful Rust port of OpenJPH
  (`ojph_decode_codeblock32`, `precinct::parse`, `gen_rev_horz_syn`,
  `resolution::pull_line`, and related). The module headers say so, and
  `rust/htj2k-core/Cargo.toml` already declares `license = "BSD-2-Clause"`.
- **`src/gpu/fdwt53.ts`, `src/gpu/fdwt97.ts`, `src/gpu/idwt53.ts`,
  `src/gpu/idwt97.ts`** (and the generated `src/gpu/idwt53.gen.ts`,
  `src/gpu/idwt97.gen.ts`) — TypeGPU/WebGPU DWT kernels ported from the OpenJPH
  algorithms above, bit-exact against the Rust reference.

Clean-room work built only from published specifications or mathematics — the
op-graph runtime, spatial-analysis ops, geometry, evo, sims, datasource, and
colour code — is original and MIT-licensed.

## OpenJPH — BSD 2-Clause License

Copyright (c) 2019, Aous Naman
Copyright (c) 2019, Kakadu Software Pty Ltd, Australia
Copyright (c) 2019, The University of New South Wales, Australia

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
