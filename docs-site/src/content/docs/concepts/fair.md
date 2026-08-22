---
title: FAIR by design
description: How the toolbox's design serves the FAIR principles — Findable, Accessible, Interoperable, Reusable.
---

The [FAIR principles](https://www.go-fair.org/fair-principles/) — **F**indable,
**A**ccessible, **I**nteroperable, **R**eusable — are the standard for research software
and data stewardship. They aren't a bolt-on here: the same choices that make the toolbox
[composable and interpretable](/concepts/composable-interpretable/) are what make it FAIR.
This page is an honest map of where the design already delivers and where the work
remains.

## Findable

- **A self-describing catalogue.** Every op and source carries a name, label,
  description, category, and rendered maths (its `help`). The palette is a findable index
  of the operations, not a wall of undocumented functions — and the same metadata drives
  the docs.
- **Stable identities for results.** The executor is content-addressed: a node's output
  is keyed by its operation and the identity of its inputs, so an intermediate value has a
  stable, referenceable identity within a run.
- *Still to do:* persistent identifiers (DOIs/accessions) for shared datasets and saved
  graphs.

## Accessible

- **Open runtime, no gatekeeping.** It runs in any WebGPU-capable browser over the
  standard `navigator.gpu` — no proprietary engine, no install, no account. Retrievable
  over the open web with an open protocol.
- **Open standard formats.** Imagery uses HTJ2K / JPEG 2000 (Part 15); tabular and
  spatial data stay as plain typed arrays and columnar buffers. Nothing is trapped in a
  bespoke binary.
- **Graphs are plain data.** A composed graph serialises to plain JSON (the canvas state
  ↔ the runtime `Graph` IR), so it can be stored, diffed, and shared like any artefact.

## Interoperable

- **Plain arrays in, plain arrays out.** Primitives take and return typed arrays with no
  framework object to construct — the [composable](/concepts/composable-interpretable/)
  contract. That's what lets the kernels port to deck.gl / [MDV](https://mdv.molbiol.ox.ac.uk/) /
  SpatialData.js rather than locking analysis inside this tool.
- **An explicit, shared schema.** The field-type model gives every value a stated *shape*
  (grid, points, …), *element* algebra (scalar, complex, vector, …), and *basis* (spatial
  vs wavelet). Ops declare their contracts, so a value's meaning travels with it instead
  of living in a comment.
- **One algebra, many scales.** The same operations express the analysis whether you run
  them interactively in the browser or, in principle, as an equivalent batch/cluster job —
  the interactive exploration and the production run speak the same language.

## Reusable

- **The graph *is* the provenance.** A declarative operation graph records exactly how a
  result was produced — every source, parameter, and step. Re-running it reproduces the
  result; sharing it shares the method, not just the output.
- **Reproducible by construction.** Randomised steps draw from seeded GPU RNGs (no
  `Math.random`), and every primitive ships a CPU golden with a GPU test asserting
  bit-exact (integer) or bounded-error (float) agreement — so a reused result is a
  *trustworthy* result.
- **Reuse is a first-class mechanism.** Named subgraphs are defined once and instantiated
  many times (live-linked, so editing the definition updates every use). The decision
  records ([ADRs](https://github.com/xinaesthete/intraspatial/tree/main/docs/decisions))
  capture the rationale behind the contracts so future users inherit the *why*.
- *Still to do:* a clear, explicit data/usage licence — the remaining piece of the
  Reusable pillar.

## Why it matters here

This is analysis tooling for scientific data — spatial biology and imaging among the
targets. A result is only useful to a collaborator if they can find the operation that
made it, run it without a bespoke stack, feed it data from their own pipeline, and trust
that re-running reproduces the figure. FAIR is simply the name for getting all four of
those right at once, and the toolbox is built so they come for free rather than as an
afterthought.
