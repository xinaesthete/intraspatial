# Declaring that several elements describe the same instances

**Draft proposal for discussion — SpatialData / scverse.**
Status: request for comment. Nothing implemented; no PR attached.

## Summary

A SpatialData table declares the element it annotates through `region`, `region_key` and
`instance_key`. Each row of a table therefore resolves to exactly one `(element, instance)` pair.

But a single set of biological instances is routinely represented by **several elements at once** —
a label mask, a boundary polygon, a centroid circle, a nucleus boundary. The format has no way to
say that those elements describe the same instances, so the correspondence lives in consumer code
as a private convention.

We would like to propose a small, ignorable piece of metadata that states it explicitly.

## The problem is already in a standard `spatialdata-io` output

This is not an edge case from an unusual dataset. A Xenium run read through `spatialdata-io` produces
five elements over the same cells:

| element | kind | instance identity | rows |
|---|---|---|---|
| `shapes/cell_circles` | shapes | `cell_id` **column** | 36 |
| `shapes/cell_boundaries` | shapes | parquet **index** (`__index_level_0__`) | 36 |
| `shapes/nucleus_boundaries` | shapes | parquet **index** | 23 |
| `labels/cell_labels` | labels | pixel value | — |
| `labels/nucleus_labels` | labels | pixel value | — |

The table's `region` is `"cell_circles"`. The other four hold the same cells, and **nothing in the
store records that**. A consumer wanting to shade `cell_boundaries` by a column of the table has to
already know that Xenium writes the cell id into the parquet index of that element — knowledge that
comes from the reader's source, not from the data.

Two details from that same store are worth carrying into the design, because they rule out the
obvious naive version of this proposal:

- **The identity is expressed in more than one way within a single store** — a named `cell_id` column
  in one element, an unnamed index in two others. So an assertion has to be about instance *values*,
  not about where they are stored.
- **Coverage is not always complete.** `nucleus_boundaries` has 23 rows against the table's 36: not
  every cell has a detected nucleus. So the relation is "these elements draw their instances from the
  same space", **not** "these elements contain the same instances". A spec that assumes a bijection
  would be wrong on the very first real dataset.

*(Observations taken from a store written by SpatialData 0.4.0, format version 0.1, with
`image-label` version `0.4-dev-spatialdata`. Element names are `spatialdata-io`'s Xenium reader
output. We would welcome correction if any of this is already expressible and we have missed it.)*

## Why `region` / `region_key` does not already cover it

`region` may name several elements, and `region_key` names the `obs` column that says which element
each row belongs to. That mechanism **partitions rows across elements** — row *i* belongs to element
A, row *j* to element B.

What is missing is the orthogonal case: one row, several elements. `region_key` and `instance_key`
are each a single `obs` column, so a row yields one element name and one instance value. There is no
position in the model where a second representation of the same row could be declared.

Both are useful and they do not overlap. Partitioning is "these rows are about different things";
what we are asking for is "these elements are different views of the same thing".

## Consequences today

- Every consumer re-implements the same convention privately, and the conventions differ. A viewer
  that gets it wrong fails silently — the geometry renders, it is simply joined to the wrong data,
  or to nothing.
- Alternative geometries are effectively second-class: they can be drawn but not coloured by
  anything in the table, unless the consumer hard-codes the relationship.
- Round-tripping through a generic tool can drop the association entirely, because there was never
  anything written down to preserve.
- Workarounds are all lossy in some way. Duplicating the table per element is correct by the spec but
  duplicates `obs` (and `X`, unless one copy goes without) into two structures that can drift.

## Proposal

Store-level metadata, alongside `spatialdata_attrs` in the root attributes, asserting that a set of
elements draws instances from one shared space:

```json
"instance_views": {
  "version": "0.1-draft",
  "groups": [
    {
      "instance_key": "cell_id",
      "primary": "shapes/cell_circles",
      "members": [
        { "element": "shapes/cell_circles",        "role": "centroid" },
        { "element": "shapes/cell_boundaries",     "role": "cell_boundary" },
        { "element": "shapes/nucleus_boundaries",  "role": "nucleus_boundary", "coverage": "partial" },
        { "element": "labels/cell_labels",         "role": "cell_mask" },
        { "element": "labels/nucleus_labels",      "role": "nucleus_mask",     "coverage": "partial" }
      ]
    }
  ]
}
```

- `instance_key` — the name of the shared identity, matching the annotating table's `instance_key`.
- `primary` — the element the table's `region` already names. **Must** be present and **must** agree
  with the existing declaration.
- `members[].element` — path-qualified element name, so `shapes/x` and `labels/x` are distinguishable.
- `members[].role` — a **free-text hint**, deliberately not an enumeration (see non-goals).
- `members[].coverage` — `"full"` (default) or `"partial"`, for the `nucleus_boundaries` case.
- `groups` is a list: a dataset with one region per section needs one group per section.

It lives at the root rather than in a table's `uns` because the relation holds between elements and
may be referenced by more than one table.

### Three properties that are the whole design

1. **Ignorable.** A reader that has never seen the key must open the store and behave exactly as it
   does today. That is what `primary` is for — the extension only *adds* aliases and never redirects
   an existing declaration. The acceptance test is that an unmodified `spatialdata.read_zarr` gives
   an identical object.
2. **Checkable.** An assertion of shared identity that is false is worse than no assertion. A
   validator should sample real values — label values against the table's instance column, parquet
   index against the same — rather than trusting the declaration. Under `coverage: "partial"` the
   check is subset, not equality.
3. **Additive only.** No existing field changes meaning, and no element gains a required attribute.

## Non-goals

Kept deliberately narrow, because the wider versions are where a proposal like this stalls.

- **Not a role ontology.** `role` is a hint for a UI to label a layer. Standardising the vocabulary
  of biological relationships (nucleus vs. membrane vs. expanded vs. centroid) is a much larger
  discussion and is not needed to fix the join.
- **Not a geometry conversion contract.** Nothing here says a boundary can be derived from a mask, or
  that they agree geometrically. Only that they are indexed by the same instances.
- **Not a change to `region` / `region_key` semantics.** Partitioning and aliasing stay separate.
- **Not multi-table.** Which table a consumer joins from is unchanged and remains the table's own
  business.

## Open questions

1. **Root attributes or a per-element attribute?** The reverse arrangement — each element pointing at
   its primary — is more local and survives an element being copied between stores, but makes it
   costly to enumerate a group and easy for two elements to disagree.
2. **Should `primary` be required?** Requiring it guarantees graceful degradation, but forbids
   declaring a group with no annotating table at all, which is otherwise a legal and useful state.
3. **Is `coverage` enough**, or does partial coverage need to name *which* instances are present?
   Our reading is that the element already answers that and duplicating it would only create a second
   thing to keep in sync.
4. **Naming.** `instance_views` is a placeholder. While this is a draft it should carry a namespace
   prefix; there is precedent for namespaced values in the store already
   (`spatialdata-encoding-type: "ngff:regions_table"`).

## What we would contribute

We are building a browser-side reader for SpatialData zarr and hit this while modelling a public IMC
dataset that has the same shape (a segmentation mask, and centroids, over one cell table). We are
happy to:

- implement the draft in our reader and report what breaks;
- write the validator described above, including the value-sampling check;
- test against `spatialdata-io` Xenium output, since it is the case that motivated this.

Our own dataset is a secondary motivation. The argument we would make is the first table in this
document: a standard 10x export, read by the official reader, cannot express what it plainly
contains.

## How to verify the observations

Everything above came from inspecting a store on disk rather than from reading the spec:

```bash
# The five elements, and which one the table annotates
ls spatialdata.zarr/shapes spatialdata.zarr/labels
python -c "import json; print(json.load(open('spatialdata.zarr/tables/table/.zattrs'))['region'])"

# Row counts: cell_circles 36, cell_boundaries 36, nucleus_boundaries 23
python -c "import pyarrow.parquet as pq; \
  [print(e, pq.read_metadata(f'spatialdata.zarr/shapes/{e}/shapes.parquet').num_rows) \
   for e in ('cell_circles','cell_boundaries','nucleus_boundaries')]"

# Where the instance id lives, per element
python -c "import pyarrow.parquet as pq; \
  [print(e, pq.read_schema(f'spatialdata.zarr/shapes/{e}/shapes.parquet').names) \
   for e in ('cell_circles','cell_boundaries','nucleus_boundaries')]"
```
