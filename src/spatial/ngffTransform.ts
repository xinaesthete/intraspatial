// NGFF / SpatialData coordinate transformations and physical units — reading the 2-D spatial
// mapping and its unit out of the metadata, rather than assuming either.
//
// This matters for the cell statistics because every parameter they take is a LENGTH. The paper's
// TCM uses a 100 µm neighbourhood radius and a 50 µm bandwidth; entering "100" against a store
// whose coordinates are pixels is a silent factor-of-nothing error that changes the answer and
// looks fine. So the unit has to come from the data where the data states it — and, where it does
// not, has to be visibly UNSTATED rather than quietly assumed.
//
// What the spec actually gives us:
//   • a transform has an `input`/`output` pair naming coordinate systems and listing `axes`, each
//     with a `name` ("x", "y", "c", …), a `type` ("space", "channel", …) and a `unit`;
//   • `type` is one of identity / scale / translation / affine / sequence (the ones that occur for
//     2-D elements; `byDimension` and the non-linear types are out of scope and reported as such);
//   • the unit is a UDUNITS-2 name — "micrometer", "millimeter", … — and SpatialData writes the
//     literal placeholder **"unit"** when nothing was specified, which must NOT be mistaken for a
//     real unit.
//
// Kept dependency-free in `src/` so it is covered by the CPU suite; the playground supplies the
// already-parsed attribute objects.

/** One axis of an NGFF coordinate system. */
export interface NgffAxis {
  readonly name?: string;
  readonly type?: string;
  readonly unit?: string;
}

/** A 2-D affine: `world = [[a c],[b d]] · array + [tx, ty]`. Rotation and shear are representable,
 *  so a rotated element is not silently flattened to a scale. */
export interface Affine2 {
  readonly a: number;
  readonly b: number;
  readonly c: number;
  readonly d: number;
  readonly tx: number;
  readonly ty: number;
}

export const IDENTITY2: Affine2 = { a: 1, b: 0, c: 0, d: 1, tx: 0, ty: 0 };

/** `second ∘ first` — apply `first`, then `second`. */
export function compose2(first: Affine2, second: Affine2): Affine2 {
  return {
    a: second.a * first.a + second.c * first.b,
    b: second.b * first.a + second.d * first.b,
    c: second.a * first.c + second.c * first.d,
    d: second.b * first.c + second.d * first.d,
    tx: second.a * first.tx + second.c * first.ty + second.tx,
    ty: second.b * first.tx + second.d * first.ty + second.ty,
  };
}

/** Micrometres per unit of the named UDUNITS-2 unit, or `undefined` when the unit is unknown, not a
 *  length, or SpatialData's "unit" placeholder.
 *
 *  Returning `undefined` rather than defaulting to 1 is the whole point: "we do not know the scale"
 *  and "the scale is one micrometre" are different states, and conflating them is how a length gets
 *  reported in the wrong unit with full confidence. */
export function micrometresPer(unit: string | undefined): number | undefined {
  if (!unit) return undefined;
  const u = unit.trim().toLowerCase();
  switch (u) {
    case "micrometer":
    case "micrometre":
    case "micron":
    case "um":
    case "µm":
      return 1;
    case "nanometer":
    case "nanometre":
    case "nm":
      return 1e-3;
    case "millimeter":
    case "millimetre":
    case "mm":
      return 1e3;
    case "centimeter":
    case "centimetre":
    case "cm":
      return 1e4;
    case "meter":
    case "metre":
    case "m":
      return 1e6;
    case "angstrom":
      return 1e-4;
    // "pixel" is a real NGFF unit but not a physical length; "unit" is SpatialData's placeholder
    // for "unspecified". Both mean: the store has not told us the scale.
    default:
      return undefined;
  }
}

export interface ResolvedSpace {
  /** Array/element space → the target coordinate system, in x/y. */
  readonly affine: Affine2;
  /** The output system's name, when the metadata gave one. */
  readonly system?: string;
  /** The output x/y unit VERBATIM, e.g. "micrometer" or SpatialData's "unit". */
  readonly unit?: string;
  /** Micrometres per world unit, when the unit names a length. */
  readonly micrometres?: number;
  /** Transform types encountered that this resolver does not implement. Non-empty means `affine`
   *  is an approximation and should be reported as such rather than trusted. */
  readonly unsupported: string[];
}

interface Ctx {
  axes: readonly NgffAxis[];
  unsupported: string[];
}

const asRecord = (v: unknown): Record<string, unknown> | undefined =>
  v && typeof v === "object" && !Array.isArray(v) ? (v as Record<string, unknown>) : undefined;
const asArray = (v: unknown): unknown[] | undefined => (Array.isArray(v) ? v : undefined);

/** Index of the axis named `want` (case-insensitive), or -1. */
function axisIndex(axes: readonly NgffAxis[], want: string): number {
  return axes.findIndex((ax) => (ax.name ?? "").toLowerCase() === want);
}

/** Pull the x and y components out of a per-axis vector, defaulting anything absent. */
function xy(vec: unknown, axes: readonly NgffAxis[], dflt: number): [number, number] {
  const arr = asArray(vec);
  if (!arr) return [dflt, dflt];
  const ix = axisIndex(axes, "x");
  const iy = axisIndex(axes, "y");
  // Without axis names, assume the NGFF convention that the LAST two entries are (y, x) — the axis
  // order is c,z,y,x, so a 3-vector [1,1,1] over c,y,x still lands correctly.
  const px = ix >= 0 ? ix : arr.length - 1;
  const py = iy >= 0 ? iy : arr.length - 2;
  const num = (i: number) => (typeof arr[i] === "number" ? (arr[i] as number) : dflt);
  return [num(px), num(py)];
}

function oneTransform(t: unknown, ctx: Ctx): Affine2 {
  const rec = asRecord(t);
  if (!rec) return IDENTITY2;
  const type = typeof rec.type === "string" ? rec.type : "";
  // A transform may carry its own axes (SpatialData writes input/output); prefer those.
  const inputAxes = (asRecord(rec.input)?.axes as NgffAxis[] | undefined) ?? ctx.axes;
  switch (type) {
    case "identity":
      return IDENTITY2;
    case "scale": {
      const [sx, sy] = xy(rec.scale, inputAxes, 1);
      return { a: sx, b: 0, c: 0, d: sy, tx: 0, ty: 0 };
    }
    case "translation": {
      const [tx, ty] = xy(rec.translation, inputAxes, 0);
      return { ...IDENTITY2, tx, ty };
    }
    case "sequence": {
      let acc = IDENTITY2;
      for (const step of asArray(rec.transformations) ?? []) {
        acc = compose2(acc, oneTransform(step, { ...ctx, axes: inputAxes }));
      }
      return acc;
    }
    case "affine": {
      // An (n)×(n+1) or (n+1)×(n+1) row-major matrix in axis order; take the x and y rows/columns.
      const rows = asArray(rec.affine);
      const ix = axisIndex(inputAxes, "x");
      const iy = axisIndex(inputAxes, "y");
      if (!rows || ix < 0 || iy < 0) {
        ctx.unsupported.push("affine (unnamed axes)");
        return IDENTITY2;
      }
      const cell = (r: number, c: number): number => {
        const row = asArray(rows[r]);
        const v = row?.[c];
        return typeof v === "number" ? v : r === c ? 1 : 0;
      };
      const last = (asArray(rows[ix])?.length ?? 0) - 1;
      return {
        a: cell(ix, ix),
        c: cell(ix, iy),
        tx: cell(ix, last),
        b: cell(iy, ix),
        d: cell(iy, iy),
        ty: cell(iy, last),
      };
    }
    default:
      // byDimension, and the non-linear types. Recorded, not guessed at.
      if (type) ctx.unsupported.push(type);
      return IDENTITY2;
  }
}

export interface ResolveOptions {
  /** Prefer the transform whose `output.name` is this coordinate system. */
  target?: string;
  /** Axes to use when a transform does not carry its own (e.g. bare NGFF dataset transforms). */
  axes?: readonly NgffAxis[];
}

/**
 * Resolve an element's `coordinateTransformations` to a 2-D affine plus its physical unit.
 *
 * Accepts either shape found in the wild: SpatialData's list of input/output-tagged transforms, or
 * a bare NGFF list (dataset-level `scale`/`translation`) for which `opts.axes` supplies the naming.
 * When several transforms are present, the one whose output system matches `target` wins; without a
 * match the first is used, because a single-system element is the common case and refusing to
 * resolve it would be unhelpful.
 */
export function resolveNgffXY(transformations: unknown, opts: ResolveOptions = {}): ResolvedSpace | undefined {
  const list = asArray(transformations);
  if (!list || list.length === 0) return undefined;
  const named = opts.target ? list.find((t) => (asRecord(asRecord(t)?.output)?.name as string | undefined) === opts.target) : undefined;
  const chosen = named ?? list[0];
  const ctx: Ctx = { axes: opts.axes ?? [], unsupported: [] };

  // A bare NGFF list (no input/output tagging) is a SEQUENCE of transforms, not alternatives.
  const tagged = asRecord(chosen)?.output !== undefined;
  let affine = IDENTITY2;
  if (tagged) {
    affine = oneTransform(chosen, ctx);
  } else {
    for (const t of list) affine = compose2(affine, oneTransform(t, ctx));
  }

  const out = asRecord(asRecord(chosen)?.output);
  const outAxes = (out?.axes as NgffAxis[] | undefined) ?? opts.axes ?? [];
  const ix = axisIndex(outAxes, "x");
  const unit = (ix >= 0 ? outAxes[ix]?.unit : undefined) ?? outAxes.find((ax) => ax.type === "space")?.unit;
  return {
    affine,
    system: typeof out?.name === "string" ? out.name : undefined,
    unit,
    micrometres: micrometresPer(unit),
    unsupported: ctx.unsupported,
  };
}
