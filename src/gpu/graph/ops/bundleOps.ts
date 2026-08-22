// Extract and combine ops for composite values (ADR-0023).
//
// A bundle is several values that are ONE value — a grid index is its `start`, `items` and
// `lattice` together — so the parts cannot be wired from different producers. These factories
// generate the two ops that open and close one: `extractOp` takes a part out, `combineOp` puts a
// bundle together.
//
// **Extraction borrows.** The extract op returns the part's `FieldValue` unchanged, sharing the
// bundle's `ResidentBuffer`. It must NOT lease and copy: copying `items` at a million points to
// look at it is exactly what the bundle exists to avoid. The executor notices that the payload
// came in on an input and records a borrow instead of ownership, keeping the bundle alive until
// this port's own consumers have run.
//
// Factories rather than one dynamic `extract` op: ports are read statically by the palette, by
// `inferShapes` and by edge validation, so a runtime part name would cost per-part typing at
// build time. A new bundle type is a few lines of registration.
import type { FieldValue, Shape, ShapeKind } from "../handle";
import type { OpType, Params } from "../op";

/** One member of a bundle. The `kind` is what makes an extract op's output port properly typed
 *  rather than `any`: the part's exact SHAPE needs a run, but its kind never changes, so the
 *  composer can colour the port and refuse a wrong edge at wiring time. */
export interface BundlePart {
  readonly name: string;
  readonly kind: ShapeKind | "any";
  /** Display label for the extract op ("cell offsets"). Defaults to `name`. */
  readonly label?: string;
  readonly describe?: string;
}

/** Describes one bundle type: its `name` tag and the parts it always carries. */
export interface BundleSpec {
  /** Nominal tag — `shapesEqual` compares it, so two bundles with the same part shapes but
   *  different names do not interchange. */
  readonly name: string;
  /** Display label for the palette ("Bucket grid") — also the palette group its part ops sit in. */
  readonly label: string;
  /** The members, in the order the combine op takes them. */
  readonly parts: readonly BundlePart[];
}

/** Part names, for a `PortSpec.bundle` descriptor. */
export function partNames(spec: BundleSpec): readonly string[] {
  return spec.parts.map((p) => p.name);
}

function partSpec(spec: BundleSpec, name: string): BundlePart {
  const p = spec.parts.find((x) => x.name === name);
  if (!p) throw new Error(`${spec.name}: no part "${name}"`);
  return p;
}

function bundleShape(s: Shape, spec: BundleSpec, who: string): Extract<Shape, { kind: "bundle" }> {
  if (s.kind !== "bundle" || s.name !== spec.name) {
    throw new Error(`${who}: expected a "${spec.name}" bundle, got ${s.kind === "bundle" ? `bundle "${s.name}"` : s.kind}`);
  }
  return s;
}

function partOf(v: FieldValue, part: string, who: string): FieldValue {
  const got = v.parts?.[part];
  if (!got) throw new Error(`${who}: the bundle has no part "${part}"`);
  return got;
}

/**
 * `bundle -> that bundle's <part>`. The value is handed back as-is (a borrow, see the header),
 * which is why this op is `resident`: it must be allowed to pass a resident value through
 * untouched rather than have the executor download it on the way out.
 */
export function extractOp(spec: BundleSpec, part: string): OpType {
  const p = partSpec(spec, part);
  const name = `${spec.name}.${part}`;
  const who = name;
  return {
    name,
    // The label says what the part IS; the port keeps the structural name the parts are keyed by.
    label: `${spec.label} → ${p.label ?? part}`,
    category: spec.label,
    describe: p.describe ?? `Take the \`${part}\` part out of a ${spec.label.toLowerCase()}.`,
    inputs: [{ name: "bundle", kind: "bundle", bundle: { name: spec.name, parts: partNames(spec) } }],
    // Typed by the part's declared kind, not `any`: the exact shape still comes from `inferShapes`
    // (it depends on the producer's params), but the KIND is fixed, so the composer colours the
    // port and rejects a wrong edge without waiting for a run.
    outputs: [{ name: part, kind: p.kind }],
    params: [],
    inferShapes(inputs) {
      const s = bundleShape(inputs[0]!, spec, who);
      const got = s.parts[part];
      if (!got) throw new Error(`${who}: the bundle has no part "${part}"`);
      return [got];
    },
    inferPlacement(inputs) {
      // A part carries its own placement; the bundle's (if any) is not it.
      return [undefined];
    },
    resident: true,
    async execute(_ctx, inputs) {
      return [partOf(inputs[0]!, part, who)];
    },
    cpuGolden(inputs) {
      return [partOf(inputs[0]!, part, who)];
    },
  };
}

/**
 * `<parts...> -> bundle`. The inverse of `extractOp`, and a borrow in the same way: the bundle
 * points at its inputs' payloads rather than copying them, so the inputs stay alive as long as
 * the bundle does.
 */
export function combineOp(spec: BundleSpec): OpType {
  const who = `${spec.name}.bundle`;
  return {
    name: who,
    label: `${spec.label} ← parts`,
    category: spec.label,
    describe: `Assemble a ${spec.label.toLowerCase()} from its ${spec.parts.length} parts.`,
    inputs: spec.parts.map((p) => ({ name: p.name, kind: p.kind })),
    outputs: [{ name: "bundle", kind: "bundle", bundle: { name: spec.name, parts: partNames(spec) } }],
    params: [],
    inferShapes(inputs) {
      const parts: Record<string, Shape> = {};
      spec.parts.forEach((p, i) => {
        parts[p.name] = inputs[i]!;
      });
      return [{ kind: "bundle", name: spec.name, parts }];
    },
    resident: true,
    async execute(_ctx, inputs) {
      return [bundleValue(spec, inputs)];
    },
    cpuGolden(inputs) {
      return [bundleValue(spec, inputs)];
    },
  };
}

/** The bundle `FieldValue` for a positional list of parts. Shared by `combineOp` and by any
 *  producer that builds a bundle directly (`gridIndex` does). */
export function bundleValue(spec: BundleSpec, values: FieldValue[]): FieldValue {
  const parts: Record<string, FieldValue> = {};
  const shapes: Record<string, Shape> = {};
  spec.parts.forEach((p, i) => {
    const v = values[i];
    if (!v) throw new Error(`${spec.name}.bundle: missing part "${p.name}"`);
    parts[p.name] = v;
    shapes[p.name] = v.shape;
  });
  // A bundle has no dtype of its own; each part keeps its own. `f32` is the neutral label the
  // rest of the value model expects on every value.
  return { shape: { kind: "bundle", name: spec.name, parts: shapes }, dtype: "f32", parts };
}
