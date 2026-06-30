// The pedigree — a serialisable family tree of bred specimens. Mutator's interface was
// always aesthetic selection *with memory*: Latham kept the lineage so a form could be
// traced, re-selected, and branched. Here it doubles as provenance (the repo's FAIR
// ethos, docs-site/concepts/fair.md): a lineage is plain JSON, so a breeding session
// saves, shares, and replays. Pure data + pure (immutable) updates.
import { deserializeSpecimen, serializeSpecimen, type Specimen, type SerializedSpecimen } from "./specimen";

export type BirthOp = "seed" | "mutate" | "marry" | "steer";

export interface PedigreeNode {
  id: string;
  specimen: SerializedSpecimen;
  /** [] for a root, [p] for a mutation/steer, [a,b] for a marriage. */
  parents: string[];
  op: BirthOp;
  generation: number;
}

export interface Pedigree {
  nodes: Record<string, PedigreeNode>;
  /** Ids with no parents. */
  roots: string[];
  /** The artist's current selection path through the tree (most recent last). */
  selectedPath: string[];
}

export function emptyPedigree(): Pedigree {
  return { nodes: {}, roots: [], selectedPath: [] };
}

/** A specimen's stable id within a pedigree — its own seed in hex. Seeds are derived
 *  uniquely per birth (see `mutator`), so this collides only if two individuals are
 *  literally the same draw. */
export function specimenId(sp: Specimen): string {
  return (sp.seed >>> 0).toString(16).padStart(8, "0");
}

/** Record a birth, immutably. Returns a new pedigree; the input is untouched. */
export function recordBirth(
  ped: Pedigree,
  specimen: Specimen,
  op: BirthOp,
  parents: string[],
  generation: number,
): Pedigree {
  const id = specimenId(specimen);
  if (ped.nodes[id]) return ped; // already recorded
  const node: PedigreeNode = { id, specimen: serializeSpecimen(specimen), parents, op, generation };
  return {
    nodes: { ...ped.nodes, [id]: node },
    roots: parents.length === 0 ? [...ped.roots, id] : ped.roots,
    selectedPath: ped.selectedPath,
  };
}

/** Mark `id` as the current selection (appends to the path). */
export function select(ped: Pedigree, id: string): Pedigree {
  if (!ped.nodes[id]) return ped;
  return { ...ped, selectedPath: [...ped.selectedPath, id] };
}

/** Walk from a node back to its root(s) — the ancestry of an individual. */
export function ancestry(ped: Pedigree, id: string): PedigreeNode[] {
  const out: PedigreeNode[] = [];
  const seen = new Set<string>();
  const visit = (nid: string) => {
    const node = ped.nodes[nid];
    if (!node || seen.has(nid)) return;
    seen.add(nid);
    out.push(node);
    for (const p of node.parents) visit(p);
  };
  visit(id);
  return out;
}

export function toJSON(ped: Pedigree): unknown {
  return ped;
}

export function fromJSON(j: unknown): Pedigree {
  const ped = j as Pedigree;
  // Round-trip the specimens through (de)serialize to normalise array types.
  const nodes: Record<string, PedigreeNode> = {};
  for (const [id, n] of Object.entries(ped.nodes)) {
    nodes[id] = { ...n, specimen: serializeSpecimen(deserializeSpecimen(n.specimen)) };
  }
  return { nodes, roots: [...ped.roots], selectedPath: [...ped.selectedPath] };
}
