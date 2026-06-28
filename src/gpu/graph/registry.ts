// Op registry — the single source of truth for what nodes exist. `listOps()` is the
// React Flow palette; `getOp(name)` instantiates a node's behaviour. Ops register
// themselves on import (see ./ops/index.ts).
import type { OpType } from "./op";

const registry = new Map<string, OpType>();

export function registerOp(op: OpType): OpType {
  if (registry.has(op.name)) {
    throw new Error(`registry: duplicate op "${op.name}"`);
  }
  registry.set(op.name, op);
  return op;
}

export function getOp(name: string): OpType {
  const op = registry.get(name);
  if (!op) throw new Error(`registry: unknown op "${name}" (have: ${[...registry.keys()].join(", ")})`);
  return op;
}

export function hasOp(name: string): boolean {
  return registry.has(name);
}

export function listOps(): OpType[] {
  return [...registry.values()];
}
