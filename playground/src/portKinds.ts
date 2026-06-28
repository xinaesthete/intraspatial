// Shape-kind → colour, shared by every node's port handles so a port's type reads
// the same way everywhere (op nodes, group nodes, boundary stubs).
export const KIND_COLOR: Record<string, string> = {
  points: "#7cc4ff",
  grid: "#9be29b",
  matrix: "#e2b85b",
  scalar: "#d79bff",
  opaque: "#ff9bb5",
  any: "#bbbbbb",
};

export const kindColor = (kind: string): string => KIND_COLOR[kind] ?? "#bbbbbb";
