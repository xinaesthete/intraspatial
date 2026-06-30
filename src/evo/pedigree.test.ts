import { describe, expect, it } from "vitest";
import { traitSpaceFromParams } from "./traitSpace";
import { randomSpecimen } from "./specimen";
import { marry } from "./mutator";
import { ancestry, emptyPedigree, fromJSON, recordBirth, select, specimenId, toJSON } from "./pedigree";
import { mulberry32 } from "./rng";
import type { ParamSpec } from "../gpu/graph/op";

const space = traitSpaceFromParams([{ name: "x", type: "number", default: 0.5, min: 0, max: 1 }] as ParamSpec[]);

describe("pedigree", () => {
  it("records roots and children, immutably", () => {
    const a = randomSpecimen(space, 1);
    const b = randomSpecimen(space, 2);
    let ped = emptyPedigree();
    ped = recordBirth(ped, a, "seed", [], 0);
    ped = recordBirth(ped, b, "seed", [], 0);
    const before = ped;
    const child = marry(space, a, b, mulberry32(9));
    ped = recordBirth(ped, child, "marry", [specimenId(a), specimenId(b)], 1);

    expect(before.nodes[specimenId(child)]).toBeUndefined(); // earlier value untouched
    expect(ped.roots).toEqual([specimenId(a), specimenId(b)]);
    expect(ped.nodes[specimenId(child)]!.parents).toEqual([specimenId(a), specimenId(b)]);
  });

  it("ancestry walks back to the roots", () => {
    const a = randomSpecimen(space, 1);
    const b = randomSpecimen(space, 2);
    let ped = emptyPedigree();
    ped = recordBirth(ped, a, "seed", [], 0);
    ped = recordBirth(ped, b, "seed", [], 0);
    const child = marry(space, a, b, mulberry32(9));
    ped = recordBirth(ped, child, "marry", [specimenId(a), specimenId(b)], 1);
    const ids = ancestry(ped, specimenId(child)).map((n) => n.id);
    expect(ids).toContain(specimenId(a));
    expect(ids).toContain(specimenId(b));
    expect(ids).toContain(specimenId(child));
  });

  it("survives a JSON round-trip", () => {
    const a = randomSpecimen(space, 1);
    let ped = recordBirth(emptyPedigree(), a, "seed", [], 0);
    ped = select(ped, specimenId(a));
    const round = fromJSON(JSON.parse(JSON.stringify(toJSON(ped))));
    expect(round.roots).toEqual(ped.roots);
    expect(round.selectedPath).toEqual(ped.selectedPath);
    expect(round.nodes[specimenId(a)]!.specimen.seed).toBe(a.seed);
  });
});
