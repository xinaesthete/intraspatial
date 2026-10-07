// Pick channels: whole gene sets (counted as one channel each) and single genes. A gene inside an
// active set is not offered on its own — channels must not share genes.

import { useMemo, useState } from "react";
import type { PointsFeature } from "../datasource/pointsTileLoader";
import { isControl, NEGATIVE_CONTROLS, type Selection, STARTER_SETS, setMembers } from "./channelModel";

export const MAX_CHANNELS = 32;

interface Props {
  readonly features: readonly PointsFeature[];
  readonly selection: Selection;
  readonly onChange: (s: Selection) => void;
  readonly missing: Readonly<Record<string, readonly string[]>>;
}

export function ChannelPicker({ features, selection, onChange, missing }: Props) {
  const [query, setQuery] = useState("");
  const count = selection.sets.length + selection.genes.length;
  const full = count >= MAX_CHANNELS;
  const inSets = useMemo(() => new Set(selection.sets.flatMap((s) => setMembers(s, features))), [selection.sets, features]);
  const matches = useMemo(() => {
    const q = query.trim().toUpperCase();
    if (!q) return [];
    return features
      .filter((f) => !isControl(f.name) && f.name.toUpperCase().includes(q) && !inSets.has(f.name) && !selection.genes.includes(f.name))
      .slice(0, 24);
  }, [query, features, inSets, selection.genes]);

  const toggleSet = (name: string) => {
    if (selection.sets.includes(name)) return onChange({ ...selection, sets: selection.sets.filter((s) => s !== name) });
    if (full) return;
    const members = new Set(setMembers(name, features));
    onChange({ sets: [...selection.sets, name], genes: selection.genes.filter((g) => !members.has(g)) });
  };

  return (
    <div className="picker">
      <div className="picker-head">
        Channels: {count} of at most {MAX_CHANNELS}
      </div>
      <div className="chips">
        {[...STARTER_SETS.map((s) => s.name), NEGATIVE_CONTROLS].map((name) => (
          <button
            key={name}
            type="button"
            className={selection.sets.includes(name) ? "chip on" : "chip"}
            title={setMembers(name, features).join(", ")}
            onClick={() => toggleSet(name)}
          >
            {name}
            {missing[name]?.length ? <span className="chip-note"> ({missing[name].length} not in panel)</span> : null}
          </button>
        ))}
      </div>
      <input className="search" placeholder="Add a single gene…" value={query} onChange={(e) => setQuery(e.target.value)} disabled={full} />
      {matches.length > 0 && (
        <div className="chips">
          {matches.map((f) => (
            <button
              key={f.code}
              type="button"
              className="chip"
              onClick={() => {
                onChange({ ...selection, genes: [...selection.genes, f.name] });
                setQuery("");
              }}
            >
              + {f.name}
            </button>
          ))}
        </div>
      )}
      {selection.genes.length > 0 && (
        <div className="chips">
          {selection.genes.map((g) => (
            <button
              key={g}
              type="button"
              className="chip on"
              onClick={() => onChange({ ...selection, genes: selection.genes.filter((x) => x !== g) })}
            >
              {g} ×
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
