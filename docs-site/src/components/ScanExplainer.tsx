// Step-through illustration of the reduce-then-scan prefix sum (`src/gpu/scan/prefixSum.ts`).
// Deliberately tiny: 16 numbers, blocks of 4, so every intermediate fits on one screen.
import { useState } from "react";

const BLOCK = 4;
const INPUT = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3];

function exclusiveScan(xs: number[]): number[] {
  const out: number[] = [];
  let s = 0;
  for (const x of xs) {
    out.push(s);
    s += x;
  }
  return out;
}

const blocks = Array.from({ length: INPUT.length / BLOCK }, (_, b) => INPUT.slice(b * BLOCK, (b + 1) * BLOCK));
const local = blocks.map(exclusiveScan);
const blockSums = blocks.map((blk) => blk.reduce((a, x) => a + x, 0));
const blockOffsets = exclusiveScan(blockSums);
const final = local.flatMap((blk, b) => blk.map((v) => v + blockOffsets[b]!));

const STEPS = [
  {
    title: "0 · The input",
    text: "Sixteen numbers. We want, for every position, the sum of everything BEFORE it — an exclusive prefix sum. A single loop does it in sixteen steps, one after the other; a GPU has thousands of threads that would all be idle.",
  },
  {
    title: "1 · Scan each block on its own",
    text: "Cut the row into blocks of four. Each block is scanned independently, in parallel, as if it were the whole input. Every block also writes down its own TOTAL (the number an inclusive scan would put just past its end).",
  },
  {
    title: "2 · Scan the block totals",
    text: "The four totals are a much shorter row. Scan THEM — with the very same kernel — and each block learns how much sits to its left in the whole input. For a million elements this step is itself big enough to need blocks, so the kernel recurses; it bottoms out after two or three levels.",
  },
  {
    title: "3 · Add the offsets back",
    text: "Each block adds its offset to every entry of its local scan. That is the answer. Every pass touched each element once, in parallel — the bandwidth of about two reads per element, and no thread ever waited for another block.",
  },
];

const cell = (v: number, hue: number, dim = false): React.CSSProperties => ({
  display: "inline-block",
  minWidth: "2.2em",
  padding: "0.25em 0.1em",
  margin: "0.1em",
  textAlign: "center",
  borderRadius: "0.3em",
  background: `hsl(${hue} 60% ${dim ? 92 : 80}% / ${dim ? 0.5 : 1})`,
  color: "#111",
  fontVariantNumeric: "tabular-nums",
  fontFamily: "var(--sl-font-mono, monospace)",
  opacity: dim ? 0.55 : 1,
});

function Row({ label, values, hue, dim, grouped }: { label: string; values: number[]; hue: number; dim?: boolean; grouped?: boolean }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: "0.75em", margin: "0.35em 0", flexWrap: "wrap" }}>
      <span style={{ minWidth: "9em", fontSize: "0.85em", opacity: 0.8 }}>{label}</span>
      <span>
        {values.map((v, i) => (
          <span
            key={`${label}-${i}`}
            style={{ ...cell(v, hue, dim), marginLeft: grouped && i > 0 && i % BLOCK === 0 ? "0.9em" : undefined }}
          >
            {v}
          </span>
        ))}
      </span>
    </div>
  );
}

export default function ScanExplainer() {
  const [step, setStep] = useState(0);
  const s = STEPS[step]!;
  return (
    <div style={{ border: "1px solid var(--sl-color-gray-5, #ccc)", borderRadius: "0.5em", padding: "1em", margin: "1em 0" }}>
      <div style={{ display: "flex", gap: "0.5em", alignItems: "center", marginBottom: "0.5em" }}>
        <button type="button" onClick={() => setStep((x) => Math.max(0, x - 1))} disabled={step === 0}>
          ◀
        </button>
        <strong style={{ flex: 1 }}>{s.title}</strong>
        <button type="button" onClick={() => setStep((x) => Math.min(STEPS.length - 1, x + 1))} disabled={step === STEPS.length - 1}>
          ▶
        </button>
      </div>
      <p style={{ fontSize: "0.9em", minHeight: "4.5em" }}>{s.text}</p>
      <Row label="input" values={INPUT} hue={210} grouped={step >= 1} dim={step >= 2} />
      {step >= 1 && <Row label="per-block scan" values={local.flat()} hue={140} grouped dim={step >= 3} />}
      {step >= 1 && <Row label="block totals" values={blockSums} hue={30} dim={step >= 3} />}
      {step >= 2 && <Row label="scanned totals" values={blockOffsets} hue={30} />}
      {step >= 3 && <Row label="result" values={final} hue={140} grouped />}
    </div>
  );
}
