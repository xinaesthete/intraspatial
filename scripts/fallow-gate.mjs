#!/usr/bin/env node
// The fallow PR gate (`pnpm lint:fallow`, ci.yml's `fallow` job). Runs `fallow audit` once and
// applies our own verdict to its new-vs-inherited attribution:
//
// - FAILS only on dead code the changeset INTRODUCES: unused files/exports/dependencies, unresolved
//   imports, cycles. Those are facts about the import graph.
// - WARNS on introduced complexity and duplication, and never fails on them. With no coverage data
//   fallow's CRAP is CC² + CC, so any five-branch function "fails" (docs/fallow-findings.md).
// - Leaves INHERITED findings out of the annotations: a PR that touches a file is not answerable for
//   what the file already held. They are counted in the job summary.
//
// Usage:   node scripts/fallow-gate.mjs [base-ref]     (default: main)

import { execFileSync } from "node:child_process";
import { appendFileSync } from "node:fs";

const base = process.argv[2] ?? "main";

let raw;
try {
  raw = execFileSync("fallow", ["audit", "--base", base, "--format", "json"], { encoding: "utf8", maxBuffer: 256 << 20 });
} catch (err) {
  // `audit` exits 1 on its own fail verdict; the JSON is still on stdout. No JSON means it broke.
  if (!err.stdout) throw err;
  raw = err.stdout;
}
const report = JSON.parse(raw);

// Escape a workflow-command property / message (GitHub's own rules).
const esc = (s) => String(s).replaceAll("%", "%25").replaceAll("\r", "%0D").replaceAll("\n", "%0A");
const annotate = (level, { file, line, title, message }) => {
  const at = file ? ` file=${esc(file)}${line ? `,line=${line}` : ""},` : " ";
  console.log(`::${level}${at}title=${esc(title)}::${esc(message)}`);
};

// Dead code: every category is a list of findings, each carrying `introduced`.
const deadCode = Object.entries(report.dead_code ?? {}).flatMap(([kind, items]) =>
  Array.isArray(items) ? items.filter((i) => i?.introduced).map((i) => ({ kind, ...i })) : [],
);
const complexity = (report.complexity?.findings ?? []).filter((f) => f.introduced);
const dupes = (report.duplication?.clone_groups ?? []).filter((g) => g.introduced);

for (const d of deadCode) {
  const what = d.package_name ?? d.export_name ?? d.name ?? d.specifier ?? d.path;
  annotate("error", { file: d.path, line: d.line, title: `fallow: ${d.kind.replaceAll("_", " ")}`, message: String(what) });
}
for (const f of complexity) {
  const scores = [
    `cyclomatic ${f.cyclomatic}`,
    `cognitive ${f.cognitive}`,
    f.crap !== undefined ? `CRAP ${f.crap} (no coverage data)` : "",
  ];
  annotate("warning", {
    file: f.path,
    line: f.line,
    title: `fallow: complex function '${f.name}'`,
    message: scores.filter(Boolean).join(", "),
  });
}
for (const g of dupes) {
  const [first, ...rest] = g.instances ?? [];
  const also = rest.map((i) => `${i.file}:${i.start_line}`).join(", ");
  annotate("warning", { file: first?.file, line: first?.start_line, title: "fallow: duplicated code", message: `also at ${also}` });
}

const a = report.attribution ?? {};
const failed = deadCode.length > 0;
const summary = [
  `## fallow: ${failed ? "❌ new dead code" : "✅ no new dead code"}`,
  "",
  "| | introduced | inherited |",
  "| --- | ---: | ---: |",
  `| dead code (gates) | ${deadCode.length} | ${a.dead_code_inherited ?? "?"} |`,
  `| complexity (advisory) | ${complexity.length} | ${a.complexity_inherited ?? "?"} |`,
  `| duplication (advisory) | ${dupes.length} | ${a.duplication_inherited ?? "?"} |`,
  "",
  `Against \`${base}\`, over ${report.changed_files_count ?? "?"} changed files. Inherited findings are not annotated; \`pnpm fallow:report\` lists everything.`,
].join("\n");
console.log(summary);
if (process.env.GITHUB_STEP_SUMMARY) appendFileSync(process.env.GITHUB_STEP_SUMMARY, `${summary}\n`);
process.exitCode = failed ? 1 : 0;
