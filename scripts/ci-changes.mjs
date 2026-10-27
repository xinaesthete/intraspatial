#!/usr/bin/env node
// Decide whether a diff can affect the test suites, for ci.yml's `changes` job. Prints the verdict
// and, on CI, writes `code=true|false` to $GITHUB_OUTPUT.
//
// A change is inert when every file it touches is prose or docs-site (Markdown, changesets, docs/,
// the Astro site), another workflow, or a package.json whose only edit is its `version` — which is
// exactly what a Changesets "Version packages" PR contains. Anything else, including a file this
// list doesn't know, counts as code: the default is to run the tests.
//
// Usage:   node scripts/ci-changes.mjs <base-sha> <head-sha>
//          (no base, or the all-zeros sha of a new branch, means "can't tell": code=true)

import { execFileSync } from "node:child_process";
import { appendFileSync } from "node:fs";
import { isDeepStrictEqual } from "node:util";

const INERT = [/\.md$/, /^\.changeset\//, /^docs\//, /^docs-site\//, /^LICENSE$/, /^\.github\/workflows\/(deploy-docs|release)\.yml$/];

const git = (...args) => execFileSync("git", args, { encoding: "utf8" });

// Parsed package.json at `rev` minus its version, or undefined when absent there.
const manifestSansVersion = (rev, path) => {
  let text;
  try {
    text = git("show", `${rev}:${path}`);
  } catch {
    return undefined;
  }
  const { version: _version, ...rest } = JSON.parse(text);
  return rest;
};

const versionOnly = (base, head, path) => {
  const before = manifestSansVersion(base, path);
  return before !== undefined && isDeepStrictEqual(before, manifestSansVersion(head, path));
};

const classify = (base, head) => {
  if (!base || /^0+$/.test(base)) return { code: true, reason: "no base commit to diff against" };
  const files = git("diff", "--name-only", base, head).split("\n").filter(Boolean);
  const live = files.filter((f) => !INERT.some((re) => re.test(f)) && !(/(^|\/)package\.json$/.test(f) && versionOnly(base, head, f)));
  return live.length > 0
    ? { code: true, reason: `code changed: ${live.slice(0, 10).join(", ")}${live.length > 10 ? ", …" : ""}` }
    : { code: false, reason: `only docs/versions changed (${files.length} files)` };
};

const [base, head = "HEAD"] = process.argv.slice(2);
let verdict;
try {
  verdict = classify(base, head);
} catch (err) {
  verdict = { code: true, reason: `diff failed, running everything (${err.message.split("\n")[0]})` };
}
const { code, reason } = verdict;
console.log(`code=${code}: ${reason}`);
if (process.env.GITHUB_OUTPUT) appendFileSync(process.env.GITHUB_OUTPUT, `code=${code}\n`);
