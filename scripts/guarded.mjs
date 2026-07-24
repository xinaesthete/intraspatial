#!/usr/bin/env node
// Memory + wall-clock watchdog for a child process and its descendants.
//
// Why this exists: `tsc --noEmit` on this repo runs the TypeScript 7.0.2 *native*
// compiler (the `typescript` package's bin execve's into the platform Mach-O
// binary). Normally that is fast and low-memory, but on 2026-07-23 two concurrent
// agent typechecks ballooned to 67 GB + 32 GB RSS, orphaned, survived the parent
// exit, and took the host to the macOS out-of-memory wall. This wrapper caps any
// such runaway: it polls the child's whole process tree and SIGKILLs it the moment
// total RSS or elapsed time crosses a threshold, so a runaway fails fast instead of
// killing the machine. That bound is also what makes it safe to fan typechecks out
// across parallel workers.
//
// Usage:   node scripts/guarded.mjs -- <cmd> [args...]
// Tunables (env): GUARD_MEM_MB (default 12000), GUARD_TIME_S (default 300),
//                 GUARD_POLL_MS (default 1000).

import { spawn, execSync } from "node:child_process";

const MEM_MB = Number(process.env.GUARD_MEM_MB ?? 12000);
const TIME_S = Number(process.env.GUARD_TIME_S ?? 300);
const POLL_MS = Number(process.env.GUARD_POLL_MS ?? 1000);

const argv = process.argv.slice(2);
const sep = argv.indexOf("--");
const cmd = sep >= 0 ? argv.slice(sep + 1) : argv;
if (cmd.length === 0) {
  console.error("guarded: usage: node scripts/guarded.mjs -- <cmd> [args...]");
  process.exit(2);
}

const child = spawn(cmd[0], cmd.slice(1), { stdio: "inherit" });

/** Sum RSS (MB) of `root` and every descendant; also return the pid set to kill. */
function tree(root) {
  let out;
  try {
    // macOS/BSD ps: `-A` = all processes, `key=` suppresses the column header.
    out = execSync("ps -A -o pid=,ppid=,rss=", { encoding: "utf8" });
  } catch {
    return { pids: [root], rssMB: 0 };
  }
  const kids = new Map();
  const rssKb = new Map();
  for (const line of out.split("\n")) {
    const m = line.trim().match(/^(\d+)\s+(\d+)\s+(\d+)$/);
    if (!m) continue;
    const pid = +m[1];
    const ppid = +m[2];
    rssKb.set(pid, +m[3]);
    if (!kids.has(ppid)) kids.set(ppid, []);
    kids.get(ppid).push(pid);
  }
  const pids = [];
  const stack = [root];
  while (stack.length) {
    const p = stack.pop();
    pids.push(p);
    for (const c of kids.get(p) ?? []) stack.push(c);
  }
  let kb = 0;
  for (const p of pids) kb += rssKb.get(p) ?? 0;
  return { pids, rssMB: Math.round(kb / 1024) };
}

const start = Date.now();
const timer = setInterval(() => {
  if (child.exitCode !== null || child.signalCode) return;
  if (!child.pid) return;
  const { pids, rssMB } = tree(child.pid);
  const elapsed = (Date.now() - start) / 1000;
  const reason =
    rssMB > MEM_MB
      ? `RSS ${rssMB}MB > ${MEM_MB}MB cap`
      : elapsed > TIME_S
        ? `elapsed ${elapsed | 0}s > ${TIME_S}s cap`
        : null;
  if (reason) {
    console.error(`\nguarded: ${reason} — SIGKILL pid tree [${pids.join(",")}] running: ${cmd.join(" ")}`);
    for (const p of pids) {
      try {
        process.kill(p, "SIGKILL");
      } catch {}
    }
    clearInterval(timer);
    process.exit(137);
  }
}, POLL_MS);

child.on("exit", (code, signal) => {
  clearInterval(timer);
  if (signal) {
    console.error(`guarded: child terminated by ${signal}`);
    process.exit(1);
  }
  process.exit(code ?? 0);
});
child.on("error", (err) => {
  clearInterval(timer);
  console.error(`guarded: failed to spawn "${cmd[0]}":`, err.message);
  process.exit(2);
});
