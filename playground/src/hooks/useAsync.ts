// Run an async task whenever its inputs change, keeping the previous value on screen until the
// next one lands — so a moving window shows the last result instead of blanking between recomputes.
//
// Only within one `scope`, though. A previous value is a stand-in for the next one only when it
// answers the same question: the last window's result while the window moves, but never another
// store's. Change the scope and the old value is gone at once — on the very render where the scope
// changed, not an effect later — and stays gone if the new task fails.

import { type DependencyList, useEffect, useState } from "react";

export interface AsyncState<T> {
  readonly value?: T;
  readonly error?: Error;
  readonly loading: boolean;
}

interface Scoped<T> extends AsyncState<T> {
  readonly scope: unknown;
}

/**
 * `task` reruns when `deps` change; a result that arrives after newer deps is dropped. The previous
 * value is kept while `scope` (compared with `Object.is`) stays the same; omit it to keep it always.
 */
export function useAsync<T>(task: (signal: AbortSignal) => Promise<T> | undefined, deps: DependencyList, scope?: unknown): AsyncState<T> {
  const [state, setState] = useState<Scoped<T>>({ loading: false, scope });
  // biome-ignore lint/correctness/useExhaustiveDependencies: the caller's deps (and scope) are the contract
  useEffect(() => {
    const ctl = new AbortController();
    const kept = (s: Scoped<T>) => (Object.is(s.scope, scope) ? s.value : undefined);
    const run = task(ctl.signal);
    if (!run) {
      setState((s) => ({ value: kept(s), loading: false, scope }));
      return;
    }
    setState((s) => ({ value: kept(s), loading: true, scope }));
    run.then(
      (value) => {
        if (!ctl.signal.aborted) setState({ value, loading: false, scope });
      },
      (err: unknown) => {
        if (!ctl.signal.aborted)
          setState((s) => ({ value: kept(s), error: err instanceof Error ? err : new Error(String(err)), loading: false, scope }));
      },
    );
    return () => ctl.abort();
  }, [...deps, scope]);
  // Until the effect catches up with a new scope, the old state answers the old question.
  return Object.is(state.scope, scope) ? state : { loading: state.loading };
}

/** `value`, but only once it has stopped changing for `ms` — the "settled" window of a drag. */
export function useSettled<T>(value: T, ms: number): T {
  const [settled, setSettled] = useState(value);
  useEffect(() => {
    const t = setTimeout(() => setSettled(value), ms);
    return () => clearTimeout(t);
  }, [value, ms]);
  return settled;
}
