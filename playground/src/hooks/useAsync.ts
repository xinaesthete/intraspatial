// Run an async task whenever its inputs change, keeping the previous value on screen until the
// next one lands — so a moving window shows the last result instead of blanking between recomputes.

import { type DependencyList, useEffect, useState } from "react";

export interface AsyncState<T> {
  readonly value?: T;
  readonly error?: Error;
  readonly loading: boolean;
}

/** `task` reruns when `deps` change; a result that arrives after newer deps is dropped. */
export function useAsync<T>(task: (signal: AbortSignal) => Promise<T> | undefined, deps: DependencyList): AsyncState<T> {
  const [state, setState] = useState<AsyncState<T>>({ loading: false });
  useEffect(() => {
    const ctl = new AbortController();
    const run = task(ctl.signal);
    if (!run) {
      setState((s) => ({ value: s.value, loading: false }));
      return;
    }
    setState((s) => ({ value: s.value, loading: true }));
    run.then(
      (value) => {
        if (!ctl.signal.aborted) setState({ value, loading: false });
      },
      (err: unknown) => {
        if (!ctl.signal.aborted)
          setState((s) => ({ value: s.value, error: err instanceof Error ? err : new Error(String(err)), loading: false }));
      },
    );
    return () => ctl.abort();
    // biome-ignore lint/correctness/useExhaustiveDependencies: the caller's deps are the contract
  }, deps);
  return state;
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
