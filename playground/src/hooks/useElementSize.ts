// An element's rendered size, in CSS pixels, kept current by a ResizeObserver — what a canvas needs
// to size its backing store to the screen rather than to its data.

import { type RefObject, useEffect, useState } from "react";

export function useElementSize(ref: RefObject<Element | null>): { width: number; height: number } {
  const [size, setSize] = useState({ width: 0, height: 0 });
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(([entry]) => {
      if (!entry) return;
      const { width, height } = entry.contentRect;
      setSize((s) => (s.width === width && s.height === height ? s : { width, height }));
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, [ref]);
  return size;
}

/** Device pixels per CSS pixel. */
export const devicePixelRatio = (): number => globalThis.devicePixelRatio || 1;
