// What a frame costs the GPU, in milliseconds: the time from deck finishing a frame's commands to
// the device reporting them done. Unlike frames per second it does not depend on the display's
// refresh rate or on the tab being visible (a hidden tab's animation clock is throttled).
//
// It covers everything queued on the device for that frame, so simulation steps submitted since the
// last frame are included: pause the simulation to see the drawing alone.

import { useCallback, useRef, useState } from "react";

export interface GpuFrameCost {
  /** Smoothed milliseconds per frame, or null before the first measurement. */
  readonly ms: number | null;
  /** For Deck's `onAfterRender`. Stable for a given device. */
  readonly onAfterRender: () => void;
}

export function useGpuFrameCost(device: GPUDevice | null): GpuFrameCost {
  const [ms, setMs] = useState<number | null>(null);
  const pending = useRef(false);
  const smoothed = useRef<number | null>(null);
  const shown = useRef(0);
  const onAfterRender = useCallback(() => {
    if (!device || pending.current) return;
    pending.current = true;
    const t0 = performance.now();
    void device.queue.onSubmittedWorkDone().then(() => {
      pending.current = false;
      const dt = performance.now() - t0;
      smoothed.current = smoothed.current === null ? dt : smoothed.current * 0.9 + dt * 0.1;
      // A few updates a second: this is a readout, not an animation.
      if (t0 - shown.current > 250) {
        shown.current = t0;
        setMs(smoothed.current);
      }
    });
  }, [device]);
  return { ms, onAfterRender };
}
