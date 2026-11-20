// Runs a phase diagram off the main thread and posts each grid point as it finishes. A run is
// thousands of simulation steps, so doing this on the page's own thread would freeze it.
import { type PhaseOptions, type PhasePoint, phaseDiagramRuns } from "../../../src/gpu/sim/springs/index";

export type PhaseMessage = { type: "point"; point: PhasePoint } | { type: "done" };

self.onmessage = (e: MessageEvent<PhaseOptions>) => {
  for (const point of phaseDiagramRuns(e.data)) {
    const message: PhaseMessage = { type: "point", point };
    self.postMessage(message);
  }
  const done: PhaseMessage = { type: "done" };
  self.postMessage(done);
};
