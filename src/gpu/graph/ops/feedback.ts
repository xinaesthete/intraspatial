// Metadata for the `feedback` (unit-delay) node so it appears in the palette and
// the composer can wire it. The executor handles it specially (it outputs the
// previous tick's state, seeded by `init`), so `execute` is never called — building
// the node goes through `Graph.feedback`, not the generic op path.
import type { OpType } from "../op";

export const feedbackOp: OpType = {
  name: "feedback",
  label: "Feedback (delay)",
  describe: "Outputs the previous tick's value (seeded by init); wire next to close the loop.",
  inputs: [
    { name: "init", kind: "any" },
    { name: "next", kind: "any" },
  ],
  outputs: [{ name: "state", kind: "any", dtype: "f32" }],
  params: [],
  inferShapes(inputs) {
    return [inputs[0] ?? { kind: "scalar" }];
  },
  execute() {
    throw new Error("feedback is handled by the executor (advance/pull), not executed directly");
  },
};
