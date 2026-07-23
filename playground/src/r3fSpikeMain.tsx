import { createRoot } from "react-dom/client";
import R3fSpike from "./R3fSpike";

// No StrictMode: its dev-only double-mount would init → dispose → re-init the renderer on the SAME
// canvas. R3F owns the canvas here, but the WebGPU context is still un-re-acquirable, and the whole
// point of this page is to measure renderer lifecycle — a deliberate double-mount would muddy it.
// biome-ignore lint/style/noNonNullAssertion: root element render
createRoot(document.getElementById("root")!).render(<R3fSpike />);
