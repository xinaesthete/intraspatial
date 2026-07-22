import { createRoot } from "react-dom/client";
import SpatialScene from "./SpatialScene";

// No StrictMode here: its dev-only double-mount would init → dispose → re-init the WebGPURenderer on
// the SAME canvas, and a WebGPU context can't be cleanly re-acquired on a canvas it already
// configured. One mount, one renderer.
// biome-ignore lint/style/noNonNullAssertion: root element render
createRoot(document.getElementById("root")!).render(<SpatialScene />);
