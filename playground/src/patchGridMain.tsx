import { createRoot } from "react-dom/client";
import { PatchGridDemo } from "./patchgrid/PatchGridDemo";

// No StrictMode: its dev-only double mount would bring deck a second GPU device.
const root = document.getElementById("root");
if (root) createRoot(root).render(<PatchGridDemo />);
