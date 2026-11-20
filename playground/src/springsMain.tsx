import { createRoot } from "react-dom/client";
import { SpringsDemo } from "./springs/SpringsDemo";

// No StrictMode: its dev-only double mount would start two simulation loops and two GPU devices.
const root = document.getElementById("root");
if (root) createRoot(root).render(<SpringsDemo />);
