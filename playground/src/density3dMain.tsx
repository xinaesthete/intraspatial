import { createRoot } from "react-dom/client";
import { Density3D } from "./density3d/Density3D";

// No StrictMode: its dev-only double mount would bring deck a second GPU device.
const root = document.getElementById("root");
if (root) createRoot(root).render(<Density3D />);
