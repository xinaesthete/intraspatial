import { createRoot } from "react-dom/client";
import { TissueSim } from "./springs/TissueSim";

// No StrictMode: its dev-only double mount would make two Decks, so two GPU devices, and the
// library adopts only one.
const root = document.getElementById("root");
if (root) createRoot(root).render(<TissueSim />);
