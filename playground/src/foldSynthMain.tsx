import { createRoot } from "react-dom/client";
import { FoldSynth } from "./fold/FoldSynth";

// No StrictMode: its dev-only double mount would make two Decks, so two GPU devices and two runs.
const root = document.getElementById("root");
if (root) createRoot(root).render(<FoldSynth />);
