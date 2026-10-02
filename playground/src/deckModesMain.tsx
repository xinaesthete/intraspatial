import { createRoot } from "react-dom/client";
import { DeckModes } from "./transcript/DeckModes";

// No StrictMode: its dev double-mount would create deck's device twice, and the Gram can adopt one.
// biome-ignore lint/style/noNonNullAssertion: root element render
createRoot(document.getElementById("root")!).render(<DeckModes />);
