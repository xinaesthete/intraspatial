import { createRoot } from "react-dom/client";
import { CellModes } from "./cells/CellModes";

// No StrictMode: its dev-only double effects would queue every GPU compute twice, and its double
// mount would bring deck a second device.
// biome-ignore lint/style/noNonNullAssertion: root element render
createRoot(document.getElementById("root")!).render(<CellModes />);
