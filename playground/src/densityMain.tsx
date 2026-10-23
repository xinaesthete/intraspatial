import { createRoot } from "react-dom/client";
import { DensityDemo } from "./density/DensityDemo";

// No StrictMode: its dev-only double mount would bring deck a second GPU device, and this library
// adopts only one per page.
// biome-ignore lint/style/noNonNullAssertion: root element render
createRoot(document.getElementById("root")!).render(<DensityDemo />);
