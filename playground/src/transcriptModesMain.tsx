import { createRoot } from "react-dom/client";
import { TranscriptModes } from "./transcript/TranscriptModes";

// No StrictMode: its dev-only double effects would queue every GPU compute twice.
// biome-ignore lint/style/noNonNullAssertion: root element render
createRoot(document.getElementById("root")!).render(<TranscriptModes />);
