// Lets the custom node (OpNode) report port hovers up to App, which owns the captured
// per-port values and renders the type/data tooltip.
import { createContext } from "react";

export interface PortHoverApi {
  onPortEnter(nodeId: string, port: string, isInput: boolean, kind: string, rect: DOMRect): void;
  onPortLeave(): void;
}

export const PortHoverContext = createContext<PortHoverApi | null>(null);
