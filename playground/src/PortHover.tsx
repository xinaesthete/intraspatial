// Lets the custom node (OpNode) report port hovers up to App, which owns the captured
// per-port values and renders the type/data tooltip.
import { createContext } from "react";

/** What a bundle port declares statically (ADR-0023) — the type name and its members. */
export interface BundleInfo {
  readonly name: string;
  readonly parts: readonly string[];
}

export interface PortHoverApi {
  onPortEnter(nodeId: string, port: string, isInput: boolean, kind: string, rect: DOMRect, bundle?: BundleInfo): void;
  onPortLeave(): void;
}

export const PortHoverContext = createContext<PortHoverApi | null>(null);
