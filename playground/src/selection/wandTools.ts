// PROTOTYPE: the map's tools, Hand (drag pans), Wand (hover previews the patch, drag paints it) and Brush
// (drag paints a disc), as deck widgets, and a cursor that says what a press will do before it happens.

import { IconWidget } from "@deck.gl/widgets";
import type { PressOp } from "./worldMask";

export type MapTool = "hand" | "wand" | "brush";

const svg = (body: string, size = 24) =>
  `data:image/svg+xml,${encodeURIComponent(`<svg xmlns="http://www.w3.org/2000/svg" width="${size}" height="${size}" viewBox="0 0 ${size} ${size}">${body}</svg>`)}`;

/** Widget icons, used as CSS masks: only their shape matters. */
const HAND_ICON = svg('<path d="M12 2l3 3h-2v5h-2V5H9zM12 22l-3-3h2v-5h2v5h2zM2 12l3-3v2h5v2H5v2zM22 12l-3 3v-2h-5v-2h5V9z"/>');
const WAND_ICON = svg(
  '<path d="M3 19.6 14.6 8l1.4 1.4L4.4 21z"/><path d="M18 2l1 2.6 2.6 1-2.6 1-1 2.6-1-2.6-2.6-1 2.6-1z"/><path d="M9 2.5l.6 1.5 1.5.6-1.5.6L9 6.7l-.6-1.5-1.5-.6 1.5-.6z"/>',
);

const BRUSH_ICON = svg(
  '<path d="M7 14c-1.7 0-3 1.3-3 3 0 1.3-1.2 2-2 2 .9 1.2 2.5 2 4 2 2.2 0 4-1.8 4-4 0-1.7-1.3-3-3-3zM20.7 4.7l-1.4-1.4a1 1 0 0 0-1.4 0L9 12.2l2.8 2.8 8.9-8.9a1 1 0 0 0 0-1.4z"/>',
);

const ACTIVE = "#fbbf24";

/** The toolbar: one button per tool, the active one lit. Stateless, so the page owns the tool. */
export function toolWidgets(tool: MapTool, choose: (tool: MapTool) => void): IconWidget[] {
  return [
    new IconWidget({
      id: "tool-hand",
      placement: "top-left",
      icon: HAND_ICON,
      label: "Hand (H): drag to pan",
      color: tool === "hand" ? ACTIVE : "",
      onClick: () => choose("hand"),
    }),
    new IconWidget({
      id: "tool-wand",
      placement: "top-left",
      icon: WAND_ICON,
      label:
        "Wand (W): hover to preview, drag to paint; Shift add, Alt subtract, Shift+Alt intersect; Cmd/Ctrl+scroll or [ ] sizes the patch; Space pans",
      color: tool === "wand" ? ACTIVE : "",
      onClick: () => choose("wand"),
    }),
    new IconWidget({
      id: "tool-brush",
      placement: "top-left",
      icon: BRUSH_ICON,
      label: "Brush (B): drag to paint a disc; Shift add, Alt subtract, Shift+Alt intersect; Cmd/Ctrl+scroll or [ ] sizes it",
      color: tool === "brush" ? ACTIVE : "",
      onClick: () => choose("brush"),
    }),
  ];
}

const BADGE: Record<PressOp, string> = {
  replace: "",
  add: '<path d="M22 25h8M26 21v8"/>',
  subtract: '<path d="M22 25h8"/>',
  intersect: '<path d="M23 22l6 6M29 22l-6 6"/>',
  piece: '<circle cx="26" cy="25" r="4.5"/><path d="M23.5 25h5"/>',
};

/** A crosshair with the op's badge, drawn twice (dark under light) so it reads on any background. */
function wandCursor(op: PressOp): string {
  const marks = `<path d="M16 3v9M16 20v9M3 16h9M20 16h9"/><circle cx="16" cy="16" r="1.2"/>${BADGE[op]}`;
  const body = `<g fill="none" stroke-linecap="round"><g stroke="#000" stroke-width="3.5">${marks}</g><g stroke="#fff" stroke-width="1.5">${marks}</g></g>`;
  return `url("${svg(body, 32)}") 16 16, crosshair`;
}

/** The cursor for the tool and the keys held. Space in a painting tool is Photoshop's temporary hand. */
export function toolCursor(tool: MapTool, op: PressOp, space: boolean): string {
  return tool === "hand" || space ? "grab" : wandCursor(op);
}
