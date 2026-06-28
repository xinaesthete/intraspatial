// Browser stub for the Node-only `webgpu` (Dawn) package. In the browser,
// src/gpu/device.ts uses `navigator.gpu` and never calls these — but the static
// `import { create, globals } from "webgpu"` must still resolve in the bundle.
export function create(_args?: unknown): never {
  throw new Error("webgpu stub: the browser uses navigator.gpu, not the Dawn addon");
}
export const globals: Record<string, unknown> = {};
