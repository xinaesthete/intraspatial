import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import typegpu from "unplugin-typegpu/vite";
import { fileURLToPath } from "node:url";

// The composer imports the toolbox's own runtime + ops from ../src. Those modules
// pull in src/gpu/device.ts, which statically imports the Node-only `webgpu` (Dawn)
// package — in the browser we use `navigator.gpu` instead, so we alias `webgpu` to a
// stub whose `create`/`globals` are never actually called (device.ts prefers
// navigator.gpu when present). unplugin-typegpu transpiles the `"use gpu"` kernels,
// exactly as the GPU vitest config does, so the SAME op definitions run here.
export default defineConfig({
  plugins: [react(), typegpu()],
  resolve: {
    alias: {
      webgpu: fileURLToPath(new URL("./src/webgpu-stub.ts", import.meta.url)),
    },
    // The composer's own imports and the ../src imports must share ONE TypeGPU
    // instance, or its internal registries (and `instanceof` checks) break.
    dedupe: ["typegpu"],
  },
  server: {
    port: Number(process.env.PORT) || 5173,
    fs: { allow: [".."] }, // allow importing ../src/...
  },
});
