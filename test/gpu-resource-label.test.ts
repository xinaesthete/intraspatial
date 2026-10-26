// The `intraspatial/gpu-resource-label` Oxlint rule (scripts/lint/gpu-resource-label.mjs): what it
// flags, and the forms it deliberately lets through.
import { RuleTester } from "vite-plus/lint/plugins-dev";
import { describe, it } from "vite-plus/test";

import plugin from "../scripts/lint/gpu-resource-label.mjs";

RuleTester.describe = describe;
RuleTester.it = it;

const tester = new RuleTester({ languageOptions: { parserOptions: { lang: "ts" } } });
const LABEL = /GPU object without a `label`/;
const TGPU_BUFFER = /Buffer with no visible name/;
const TGPU_LAYOUT = /TypeGPU bind group layout without `\.\$name/;

tester.run("gpu-resource-label", plugin.rules["gpu-resource-label"], {
  valid: [
    'device.createBuffer({ label: "k:b", size: 4, usage: 1 });',
    "device.createBindGroup({ label, layout, entries: [] });",
    'device.createCommandEncoder({ label: "k" });',
    'enc.beginComputePass({ label: "k:pass" });',
    // luma.gl names resources with `id`, which it forwards as the label.
    'device.createBuffer({ id: "layer:max", handle, byteLength: 4 });',
    'root.createBuffer(d.f32).$usage("uniform").$name("k:params");',
    'tgpu.bindGroupLayout({ a: { uniform: d.f32 } }).$name("k");',
    // Wrapper over a raw buffer, which carries the label.
    "root.createBuffer(d.arrayOf(d.f32, n), raw);",
    // Other descriptors in a variable are out of reach.
    "device.createTexture(desc);",
  ],
  invalid: [
    { code: "device.createBuffer({ size: 4, usage: 1 });", errors: [{ message: LABEL }] },
    { code: "device.createComputePipeline({ layout, compute: { module, entryPoint } });", errors: [{ message: LABEL }] },
    { code: "device.createCommandEncoder();", errors: [{ message: LABEL }] },
    { code: "enc.beginComputePass();", errors: [{ message: LABEL }] },
    { code: 'root.createBuffer(d.f32).$usage("uniform");', errors: [{ message: TGPU_BUFFER }] },
    { code: 'const p = { q: root.createBuffer(d.f32).$usage("storage") };', errors: [{ message: TGPU_BUFFER }] },
    { code: "tgpu.bindGroupLayout({ a: { uniform: d.f32 } });", errors: [{ message: TGPU_LAYOUT }] },
    // Indistinguishable from `root.createBuffer(schema)`: inline the descriptor so its label shows.
    { code: "device.createBuffer(desc);", errors: [{ message: TGPU_BUFFER }] },
  ],
});
