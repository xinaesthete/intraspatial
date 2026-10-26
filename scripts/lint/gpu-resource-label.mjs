// Oxlint rule `intraspatial/gpu-resource-label`: every GPU object this repo creates carries a name.
//
// Hosts (MDV) attribute WebGPU validation errors to their owner by the names in labels — Dawn cites
// the buffer, bind group, pipeline, pass and encoder involved — and an unlabelled one reports as e.g.
// "[Buffer (unlabeled)]" and can't be traced back here. Name them `<kernel>:<role>`.
//
// - Raw WebGPU descriptors need a `label`. luma.gl's `device.createBuffer` takes `id` instead and
//   forwards it as the label; raw descriptors can't carry `id` (TS rejects the excess property).
// - TypeGPU buffers and bind-group layouts need `.$name(…)` in the same chain. unplugin-typegpu would
//   otherwise auto-name them after the variable alone (`params`, `out`, `layout`), which says nothing
//   about the kernel. A TypeGPU bind group takes its layout's name. Two-argument
//   `root.createBuffer(schema, x)` is skipped: it is almost always a wrapper over a raw buffer that
//   already has a label.
//
// A raw `createBuffer` descriptor held in a variable is reported (it reads exactly like TypeGPU's
// `root.createBuffer(schema)`); other descriptors in variables, and a `$name` applied in a later
// statement, are not checked.
//
// Plain objects rather than `definePlugin`/`defineRule` (identity functions, there for typing): this
// file then imports nothing, so dependency analysis doesn't see build tooling pulling in vite-plus.

const DESCRIPTOR_METHODS = new Set([
  "createBuffer",
  "createTexture",
  "createSampler",
  "createShaderModule",
  "createBindGroup",
  "createBindGroupLayout",
  "createPipelineLayout",
  "createComputePipeline",
  "createComputePipelineAsync",
  "createRenderPipeline",
  "createRenderPipelineAsync",
  "createCommandEncoder",
  "beginComputePass",
  "beginRenderPass",
]);
// Methods whose descriptor is optional, so a bare call is also unlabelled.
const OPTIONAL_DESCRIPTOR = new Set(["createCommandEncoder", "beginComputePass", "createSampler"]);

const LABEL_MESSAGE = "GPU object without a `label` — name it `<kernel>:<role>` so validation errors can be attributed";
const TGPU_MESSAGE = {
  // Syntax can't tell `root.createBuffer(schema)` from a raw `device.createBuffer(desc)` with the
  // descriptor in a variable, so the message covers both.
  buffer:
    'Buffer with no visible name — chain `.$name("<kernel>:<role>")` on a TypeGPU buffer, or pass a raw descriptor inline with a `label`',
  layout: "TypeGPU bind group layout without `.$name(…)` — its bind groups take this name; make it `<kernel>:<role>`",
};
// `id` is luma.gl's name for the label (see the header).
const NAME_KEYS = new Set(["label", "id"]);

const memberName = (node) =>
  node.type === "MemberExpression" && !node.computed && node.property.type === "Identifier" ? node.property.name : undefined;

const keyName = (prop) => {
  if (prop.type !== "Property" || prop.computed) return undefined;
  if (prop.key.type === "Identifier") return prop.key.name;
  return prop.key.type === "Literal" ? String(prop.key.value) : undefined;
};

const isObjectLiteral = (node) => node.type === "ObjectExpression";

const isDescriptorCall = (method, args) => DESCRIPTOR_METHODS.has(method) && args.length === 1 && isObjectLiteral(args[0]);
const isNamed = (descriptor) => descriptor.properties.some((p) => NAME_KEYS.has(keyName(p)));
const isOmittedDescriptor = (method, args) => OPTIONAL_DESCRIPTOR.has(method) && args.length === 0;

/** A raw WebGPU call's problem: a descriptor literal without a name, or an optional descriptor left out. */
function rawProblem(method, args) {
  if (isDescriptorCall(method, args)) return isNamed(args[0]) ? undefined : LABEL_MESSAGE;
  return isOmittedDescriptor(method, args) ? LABEL_MESSAGE : undefined;
}

const isTgpuBuffer = (method, args) => method === "createBuffer" && args.length === 1 && !isObjectLiteral(args[0]);
const isTgpuLayout = (method, callee) =>
  method === "bindGroupLayout" && callee.object.type === "Identifier" && callee.object.name === "tgpu";

/** `root.createBuffer(schema)` (one non-literal argument) or `tgpu.bindGroupLayout(…)`, else undefined. */
function typegpuKind(node, method) {
  if (isTgpuBuffer(method, node.arguments)) return "buffer";
  return isTgpuLayout(method, node.callee) ? "layout" : undefined;
}

/** True when some enclosing call is `….$name(…)` — the chain the object was created in names it. */
function namedInChain(node) {
  for (let p = node.parent; p; p = p.parent) {
    if (p.type === "CallExpression" && memberName(p.callee) === "$name") return true;
  }
  return false;
}

const gpuResourceLabel = {
  meta: {
    docs: { description: "Require a `<kernel>:<role>` label (or TypeGPU `$name`) on every GPU object" },
  },
  create(context) {
    return {
      CallExpression(node) {
        const method = memberName(node.callee);
        if (!method) return;
        const raw = rawProblem(method, node.arguments);
        if (raw) {
          context.report({ node: node.callee.property, message: raw });
          return;
        }
        const kind = typegpuKind(node, method);
        if (kind && !namedInChain(node)) context.report({ node, message: TGPU_MESSAGE[kind] });
      },
    };
  },
};

export default {
  meta: { name: "intraspatial" },
  rules: { "gpu-resource-label": gpuResourceLabel },
};
