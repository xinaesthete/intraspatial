// Make the WebGPU global types (GPUDevice, GPUBuffer, GPUComputePipeline, …) available to
// the docs-site typechecker. The renderer + dancer sim (and the shared src/gpu code they
// import) use them; @webgpu/types is installed but the Astro tsconfig doesn't reference it.
/// <reference types="@webgpu/types" />
