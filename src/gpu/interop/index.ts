// Interop utilities — run our raw-WebGPU/TypeGPU compute on a device owned by another
// context (three.js, deck.gl, …) so buffers can be shared without readback. Host-specific
// buffer bridging lives next to each host; the device-adoption seam is host-agnostic here.
export { adoptDevice } from "./adoptDevice";
