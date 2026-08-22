// Root export of the `intraspatial` package: the engine (operation-graph runtime) + device.
// Catalogues are subpath exports — `intraspatial/gpu/spatial`, `intraspatial/datasource`,
// `intraspatial/geometry`, `intraspatial/evo`, `intraspatial/spatial`, `intraspatial/color`,
// `intraspatial/gpu/sim` — and any module can be deep-imported as `intraspatial/<path>`.

export { getDevice } from "./gpu/device";
export * from "./gpu/graph";
