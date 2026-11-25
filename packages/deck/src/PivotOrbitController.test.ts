// The pivot controller's one promise: while the scene is turned, the picked point stays on the
// pixel where the drag began. Checked against deck's own OrbitViewport, with deck's plain
// OrbitController beside it to show the difference.
import { OrbitController, OrbitViewport } from "@deck.gl/core";
import { describe, expect, it } from "vite-plus/test";
import { nearestPoint, PivotOrbitController } from "./PivotOrbitController";

const VIEW = { width: 800, height: 600, target: [0, 0, 0] as [number, number, number], zoom: 4, rotationX: 20, rotationOrbit: 30 };
const makeViewport = (props: Record<string, unknown>) => new OrbitViewport({ orbitAxis: "Y", fovy: 35, ...props });

/** A controller's state, built the way deck builds it, with the pivot picker in its options. */
function stateOf(Controller: typeof OrbitController, pivot: number[] | null) {
  const controller = new Controller({
    timeline: { addChannel: () => 0, removeChannel() {}, getTime: () => 0 },
    eventManager: { on() {}, off() {}, watch() {}, unwatch() {} },
    onViewStateChange() {},
    onStateChange() {},
    makeViewport,
    // A stand-in for deck's timeline and event manager, which a state never touches.
  } as object as ConstructorParameters<typeof OrbitController>[0]);
  const options = { ...VIEW, makeViewport, pickPivot: () => pivot };
  return new controller.ControllerState(options);
}

/** Rotate by dragging from `from` to `to`, and say where `point` ends up on screen. */
function dragAndProject(
  Controller: typeof OrbitController,
  point: number[],
  pivot: number[] | null,
  from: [number, number],
  to: [number, number],
) {
  const end = stateOf(Controller, pivot).rotateStart({ pos: from }).rotate({ pos: to });
  const props = end.getViewportProps();
  return { pixel: makeViewport(props).project(point), props };
}

describe("PivotOrbitController", () => {
  // Off-centre and off the target's depth: the case a fixed-target orbit gets wrong.
  const point = [6, -3, 4];
  const home = makeViewport(VIEW).project(point);
  const moved = (pixel: number[]) => Math.hypot(pixel[0]! - home[0]!, pixel[1]! - home[1]!);

  it("keeps the pivot where it is on screen, wherever the drag is made", () => {
    // Once grabbing the point itself, once dragging far from it (a tracked object).
    for (const from of [[home[0]!, home[1]!] as [number, number], [120, 500] as [number, number]]) {
      const { pixel, props } = dragAndProject(PivotOrbitController, point, point, from, [from[0] + 140, from[1] - 60]);
      expect(moved(pixel)).toBeLessThan(0.01);
      // It did turn, and it moved the target to do it.
      expect(props.rotationOrbit).not.toBe(VIEW.rotationOrbit);
      expect(Math.hypot(...props.target!)).toBeGreaterThan(0.1);
    }
  });

  it("is deck's own orbit when there is no pivot", () => {
    const from: [number, number] = [home[0]!, home[1]!];
    const to: [number, number] = [from[0] + 140, from[1] - 60];
    const plain = dragAndProject(OrbitController, point, null, from, to);
    const unpicked = dragAndProject(PivotOrbitController, point, null, from, to);
    expect(unpicked.props).toEqual(plain.props);
    // And deck's own orbit lets the point slide away.
    expect(moved(plain.pixel)).toBeGreaterThan(5);
  });

  it("forgets the pivot when the drag ends", () => {
    const state = stateOf(PivotOrbitController, point).rotateStart({ pos: [10, 10] });
    expect(state.getState()).toMatchObject({ startRotatePivot: point });
    expect(state.rotateEnd().getState()).toMatchObject({ startRotatePivot: null, startRotatePivotPixel: null });
  });
});

describe("nearestPoint", () => {
  // Screen position is x, y; depth is z: an identity "viewport".
  const flat = { project: (p: number[]) => p };
  const positions = new Float32Array([0, 0, 5, 0, 100, 0, 5, 0, 103, 0, 1, 0, 400, 400, 0, 0]);

  it("picks the nearest on screen, and the nearer to the camera among close rivals", () => {
    expect(nearestPoint(positions, [2, 1], flat)?.index).toBe(0);
    // Points 1 and 2 are both within a pixel or so of the pointer: 2 is in front.
    expect(nearestPoint(positions, [101.5, 0], flat)?.index).toBe(2);
  });

  it("picks nothing beyond the reach, and honours skip", () => {
    expect(nearestPoint(positions, [250, 250], flat)).toBeNull();
    expect(nearestPoint(positions, [101.5, 0], flat, { skip: (i) => i === 2 })?.index).toBe(1);
    expect(nearestPoint(positions, [2, 1], flat, { count: 0 })).toBeNull();
  });
});
