// An orbit controller that turns the scene about a point the scene chooses, not about a fixed
// target: the particle under the cursor, or an object the camera is tracking. The point stays
// where it is on screen while everything else swings round it.
//
// deck's MapController has this (`rotationPivot`); its OrbitController does not, and on WebGPU
// deck cannot find the depth under the cursor at all (controller picking is synchronous, WebGPU
// picking is not). So the scene supplies the points, as two separate questions:
//   - `getPivot(pos, viewport)`: what to turn about when a rotation starts at `pos`. Any world
//     point, under the pointer or not; null orbits the target as usual.
//   - `getAnchor(pos, viewport)`: the world point under the pointer, which zoom and pan hold on to,
//     so they work at its depth; null uses the target's depth as usual.
// `nearestPoint` answers both for particles. After psychogeo's MapCameraControls; see
// docs/camera-comparison.md.

import { OrbitController } from "@deck.gl/core";

/** What a picker needs of a viewport: world → pixels (x, y from the view's top left, and depth). */
export interface Projector {
  project(xyz: number[]): number[];
}

/** A world point for a pointer position (pixels from the view's top left), or null for none. */
export type PivotPicker = (pos: [number, number], viewport: Projector) => number[] | null;

type OrbitStateClass = OrbitController["ControllerState"];
type OrbitStateOptions = ConstructorParameters<OrbitStateClass>[0];
type RotateArgs = Parameters<InstanceType<OrbitStateClass>["rotate"]>[0];

type PivotStart = { startRotatePivot?: number[] | null; startRotatePivotPixel?: number[] | null };

/** deck's orbit state, with the pivot of the rotation in progress carried beside its own start values. */
function withPivot(Base: OrbitStateClass): OrbitStateClass {
  return class PivotOrbitState extends Base {
    private readonly pickPivot?: (pos: number[]) => number[] | null;

    constructor(options: OrbitStateOptions & PivotStart & { pickPivot?: (pos: number[]) => number[] | null }) {
      super(options);
      this.pickPivot = options.pickPivot;
      // Interaction state travels through `getState()`: deck spreads it into the next state's options.
      const carried: PivotStart = {
        startRotatePivot: options.startRotatePivot ?? null,
        startRotatePivotPixel: options.startRotatePivotPixel ?? null,
      };
      Object.assign(this.getState(), carried);
    }

    override rotateStart(args: { pos: [number, number] }) {
      const pivot = this.pickPivot?.(args.pos) ?? null;
      // Where the pivot is on screen now is where it must stay.
      const pixel = pivot ? this.makeViewport(this.getViewportProps()).project(pivot).slice(0, 2) : null;
      const start: PivotStart = { startRotatePivot: pivot, startRotatePivotPixel: pixel };
      return super.rotateStart(args)._getUpdatedState(start);
    }

    override rotate(args: RotateArgs) {
      const turned = super.rotate(args);
      const state: { startRotatePos?: number[] } & PivotStart = this.getState();
      const { startRotatePivot: pivot, startRotatePivotPixel: pixel } = state;
      if (!pivot || !pixel) return turned;
      // The new angles have swung the pivot away. Move the target so that it is back on its pixel.
      const viewport = this.makeViewport(turned.getViewportProps());
      return turned._getUpdatedState(viewport.panByPosition(pivot, pixel));
    }

    override rotateEnd() {
      const end: PivotStart = { startRotatePivot: null, startRotatePivotPixel: null };
      return super.rotateEnd()._getUpdatedState(end);
    }
  };
}

type Pickers = { getPivot?: PivotPicker; getAnchor?: PivotPicker };
type SetPropsArg = Parameters<OrbitController["setProps"]>[0];

/**
 * Use as a view's controller: `controller: { type: PivotOrbitController, getPivot, getAnchor }`.
 * Without them, or when they return null, this is deck's OrbitController.
 */
export class PivotOrbitController extends OrbitController {
  constructor(...args: ConstructorParameters<typeof OrbitController>) {
    super(...args);
    this.ControllerState = withPivot(this.ControllerState);
    this._unproject3D = (pos) => this.pick("getAnchor", pos);
  }

  private pick(which: keyof Pickers, pos: number[]): number[] | null {
    const props: object = this.props;
    const picker = which in props ? (props as Pickers)[which] : undefined;
    if (typeof picker !== "function") return null;
    return picker([pos[0] ?? 0, pos[1] ?? 0], this.makeViewport(this.controllerState.getViewportProps()));
  }

  override setProps(props: SetPropsArg & { pickPivot?: (pos: number[]) => number[] | null }): void {
    // Passed on to the state's constructor, as deck passes its own `unproject3D`.
    props.pickPivot = (pos) => this.pick("getPivot", pos);
    super.setProps(props);
  }
}

export interface NearestPointOptions {
  /** f32 per point in `positions`, x y z first. */
  readonly stride?: number;
  /** Points to consider: the first `count`. Default: all. */
  readonly count?: number;
  /** A point further than this from the pointer, in pixels, is not picked. */
  readonly maxPixels?: number;
  /** Leave out point `i` (a dead or hidden particle). */
  readonly skip?: (i: number) => boolean;
}

/**
 * The point nearest the pointer on screen: its index and world position, or null when none is
 * within `maxPixels`. Among points equally near, the one nearest the camera. For a few thousand
 * points this is fast enough to run at the start of a drag.
 */
export function nearestPoint(
  positions: ArrayLike<number>,
  pos: readonly [number, number],
  viewport: Projector,
  o: NearestPointOptions = {},
): { index: number; position: [number, number, number] } | null {
  const stride = o.stride ?? 4;
  const count = Math.min(o.count ?? Number.POSITIVE_INFINITY, Math.floor(positions.length / stride));
  const reach = o.maxPixels ?? 24;
  let best = -1;
  let bestDistance = reach;
  let bestDepth = Number.POSITIVE_INFINITY;
  for (let i = 0; i < count; i++) {
    if (o.skip?.(i)) continue;
    const [x, y, depth = 0] = viewport.project([positions[i * stride]!, positions[i * stride + 1]!, positions[i * stride + 2]!]);
    if (x === undefined || y === undefined) continue;
    const d = Math.hypot(x - pos[0], y - pos[1]);
    // NaN (a point behind the camera) fails both comparisons and is passed over.
    if (d < bestDistance - 0.5 || (d <= bestDistance + 0.5 && d <= reach && depth < bestDepth)) {
      best = i;
      bestDistance = Math.min(d, bestDistance);
      bestDepth = depth;
    }
  }
  if (best < 0) return null;
  return { index: best, position: [positions[best * stride]!, positions[best * stride + 1]!, positions[best * stride + 2]!] };
}
