// Phase diagrams: run a scenario over a grid of two parameters and record a measurement at each
// point. The result says which regions of parameter space give which behaviour, which is how the
// paper summarises its model (stable spheroid against fingering, over matrix relaxation,
// proliferation and viscosity).
import { projectedShape, type ShapeReadout } from "./readouts";
import { CELL, LEADER, type SpheroidConfig, spheroidScenario } from "./scenarios";

/** The `SpheroidConfig` settings a phase axis can vary: the numeric ones, bar `dims`, since a
 *  diagram measures the 2D outline. */
export type PhaseSetting = Exclude<
  { [K in keyof SpheroidConfig]-?: Required<SpheroidConfig>[K] extends number ? K : never }[keyof SpheroidConfig],
  "dims"
>;

export interface PhaseAxis {
  /** The `SpheroidConfig` field this axis varies. */
  readonly key: PhaseSetting;
  readonly values: readonly number[];
}

export interface PhasePoint {
  /** Index into each axis's values. */
  readonly ix: number;
  readonly iy: number;
  readonly x: number;
  readonly y: number;
  readonly shape: ShapeReadout;
  /** Area relative to the starting outline; NaN when the starting outline has no area. */
  readonly growth: number;
  readonly brokenBonds: number;
  readonly divisions: number;
}

export interface PhaseOptions {
  readonly x: PhaseAxis;
  readonly y: PhaseAxis;
  /** Settings shared by every run. */
  readonly base?: SpheroidConfig;
  /** Steps per run. */
  readonly steps: number;
}

/**
 * The runs of a phase diagram, one grid point per `next()`, row by row. A generator, so a page
 * can draw each point as it arrives and stay responsive.
 */
export function* phaseDiagramRuns(o: PhaseOptions): Generator<PhasePoint> {
  // The y value would overwrite the x value, and every column would be labelled wrongly.
  if (o.x.key === o.y.key) throw new RangeError(`phaseDiagram: both axes vary "${o.x.key}"`);
  // A leader is still tissue: count it, or a run loses area when surface cells change species.
  const tissue = [CELL, LEADER];
  for (let iy = 0; iy < o.y.values.length; iy++) {
    for (let ix = 0; ix < o.x.values.length; ix++) {
      const x = o.x.values[ix]!;
      const y = o.y.values[iy]!;
      const sc = spheroidScenario({ ...o.base, [o.x.key]: x, [o.y.key]: y });
      const start = projectedShape(sc.state, tissue).area;
      sc.advance(o.steps);
      const shape = projectedShape(sc.state, tissue);
      yield { ix, iy, x, y, shape, growth: shape.area / start, brokenBonds: sc.totals.broken, divisions: sc.totals.divisions };
    }
  }
}

/** Every point of a phase diagram. */
export function phaseDiagram(o: PhaseOptions): PhasePoint[] {
  return [...phaseDiagramRuns(o)];
}
