// Scatter plot for `points` outputs, with visx axes + grid so coordinates are
// readable (rather than a bare canvas of dots). SVG is fine at the point counts the
// sources generate.
import { AxisBottom, AxisLeft } from "@visx/axis";
import { GridColumns, GridRows } from "@visx/grid";
import { Group } from "@visx/group";
import { scaleLinear } from "@visx/scale";

const AXIS = "#3a4254";
const LABEL = "#8a93a6";
const GRID = "#1f2433";
const DOT = "#6ba8ff";

export function PointsScatter({ data }: { data: ArrayLike<number> }) {
  const W = 300,
    H = 300;
  const m = { top: 10, right: 12, bottom: 28, left: 38 };
  const iw = W - m.left - m.right;
  const ih = H - m.top - m.bottom;

  let minX = Infinity,
    minY = Infinity,
    maxX = -Infinity,
    maxY = -Infinity;
  const pts: { x: number; y: number }[] = [];
  for (let i = 0; i + 1 < data.length; i += 2) {
    const x = data[i] ?? 0;
    const y = data[i + 1] ?? 0;
    pts.push({ x, y });
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
  }
  if (!pts.length) return null;

  const padX = (maxX - minX || 1) * 0.06;
  const padY = (maxY - minY || 1) * 0.06;
  const xScale = scaleLinear({ domain: [minX - padX, maxX + padX], range: [0, iw], nice: true });
  const yScale = scaleLinear({ domain: [minY - padY, maxY + padY], range: [ih, 0], nice: true }); // world +Y up

  const tickLabel = (anchor: "end" | "middle", dx: number, dy: number) => () => ({
    fill: LABEL,
    fontSize: 9,
    fontFamily: "ui-monospace, monospace",
    textAnchor: anchor as "end" | "middle",
    dx,
    dy,
  });

  return (
    <svg width={W} height={H} className="preview-svg" role="img" aria-label="points scatter">
      <Group left={m.left} top={m.top}>
        <GridRows scale={yScale} width={iw} stroke={GRID} numTicks={5} />
        <GridColumns scale={xScale} height={ih} stroke={GRID} numTicks={5} />
        {pts.map((p, i) => (
          // biome-ignore lint/suspicious/noArrayIndexKey: pending...
          <circle key={i} cx={xScale(p.x)} cy={yScale(p.y)} r={2.5} fill={DOT} fillOpacity={0.85} />
        ))}
        <AxisLeft scale={yScale} numTicks={5} stroke={AXIS} tickStroke={AXIS} tickLabelProps={tickLabel("end", -2, 3)} />
        <AxisBottom scale={xScale} top={ih} numTicks={5} stroke={AXIS} tickStroke={AXIS} tickLabelProps={tickLabel("middle", 0, 2)} />
      </Group>
    </svg>
  );
}
