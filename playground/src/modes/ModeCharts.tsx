// The modes as charts, the same on every page: the legend, each mode's share of the variation, and
// which channels make up each. A page that can run a null envelope passes it, and its controls as
// `nullControls`, to sit under the spectrum.

import type { ReactNode } from "react";
import type { SpectrumEnvelopeResult } from "../../../src/gpu/spatial/gramEnvelope";
import type { CoLocationModes } from "../../../src/spatial/gram";
import { Explain } from "../transcript/Explain";
import { LoadingsChart } from "./LoadingsChart";
import { ModeLegend } from "./ModeLegend";
import { ScreeChart } from "./ScreeChart";

export function ModeCharts({
  modes,
  envelope,
  nullControls,
}: {
  modes: CoLocationModes;
  envelope?: SpectrumEnvelopeResult;
  nullControls?: ReactNode;
}) {
  return (
    <>
      <ModeLegend modes={modes} />
      <div className="charts">
        <section>
          <h3>How much of the variation each mode carries</h3>
          <ScreeChart explained={modes.explained} {...(envelope ? { envelope } : {})} />
          {nullControls}
        </section>
        <section>
          <h3>Which channels make up each mode</h3>
          <LoadingsChart modes={modes} />
        </section>
      </div>
    </>
  );
}

/** What the colours mean. `mixtures` names what is mixed: "cells", "genes". */
export function ColoursExplain({ mixtures }: { mixtures: string }) {
  return (
    <Explain
      title="What do the colours mean?"
      math={String.raw`C = M M^\top, \qquad \mathrm{corr} = V \Lambda V^\top, \qquad \text{colour}(x) = \mathrm{OKLab}\big(z(x)\!\cdot\! v_1,\ z(x)\!\cdot\! v_2,\ z(x)\!\cdot\! v_3\big)`}
    >
      Each channel is smoothed into a map, and all the maps are compared at once. The comparison is summarised by a few <b>modes</b>:
      patterns of channels that rise and fall together. Mode 1 sets brightness and modes 2 and 3 set hue, through OKLab, a colour space
      built so that equal steps look equally different — so places with similar colours have similar mixtures of {mixtures}, and how
      different two colours look stands in for how different the mixtures are. Each mode is scaled by its own spread, so modes 2 and 3 are
      visible rather than flattened by mode 1. Hue weight 0 instead gives each mode colour in proportion to how much variation it really
      carries. The legend says which channels push each mode each way.
    </Explain>
  );
}

/** The spectrum and loadings charts, read without a null band. */
export function ChartsExplain() {
  return (
    <Explain title="The spectrum and the loadings">
      The bars of the spectrum are each mode's share of the total variation; they add up to 1. Each row of the loadings is one mode, each
      bar a channel's weight in it. Channels that lean the same way are found in the same places; opposite signs mean they avoid each other.
    </Explain>
  );
}
