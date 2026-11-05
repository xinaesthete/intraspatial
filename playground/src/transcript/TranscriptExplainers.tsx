// The transcript modes page's explainers: what the channels, matrix, colours, lock, radius, window,
// boundaries and quality filter mean, in plain language with the maths beside it.

import { ChartsExplain, ColoursExplain } from "../modes/ModeCharts";
import { WandExplain } from "../modes/WandPanel";
import { BoundariesExplain } from "../wand/WandPanel";
import { Explain } from "./Explain";

export function TranscriptExplainers() {
  return (
    <section className="explainers">
      <Explain title="What is a channel?">
        A channel is one gene, or a group of genes counted together — "T cells" adds up the molecules of several T-cell marker genes. A gene
        can only be in one channel at a time. <b>Negative controls</b> are probes that match no gene: they show what random background looks
        like, so a real pattern should look different from them.
      </Explain>
      <Explain
        title="What does the matrix show?"
        math={String.raw`g_{ab} = \frac{|A|\,\big(C_{ab} - S_{ab}\big)}{W_a W_b}, \qquad C = M M^\top`}
      >
        For each pair of channels, <b>g</b> compares how often their molecules lie within about 2r of each other with what you would expect
        if they were scattered at random over the window. 1 means "no more than chance", 2 means twice as often, below 1 means they keep
        apart. Molecules all sit inside tissue, so almost every pair comes out above 1 — compare pairs with each other, and with the
        negative controls, rather than with 1. In the formula, M holds each channel's smoothed map, W its total, |A| the window area and S
        the self-pairs, which are removed.
      </Explain>
      <ColoursExplain mixtures="genes" />
      <ChartsExplain />
      <WandExplain />
      <Explain title="Locking the modes">
        Normally each window gets its own modes, and each place is compared with the rest of its window: the same colour in two windows can
        mean different mixtures, and the strongest contrast in a window always gets the strongest colours. Locked, every window is measured
        against the window where you locked — same averages, same modes, same scale — so a colour means the same mixture wherever you look,
        and a window with little going on looks muted. Each map counts molecules per µm², so the lock also holds when you change the radius
        or the quality filter: a larger radius smooths the map, and its colours calm down, but a colour still means the same mixture. It
        lapses only if the channels change.
      </Explain>
      <Explain title="Why does the radius set the resolution?">
        r is the neighbourhood each molecule is smoothed over. Anything smaller than r is blurred away, so the map is computed at about
        three pixels per radius — a finer grid would cost more without showing more. A large r shows tissue-scale structure; a small r gets
        closer to single cells, at more cost. A map can only be so many pixels across (2048), so one map covers at most about 680 µm* at r =
        1, or 6.8 mm* at r = 10. Zoomed out past that, the radius is raised until the map covers the whole view, and the radius label says
        so: every molecule still counts, but the neighbourhood is coarser. Zoom in, and the radius you chose comes back.
      </Explain>
      <Explain title="Which part of the slide is analysed?">
        The part in view, once you stop moving: the white outline marks it. Zoomed far out, that can be the whole slide, and the first time
        every molecule on it has to be fetched, which takes a while. Molecules are fetched in tiles and kept, so going back over ground
        already seen is fast. Turn off "follow the view" to keep one window while you look around it. Each mode's sign is kept consistent
        between windows so colours do not flip for no reason — but if two modes swap order, the colours change, and that change is real.
      </Explain>
      <BoundariesExplain mixtures="genes" />
      <Explain title="What is the quality filter?">
        Every molecule has a quality score, qv. At 20, the chance that it was read as the wrong gene is about 1 in 100. Molecules below the
        threshold are dropped, as in Xenium's own per-cell counts. Points stored without a tiled layout are read whole, and that read brings
        only positions and genes, so for them the filter is off.
      </Explain>
    </section>
  );
}
