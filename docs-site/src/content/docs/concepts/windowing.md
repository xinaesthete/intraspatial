---
title: Windowing, not quadrats
description: A smooth, overlapping window in place of a hard threshold — the principle behind the toolbox.
---

The single idea that recurs everywhere here: **a hard threshold is a boxcar
window, and the boxcar is the worst window there is.**

## The quadrat problem

Tile space into cells and count points per cell, and your answer swings with the
arbitrary *size*, *origin*, and *phase* of the grid — the Modifiable Areal Unit
Problem. Two points 1 µm apart but across a cell edge are scored as unrelated; two
points in opposite corners of one cell are scored as together. The relationships you
care about are exactly what the grid destroys.

In signal terms, assigning each point fully to one cell is a **rectangular (boxcar)
window**, which has the worst sidelobes of any window (≈ −13 dB) — maximal spectral
leakage, maximal edge artefacts.

## The fix: a tapered, overlapping window

Replace the boxcar with a smooth, overlapping kernel — Gaussian, Hann,
Epanechnikov. Overlap recovers the cross-boundary relationships; the taper removes
the discontinuity; averaging over overlapping windows makes the result nearly
insensitive to grid phase. This is the move from a histogram to a
[kernel density estimate](/primitives/kde-splat/):

$$
\hat{f}(x) = \frac{1}{n}\sum_{i=1}^{n} K_h\!\left(x - x_i\right),
\qquad K_h(u) = \frac{1}{h}\,K\!\left(\frac{u}{h}\right)
$$

where the boxcar is the degenerate case $K(u) = \tfrac{1}{2}\,\mathbb{1}_{|u|\le 1}$.

## Why it unifies the toolbox

A window is the **shared operator** across both fronts:

- on the image side it is a [convolution kernel](/primitives/separable-convolution/);
- on the cell side it is the KDE / local-statistic kernel;
- a quadrat is just `window(shape = box, overlap = 0)` — one (bad) setting of it.

So we make the window a *parameter*, not a separate family of methods. The same
principle reappears in connectivity: a hard "edge if within radius R" graph is a
boxcar in distance, and [fuzzy adjacency](/primitives/fuzzy-adjacency/) is its
tapered version — which is what makes [fuzzy TDA](/concepts/fuzzy-tda/) tick.

## Two honest caveats

- **The null must follow the window.** Significance tests defined on hard counts
  have to be re-derived as *weighted* permutation nulls once counts become smooth.
- **For pure pairwise co-occurrence, skip the grid.** The cross-PCF and Ripley's K
  are already windowed in the distance domain; reach for a windowed grid when you
  want a spatial *map*, not a function of distance.
