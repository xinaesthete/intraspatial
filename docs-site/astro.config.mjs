// @ts-check
import { defineConfig, passthroughImageService } from 'astro/config';
import starlight from '@astrojs/starlight';

// https://astro.build/config
export default defineConfig({
  // Avoid the native `sharp` image pipeline (its build script is skipped under
  // pnpm); these docs use no optimised images.
  image: { service: passthroughImageService() },
  integrations: [
    starlight({
      title: 'GPU Spatial Primitives',
      description:
        'Composable, interpretable GPU primitives for discrete-cell spatial analysis (TypeGPU / WebGPU).',
      social: [
        { icon: 'github', label: 'GitHub', href: 'https://github.com/' },
      ],
      sidebar: [
        {
          label: 'Start here',
          items: [
            { label: 'Overview', slug: 'index' },
            { label: 'Composable & interpretable', slug: 'concepts/composable-interpretable' },
          ],
        },
        {
          label: 'Concepts',
          items: [
            { label: 'Windowing, not quadrats', slug: 'concepts/windowing' },
            { label: 'Render vs compute', slug: 'concepts/render-vs-compute' },
            { label: 'Fuzzy TDA', slug: 'concepts/fuzzy-tda' },
          ],
        },
        {
          label: 'Primitives',
          items: [
            { label: 'Nearest-neighbour distance', slug: 'primitives/nearest-neighbour-distance' },
            { label: 'Average Nearest Neighbour Index', slug: 'primitives/anni' },
            { label: 'Empty-space function', slug: 'primitives/empty-space' },
            { label: 'KDE density splat', slug: 'primitives/kde-splat' },
            { label: 'Separable convolution', slug: 'primitives/separable-convolution' },
            { label: 'Getis-Ord hotspots', slug: 'primitives/getis-ord-hotspots' },
            { label: 'Fuzzy adjacency', slug: 'primitives/fuzzy-adjacency' },
            { label: 'k-th neighbour distance', slug: 'primitives/kth-neighbour-distance' },
            { label: 'CkNN rescaled distance', slug: 'primitives/cknn' },
          ],
        },
      ],
    }),
  ],
});
