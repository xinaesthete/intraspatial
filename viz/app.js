/*
 * app.js — wiring for the three interactive panels.
 *   Panel 1: 2D subband decomposition explorer.
 *   Panel 2: reconstruction + compression demo.
 *   Panel 3: 1D lifting-scheme animator.
 *
 * Depends on dwt.js (window.DWT) and images.js (window.IMG).
 */
'use strict';

(function () {
  const $ = (id) => document.getElementById(id);

  /* =============================================================== *
   *  Shared rendering helpers.
   * =============================================================== */

  // Render an LL/approximation region (non-negative-ish, scaled to 0..255).
  function autoRangeLL(data, width, x0, y0, w, h) {
    let mn = Infinity, mx = -Infinity;
    for (let y = 0; y < h; y++)
      for (let x = 0; x < w; x++) {
        const v = data[(y0 + y) * width + (x0 + x)];
        if (v < mn) mn = v;
        if (v > mx) mx = v;
      }
    return { mn, mx: mx === mn ? mn + 1 : mx };
  }

  /* Draw a wavelet decomposition into a canvas. LL is shown with a normal
   * grayscale ramp; detail bands are signed, shown around mid-gray with
   * optional log scaling to reveal small coefficients. */
  function renderDecomposition(canvas, dec, opts) {
    const { width, height, data, bands } = dec;
    const { logScale, detailGain, highlight } = opts;
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');
    const img = ctx.createImageData(width, height);
    const out = img.data;

    // The LL region is the last band in the list.
    const ll = bands[bands.length - 1];
    const llRange = autoRangeLL(data, width, ll.x, ll.y, ll.w, ll.h);

    // Estimate a robust detail magnitude scale (per render) for normalization.
    let detailMax = 1e-6;
    for (const b of bands) {
      if (b.type === 'LL') continue;
      for (let y = 0; y < b.h; y++)
        for (let x = 0; x < b.w; x++) {
          const v = Math.abs(data[(b.y + y) * width + (b.x + x)]);
          if (v > detailMax) detailMax = v;
        }
    }

    function shadeDetail(v) {
      const a = Math.abs(v) / detailMax;
      let s = logScale ? Math.log1p(a * 40) / Math.log1p(40) : a;
      s = Math.min(1, s * detailGain);
      // Map signed coefficient around mid-gray; positive warmer, negative cooler.
      const mid = 128;
      const amp = 127 * s;
      if (v >= 0) return [mid + amp, mid + amp * 0.55, mid - amp * 0.2];
      return [mid - amp * 0.2, mid + amp * 0.55, mid + amp];
    }

    // Paint LL.
    for (let y = 0; y < ll.h; y++)
      for (let x = 0; x < ll.w; x++) {
        const v = data[(ll.y + y) * width + (ll.x + x)];
        const g = 255 * (v - llRange.mn) / (llRange.mx - llRange.mn);
        const idx = ((ll.y + y) * width + (ll.x + x)) * 4;
        out[idx] = out[idx + 1] = out[idx + 2] = g;
        out[idx + 3] = 255;
      }
    // Paint detail bands.
    for (const b of bands) {
      if (b.type === 'LL') continue;
      const dim = highlight && !(highlight.level === b.level && highlight.type === b.type);
      for (let y = 0; y < b.h; y++)
        for (let x = 0; x < b.w; x++) {
          const v = data[(b.y + y) * width + (b.x + x)];
          let [r, gg, bb] = shadeDetail(v);
          if (dim) { r *= 0.32; gg *= 0.32; bb *= 0.32; }
          const idx = ((b.y + y) * width + (b.x + x)) * 4;
          out[idx] = r; out[idx + 1] = gg; out[idx + 2] = bb; out[idx + 3] = 255;
        }
    }
    ctx.putImageData(img, 0, 0);

    // Overlay subband grid lines so the pyramid structure reads clearly.
    drawGrid(ctx, dec);
  }

  function drawGrid(ctx, dec) {
    const { width, height, levels } = dec;
    ctx.save();
    ctx.lineWidth = 1;
    let curW = width, curH = height;
    for (let lvl = 0; lvl < levels && curW >= 2 && curH >= 2; lvl++) {
      const lowW = (curW + 1) >> 1, lowH = (curH + 1) >> 1;
      ctx.strokeStyle = 'rgba(90,200,250,0.55)';
      ctx.beginPath();
      ctx.moveTo(lowW + 0.5, 0); ctx.lineTo(lowW + 0.5, curH);
      ctx.moveTo(0, lowH + 0.5); ctx.lineTo(curW, lowH + 0.5);
      ctx.stroke();
      curW = lowW; curH = lowH;
    }
    ctx.restore();
  }

  function renderPlaneGray(canvas, plane, range) {
    const { width, height, data } = plane;
    canvas.width = width; canvas.height = height;
    const ctx = canvas.getContext('2d');
    const img = ctx.createImageData(width, height);
    const mn = range ? range.mn : 0, mx = range ? range.mx : 255;
    for (let i = 0; i < width * height; i++) {
      const g = 255 * (data[i] - mn) / (mx - mn);
      const c = g < 0 ? 0 : g > 255 ? 255 : g;
      img.data[i * 4] = img.data[i * 4 + 1] = img.data[i * 4 + 2] = c;
      img.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
  }

  function renderError(canvas, a, b) {
    const { width, height } = a;
    canvas.width = width; canvas.height = height;
    const ctx = canvas.getContext('2d');
    const img = ctx.createImageData(width, height);
    let mx = 1e-6;
    for (let i = 0; i < width * height; i++) mx = Math.max(mx, Math.abs(a.data[i] - b.data[i]));
    for (let i = 0; i < width * height; i++) {
      const e = Math.abs(a.data[i] - b.data[i]) / mx;
      // blue(low) -> red(high) heat ramp
      img.data[i * 4] = 255 * e;
      img.data[i * 4 + 1] = 40 * e;
      img.data[i * 4 + 2] = 255 * (1 - e) * 0.6;
      img.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
    return mx;
  }

  /* =============================================================== *
   *  STATE
   * =============================================================== */
  const SIZE = 256; // working resolution
  const state = {
    sourceKey: 'fractal',
    plane: null,
    kernel: '9/7',
    levels: 4,
    logScale: true,
    detailGain: 1.0,
    highlight: null, // {level,type}
    dec: null,
  };

  function loadSynthetic(key) {
    state.sourceKey = key;
    state.plane = IMG.SYNTH_IMAGES[key].gen(SIZE, SIZE);
    recompute();
  }

  function recompute() {
    state.dec = DWT.dwt2dForward(state.plane, state.kernel, state.levels);
    renderDecomposition($('decomp-canvas'), state.dec, {
      logScale: state.logScale,
      detailGain: state.detailGain,
      highlight: state.highlight,
    });
    renderPlaneGray($('source-canvas'), state.plane, { mn: 0, mx: 255 });
    updateBandList();
    runCompression(); // keep panel 2 in sync
  }

  /* =============================================================== *
   *  Panel 1 controls
   * =============================================================== */
  function buildSourceButtons() {
    const wrap = $('source-buttons');
    wrap.innerHTML = '';
    for (const [key, info] of Object.entries(IMG.SYNTH_IMAGES)) {
      const b = document.createElement('button');
      b.className = 'chip';
      b.textContent = info.label;
      b.dataset.key = key;
      b.onclick = () => {
        loadSynthetic(key);
        markActive(wrap, b);
      };
      wrap.appendChild(b);
      if (key === state.sourceKey) b.classList.add('active');
    }
  }

  function markActive(wrap, el) {
    wrap.querySelectorAll('.chip').forEach((c) => c.classList.remove('active'));
    el.classList.add('active');
  }

  function updateBandList() {
    const el = $('band-info');
    if (!state.highlight) {
      el.textContent = 'Click a subband in the pyramid to isolate it. LL = approximation; HL/LH/HH = horizontal/vertical/diagonal detail.';
      return;
    }
    const b = state.dec.bands.find(
      (x) => x.level === state.highlight.level && x.type === state.highlight.type
    );
    if (!b) { el.textContent = ''; return; }
    let energy = 0, count = 0, mx = 0;
    for (let y = 0; y < b.h; y++)
      for (let x = 0; x < b.w; x++) {
        const v = state.dec.data[(b.y + y) * state.dec.width + (b.x + x)];
        energy += v * v; count++; mx = Math.max(mx, Math.abs(v));
      }
    const meaning = { LL: 'approximation (coarse image)', HL: 'horizontal detail (vertical edges)', LH: 'vertical detail (horizontal edges)', HH: 'diagonal detail' }[b.type];
    el.innerHTML = `<b>Level ${b.level} · ${b.type}</b> — ${meaning}. ` +
      `${b.w}×${b.h} coeffs · mean energy ${(energy / count).toFixed(2)} · peak |coeff| ${mx.toFixed(1)}.`;
  }

  function bandAtPixel(px, py) {
    // Identify which subband rectangle a click landed in.
    for (const b of state.dec.bands) {
      if (px >= b.x && px < b.x + b.w && py >= b.y && py < b.y + b.h) {
        if (b.type === 'LL') return null; // LL not isolable as "detail"
        return { level: b.level, type: b.type };
      }
    }
    return null;
  }

  /* =============================================================== *
   *  Panel 2 — compression / thresholding
   * =============================================================== */
  function runCompression() {
    if (!state.dec) return;
    const keepPct = parseFloat($('keep-slider').value); // 0..100
    const { data, width, height } = state.dec;
    const n = width * height;

    // Determine a magnitude threshold that keeps the top keepPct% of
    // coefficients by magnitude (i.e. drops the smallest ones).
    const mags = new Float64Array(n);
    for (let i = 0; i < n; i++) mags[i] = Math.abs(data[i]);
    const sorted = Float64Array.from(mags).sort();
    const dropCount = Math.floor(n * (1 - keepPct / 100));
    const thresh = dropCount <= 0 ? -1 : sorted[Math.min(dropCount, n - 1)];

    const kept = new Float64Array(n);
    let nonzero = 0;
    for (let i = 0; i < n; i++) {
      if (mags[i] > thresh) { kept[i] = data[i]; nonzero++; }
    }

    const rec = DWT.dwt2dInverse({ data: kept, width, height, levels: state.dec.levels, kernel: state.dec.kernel });

    // Metrics vs the ORIGINAL source plane.
    const orig = state.plane;
    let mse = 0;
    for (let i = 0; i < n; i++) { const d = rec.data[i] - orig.data[i]; mse += d * d; }
    mse /= n;
    const psnr = mse < 1e-9 ? Infinity : 10 * Math.log10(255 * 255 / mse);
    const sparsity = 100 * (1 - nonzero / n);

    renderPlaneGray($('orig-canvas'), orig, { mn: 0, mx: 255 });
    renderPlaneGray($('recon-canvas'), rec, { mn: 0, mx: 255 });
    renderError($('error-canvas'), orig, rec);

    $('psnr-val').textContent = psnr === Infinity ? '∞ dB (lossless)' : psnr.toFixed(2) + ' dB';
    $('sparsity-val').textContent = sparsity.toFixed(1) + '% zeros';
    $('keep-val').textContent = keepPct.toFixed(keepPct < 1 ? 2 : 1) + '%';
    $('nonzero-val').textContent = nonzero.toLocaleString() + ' / ' + n.toLocaleString() + ' coeffs';
  }

  /* =============================================================== *
   *  Panel 3 — 1D lifting animator
   * =============================================================== */
  const sig1d = { values: null, kernel: '5/3', stage: 0, sub: 0 };

  function genSignal(kind, n = 32) {
    const a = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const t = i / (n - 1);
      switch (kind) {
        case 'step': a[i] = i < n / 2 ? 40 : 200; break;
        case 'ramp': a[i] = 20 + 200 * t; break;
        case 'sine': a[i] = 128 + 90 * Math.sin(2 * Math.PI * 2.5 * t); break;
        case 'noisy': a[i] = 128 + 60 * Math.sin(2 * Math.PI * 1.5 * t) + (Math.random() - 0.5) * 80; break;
      }
    }
    return a;
  }

  function setupSignal(kind) {
    sig1d.values = genSignal(kind);
    sig1d.stage = 0;
    drawLifting();
  }

  // Stages of the lifting walkthrough (for one decomposition level).
  const STAGES = [
    { name: 'Input signal', desc: 'The raw 1D signal x[n] we will transform.' },
    { name: 'Split (lazy wavelet)', desc: 'Separate even-indexed and odd-indexed samples. No math yet — just a polyphase split.' },
    { name: 'Predict', desc: 'Predict each odd sample from its even neighbours and store the residual (detail). For 5/3: d[k] = odd − ⌊(even_L+even_R)/2⌋. Smooth signals ⇒ tiny details.' },
    { name: 'Update', desc: 'Update evens using the new details to form the low-pass (approximation) so it tracks the local average. For 5/3: s[k] = even + ⌊(d_L+d_R+2)/4⌋.' },
    { name: 'Result', desc: 'Approximation (low-pass) + detail (high-pass). Recurse on the low-pass for the next level. Notice how concentrated the detail coefficients are.' },
  ];

  function liftingState() {
    // Compute the intermediate arrays for the current stage on the fly.
    const x = sig1d.values;
    const n = x.length;
    const nLow = (n + 1) >> 1;
    const even = [], odd = [];
    for (let i = 0; i < n; i++) (i % 2 === 0 ? even : odd).push(x[i]);

    const mirror = DWT.mirror;
    // Work on a copy through the lifting steps (5/3 or 9/7).
    const t = Float64Array.from(x);
    const is97 = sig1d.kernel === '9/7';
    const { A, B, G, D, K } = DWT.coeffs;

    function predict(coef) {
      for (let k = 1; k < n; k += 2) {
        const l = t[k - 1];
        const r = k + 1 < n ? t[k + 1] : t[mirror(k + 1, n)];
        if (is97) t[k] += coef * (l + r);
        else t[k] -= Math.floor((l + r) / 2);
      }
    }
    function update(coef) {
      for (let k = 0; k < n; k += 2) {
        const l = k - 1 >= 0 ? t[k - 1] : t[mirror(k - 1, n)];
        const r = k + 1 < n ? t[k + 1] : t[mirror(k + 1, n)];
        if (is97) t[k] += coef * (l + r);
        else t[k] += Math.floor((l + r + 2) / 4);
      }
    }

    if (!is97) {
      // 5/3: one predict + one update.
      const stage = sig1d.stage;
      if (stage >= 3) { predict(); update(); }
      else if (stage === 2) { predict(); }
    } else {
      // 9/7: collapse the 4 lifting steps for the final view.
      if (sig1d.stage >= 3) {
        predict(A); update(B); predict(G); update(D);
        for (let k = 0; k < n; k++) t[k] *= (k % 2 === 0) ? 1 / K : K;
      } else if (sig1d.stage === 2) {
        predict(A);
      }
    }
    const low = [], high = [];
    for (let k = 0; k < n; k++) (k % 2 === 0 ? low : high).push(t[k]);
    return { x, even, odd, low, high, nLow };
  }

  function drawLifting() {
    const cv = $('lift-canvas');
    const W = cv.clientWidth || 720, H = 300;
    cv.width = W * devicePixelRatio; cv.height = H * devicePixelRatio;
    cv.style.height = H + 'px';
    const ctx = cv.getContext('2d');
    ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
    ctx.clearRect(0, 0, W, H);

    const st = liftingState();
    const stage = sig1d.stage;
    const n = sig1d.values.length;
    const pad = 36;
    const plotW = W - pad * 2;
    const x2px = (i) => pad + (plotW * i) / (n - 1);
    // y range
    let mn = Infinity, mx = -Infinity;
    const collect = (arr) => arr.forEach((v) => { mn = Math.min(mn, v); mx = Math.max(mx, v); });
    collect(Array.from(st.x));
    if (stage >= 2) { collect(st.high); }
    if (mx - mn < 1) mx = mn + 1;
    const top = 28, bot = H - 28;
    const y2px = (v) => bot - (bot - top) * (v - mn) / (mx - mn);

    // baseline at value 0 if visible
    ctx.strokeStyle = 'rgba(255,255,255,0.08)';
    ctx.lineWidth = 1;
    if (mn < 0 && mx > 0) {
      ctx.beginPath(); ctx.moveTo(pad, y2px(0)); ctx.lineTo(W - pad, y2px(0)); ctx.stroke();
    }

    function stem(i, v, color, r = 3.5) {
      const px = x2px(i), py = y2px(v), zy = y2px(Math.max(mn, Math.min(0, mx)));
      ctx.strokeStyle = color; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(px, zy); ctx.lineTo(px, py); ctx.stroke();
      ctx.fillStyle = color; ctx.beginPath(); ctx.arc(px, py, r, 0, 7); ctx.fill();
    }

    const EVEN = '#5ac8fa', ODD = '#ff9f43', LOW = '#7bed9f', HIGH = '#ff6b81';

    if (stage === 0) {
      // raw signal as a connected line
      ctx.strokeStyle = '#cdd6e6'; ctx.lineWidth = 2; ctx.beginPath();
      for (let i = 0; i < n; i++) { const px = x2px(i), py = y2px(st.x[i]); i ? ctx.lineTo(px, py) : ctx.moveTo(px, py); }
      ctx.stroke();
      for (let i = 0; i < n; i++) stem(i, st.x[i], '#cdd6e6', 2.5);
    } else if (stage === 1) {
      // split: color evens vs odds
      for (let i = 0; i < n; i++) stem(i, st.x[i], i % 2 === 0 ? EVEN : ODD);
    } else if (stage === 2) {
      // after predict: evens unchanged (approx-to-be), odds now details
      for (let k = 0; k < n; k++) stem(k, st.x[k] === undefined ? 0 : (k % 2 === 0 ? st.x[k] : st.high[(k - 1) / 2]), k % 2 === 0 ? EVEN : HIGH);
    } else {
      // final: lows and highs side by side in coefficient order
      for (let j = 0; j < st.low.length; j++) stem(j, st.low[j], LOW);
      const off = st.nLow;
      for (let j = 0; j < st.high.length; j++) stem(off + j, st.high[j], HIGH);
      // divider between low and high blocks
      ctx.strokeStyle = 'rgba(255,255,255,0.25)';
      ctx.setLineDash([4, 4]);
      const dx = x2px(off - 0.5);
      ctx.beginPath(); ctx.moveTo(dx, top - 6); ctx.lineTo(dx, bot + 6); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = LOW; ctx.font = '12px ui-monospace,monospace';
      ctx.fillText('approximation (low-pass)', pad, top - 12);
      ctx.fillStyle = HIGH;
      ctx.fillText('detail (high-pass)', dx + 8, top - 12);
    }

    // stage caption
    $('lift-stage-name').textContent = `${stage + 1}/${STAGES.length} · ${STAGES[stage].name}`;
    $('lift-stage-desc').textContent = STAGES[stage].desc;
    $('lift-kernel-note').textContent = sig1d.kernel === '5/3'
      ? 'Integer 5/3: predict then update, all exact.'
      : '9/7: 4 lifting steps (α,β,γ,δ) + scaling, collapsed into the Result view.';
  }

  /* =============================================================== *
   *  Wiring
   * =============================================================== */
  function init() {
    buildSourceButtons();

    $('kernel-select').value = state.kernel;
    $('levels-slider').value = state.levels;
    $('levels-val').textContent = state.levels;
    $('log-toggle').checked = state.logScale;
    $('gain-slider').value = state.detailGain;

    $('kernel-select').onchange = (e) => { state.kernel = e.target.value; recompute(); };
    $('levels-slider').oninput = (e) => {
      state.levels = parseInt(e.target.value, 10);
      $('levels-val').textContent = state.levels;
      state.highlight = null;
      recompute();
    };
    $('log-toggle').onchange = (e) => { state.logScale = e.target.checked; recompute(); };
    $('gain-slider').oninput = (e) => {
      state.detailGain = parseFloat(e.target.value);
      $('gain-val').textContent = state.detailGain.toFixed(1) + '×';
      renderDecomposition($('decomp-canvas'), state.dec, { logScale: state.logScale, detailGain: state.detailGain, highlight: state.highlight });
    };
    $('gain-val').textContent = state.detailGain.toFixed(1) + '×';

    // click-to-isolate subband
    $('decomp-canvas').addEventListener('click', (ev) => {
      const cv = $('decomp-canvas');
      const rect = cv.getBoundingClientRect();
      const px = Math.floor((ev.clientX - rect.left) / rect.width * cv.width);
      const py = Math.floor((ev.clientY - rect.top) / rect.height * cv.height);
      const b = bandAtPixel(px, py);
      // toggle off if clicking the same band
      if (b && state.highlight && b.level === state.highlight.level && b.type === state.highlight.type) {
        state.highlight = null;
      } else {
        state.highlight = b;
      }
      renderDecomposition($('decomp-canvas'), state.dec, { logScale: state.logScale, detailGain: state.detailGain, highlight: state.highlight });
      updateBandList();
    });
    $('clear-highlight').onclick = () => {
      state.highlight = null;
      renderDecomposition($('decomp-canvas'), state.dec, { logScale: state.logScale, detailGain: state.detailGain, highlight: state.highlight });
      updateBandList();
    };

    // upload
    $('upload-input').onchange = (e) => {
      const file = e.target.files[0];
      if (!file) return;
      const url = URL.createObjectURL(file);
      const img = new Image();
      img.onload = () => {
        state.plane = IMG.imageElementToPlane(img, SIZE);
        state.sourceKey = 'upload';
        state.highlight = null;
        markActive($('source-buttons'), document.createElement('span')); // clear actives
        recompute();
        URL.revokeObjectURL(url);
      };
      img.src = url;
    };

    // Panel 2
    $('keep-slider').oninput = runCompression;
    document.querySelectorAll('[data-keep]').forEach((b) => {
      b.onclick = () => { $('keep-slider').value = b.dataset.keep; runCompression(); };
    });

    // Panel 3
    $('sig-select').onchange = (e) => setupSignal(e.target.value);
    $('lift-kernel').onchange = (e) => { sig1d.kernel = e.target.value; drawLifting(); };
    $('lift-prev').onclick = () => { sig1d.stage = Math.max(0, sig1d.stage - 1); drawLifting(); };
    $('lift-next').onclick = () => { sig1d.stage = Math.min(STAGES.length - 1, sig1d.stage + 1); drawLifting(); };
    let playTimer = null;
    $('lift-play').onclick = () => {
      if (playTimer) { clearInterval(playTimer); playTimer = null; $('lift-play').textContent = '▶ Play'; return; }
      $('lift-play').textContent = '⏸ Pause';
      playTimer = setInterval(() => {
        sig1d.stage = (sig1d.stage + 1) % STAGES.length;
        drawLifting();
      }, 1100);
    };

    // collapsible explainer
    document.querySelectorAll('.collapsible > summary').forEach(() => {});

    // initial data
    loadSynthetic(state.sourceKey);
    sig1d.kernel = '5/3';
    setupSignal('step');
    $('sig-select').value = 'step';
    $('lift-kernel').value = '5/3';

    // round-trip badge in the header
    const res = DWT.selfTest();
    $('selftest-badge').textContent =
      `round-trip ✓  5/3 err=${res['5/3']}  ·  9/7 err≈${res['9/7'].toExponential(1)}`;

    window.addEventListener('resize', () => drawLifting());
  }

  document.addEventListener('DOMContentLoaded', init);
})();
