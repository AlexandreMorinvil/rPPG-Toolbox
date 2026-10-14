'use strict';
// Minimal canvas charts: line/point/marker series, reference lines, shaded spans, heatmaps, hover readout.
(function () {
  const COLORS = ['#e5484d', '#3e63dd', '#30a46c', '#f76b15', '#8e4ec6', '#12a594', '#ad7f58'];

  function niceStep(span, target) {
    const raw = span / Math.max(target, 1);
    const magnitude = Math.pow(10, Math.floor(Math.log10(raw || 1)));
    const residual = raw / magnitude;
    const nice = residual > 5 ? 10 : residual > 2 ? 5 : residual > 1 ? 2 : 1;
    return nice * magnitude;
  }

  function ticks(min, max, target) {
    if (!(max > min)) return [min];
    const step = niceStep(max - min, target);
    const result = [];
    for (let v = Math.ceil(min / step) * step; v <= max + step * 1e-9; v += step) result.push(+v.toPrecision(12));
    return result;
  }

  function format(v) {
    if (v === null || v === undefined || !Number.isFinite(v)) return '–';
    const a = Math.abs(v);
    if (a !== 0 && (a < 1e-3 || a >= 1e5)) return v.toExponential(2);
    return +v.toFixed(a < 1 ? 4 : a < 100 ? 2 : 1) + '';
  }

  function extent(series, axis, fixedMin, fixedMax) {
    let min = Infinity, max = -Infinity;
    for (const s of series) {
      for (const v of s[axis]) {
        if (v === null || !Number.isFinite(v)) continue;
        if (v < min) min = v;
        if (v > max) max = v;
      }
    }
    if (fixedMin !== undefined) min = fixedMin;
    if (fixedMax !== undefined) max = fixedMax;
    if (!Number.isFinite(min) || !Number.isFinite(max)) return [0, 1];
    if (min === max) { min -= 1; max += 1; }
    return [min, max];
  }

  // Sequential color scale (viridis stops) for heatmaps.
  const HEAT = [[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]];
  function heatColor(v) {
    const t = Math.min(1, Math.max(0, Number.isFinite(v) ? v : 0)) * (HEAT.length - 1);
    const i = Math.min(HEAT.length - 2, Math.floor(t)), f = t - i;
    return HEAT[i].map((c, k) => Math.round(c + (HEAT[i + 1][k] - c) * f));
  }

  function drawHeatmap(ctx, heat, toX, toY) {
    const nx = heat.x.length, ny = heat.y.length;
    if (!nx || !ny) return;
    const image = document.createElement('canvas');
    image.width = nx; image.height = ny;
    const ictx = image.getContext('2d');
    const pixels = ictx.createImageData(nx, ny);
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        const [r, g, b] = heatColor(heat.z[j][i]);
        const o = ((ny - 1 - j) * nx + i) * 4;
        pixels.data[o] = r; pixels.data[o + 1] = g; pixels.data[o + 2] = b; pixels.data[o + 3] = 255;
      }
    }
    ictx.putImageData(pixels, 0, 0);
    const dx = nx > 1 ? (heat.x[nx - 1] - heat.x[0]) / (nx - 1) : 1;
    const dy = ny > 1 ? (heat.y[ny - 1] - heat.y[0]) / (ny - 1) : 1;
    const x0 = toX(heat.x[0] - dx / 2), x1 = toX(heat.x[nx - 1] + dx / 2);
    const y0 = toY(heat.y[ny - 1] + dy / 2), y1 = toY(heat.y[0] - dy / 2);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(image, x0, y0, x1 - x0, y1 - y0);
  }

  function marker(ctx, x, y, shape, size) {
    ctx.beginPath();
    if (shape === 'down') { ctx.moveTo(x - size, y - size); ctx.lineTo(x + size, y - size); ctx.lineTo(x, y + size * 0.6); }
    else if (shape === 'up') { ctx.moveTo(x - size, y + size); ctx.lineTo(x + size, y + size); ctx.lineTo(x, y - size * 0.6); }
    else ctx.arc(x, y, size, 0, Math.PI * 2);
    ctx.closePath(); ctx.fill();
  }

  function draw(canvas, options) {
    const opts = Object.assign({ series: [], height: 220, padding: [12, 14, 34, 52] }, options);
    const dpr = window.devicePixelRatio || 1;
    const width = canvas.clientWidth || canvas.parentElement.clientWidth || 600;
    const height = opts.height;
    canvas.style.height = height + 'px';
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);
    const style = getComputedStyle(document.body);
    const ink = style.getPropertyValue('--ink').trim() || '#222';
    const grid = style.getPropertyValue('--grid').trim() || '#e5e5e5';
    const muted = style.getPropertyValue('--muted').trim() || '#777';
    const [top, right, bottom, left] = opts.padding;
    const plot = { x: left, y: top, w: Math.max(10, width - left - right), h: Math.max(10, height - top - bottom) };

    const series = opts.series.map((s, i) => Object.assign({ color: COLORS[i % COLORS.length], type: 'line' }, s));
    const heat = opts.heatmap && opts.heatmap.x.length && opts.heatmap.y.length ? opts.heatmap : null;
    const hasData = heat || series.some(s => s.y.some(v => v !== null && Number.isFinite(v)));
    if (!hasData) {
      ctx.fillStyle = muted;
      ctx.font = '13px system-ui, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(opts.empty || 'No data', width / 2, height / 2);
      canvas._chart = null;
      return;
    }
    const ranged = heat ? series.concat([{ x: heat.x, y: heat.y }]) : series;
    let [xMin, xMax] = extent(ranged, 'x', opts.xMin, opts.xMax);
    let [yMin, yMax] = extent(ranged, 'y', opts.yMin, opts.yMax);
    if (opts.identity) {
      const lo = Math.min(xMin, yMin), hi = Math.max(xMax, yMax);
      xMin = yMin = lo; xMax = yMax = hi;
    }
    for (const line of opts.hlines || []) { yMin = Math.min(yMin, line.y); yMax = Math.max(yMax, line.y); }
    if (opts.yMin === undefined && opts.yMax === undefined && !heat) {
      const pad = (yMax - yMin) * 0.06;
      yMin -= pad; yMax += pad;
    }
    const toX = v => plot.x + (v - xMin) / (xMax - xMin) * plot.w;
    const toY = v => plot.y + plot.h - (v - yMin) / (yMax - yMin) * plot.h;

    ctx.font = '11px system-ui, sans-serif';
    ctx.strokeStyle = grid;
    ctx.fillStyle = muted;
    ctx.lineWidth = 1;
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    for (const t of ticks(yMin, yMax, plot.h / 40)) {
      const y = toY(t);
      ctx.beginPath(); ctx.moveTo(plot.x, y); ctx.lineTo(plot.x + plot.w, y); ctx.stroke();
      ctx.fillText(format(t), plot.x - 6, y);
    }
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    for (const t of ticks(xMin, xMax, plot.w / 80)) {
      const x = toX(t);
      ctx.beginPath(); ctx.moveTo(x, plot.y); ctx.lineTo(x, plot.y + plot.h); ctx.stroke();
      ctx.fillText(format(t), x, plot.y + plot.h + 5);
    }
    ctx.strokeStyle = ink;
    ctx.strokeRect(plot.x, plot.y, plot.w, plot.h);
    if (opts.xLabel) { ctx.fillStyle = muted; ctx.fillText(opts.xLabel, plot.x + plot.w / 2, height - 14); }
    if (opts.yLabel) {
      ctx.save(); ctx.translate(12, plot.y + plot.h / 2); ctx.rotate(-Math.PI / 2);
      ctx.textBaseline = 'middle'; ctx.fillText(opts.yLabel, 0, 0); ctx.restore();
    }

    ctx.save();
    ctx.beginPath(); ctx.rect(plot.x, plot.y, plot.w, plot.h); ctx.clip();
    if (heat) drawHeatmap(ctx, heat, toX, toY);
    for (const span of opts.vspans || []) {
      ctx.globalAlpha = span.alpha ?? 0.15; ctx.fillStyle = span.color || muted;
      const x0 = toX(span.x0), x1 = toX(span.x1);
      ctx.fillRect(x0, plot.y, Math.max(1, x1 - x0), plot.h);
      ctx.globalAlpha = 1;
    }
    for (const band of opts.hbands || []) {
      ctx.globalAlpha = band.alpha ?? 0.12; ctx.fillStyle = band.color || muted;
      const y0 = toY(band.y1), y1 = toY(band.y0);
      ctx.fillRect(plot.x, y0, plot.w, y1 - y0);
      ctx.globalAlpha = 1;
    }
    if (opts.identity) {
      ctx.strokeStyle = muted; ctx.setLineDash([5, 4]);
      ctx.beginPath(); ctx.moveTo(toX(xMin), toY(xMin)); ctx.lineTo(toX(xMax), toY(xMax)); ctx.stroke();
      ctx.setLineDash([]);
    }
    for (const line of opts.hlines || []) {
      ctx.strokeStyle = line.color || muted; ctx.setLineDash(line.dash || []);
      ctx.beginPath(); ctx.moveTo(plot.x, toY(line.y)); ctx.lineTo(plot.x + plot.w, toY(line.y)); ctx.stroke();
      ctx.setLineDash([]);
      if (line.label) { ctx.fillStyle = line.color || muted; ctx.textAlign = 'left'; ctx.textBaseline = 'bottom'; ctx.fillText(line.label, plot.x + 4, toY(line.y) - 2); }
    }
    for (const line of opts.vlines || []) {
      ctx.strokeStyle = line.color || ink; ctx.setLineDash(line.dash || []);
      ctx.beginPath(); ctx.moveTo(toX(line.x), plot.y); ctx.lineTo(toX(line.x), plot.y + plot.h); ctx.stroke();
      ctx.setLineDash([]);
    }
    for (const s of series) {
      ctx.strokeStyle = s.color; ctx.fillStyle = s.color; ctx.lineWidth = s.width || 1.5;
      if (s.type === 'marks') {
        for (let i = 0; i < s.x.length; i++) {
          if (!Number.isFinite(s.x[i]) || !Number.isFinite(s.y[i])) continue;
          marker(ctx, toX(s.x[i]), toY(s.y[i]), s.shape, s.radius || 4);
        }
      } else if (s.type === 'points') {
        for (let i = 0; i < s.x.length; i++) {
          if (!Number.isFinite(s.x[i]) || !Number.isFinite(s.y[i])) continue;
          ctx.globalAlpha = s.alpha || 0.7;
          ctx.beginPath(); ctx.arc(toX(s.x[i]), toY(s.y[i]), s.radius || 3, 0, Math.PI * 2); ctx.fill();
        }
        ctx.globalAlpha = 1;
      } else {
        ctx.setLineDash(s.dash || []);
        ctx.beginPath();
        let pen = false;
        for (let i = 0; i < s.x.length; i++) {
          const ok = Number.isFinite(s.x[i]) && Number.isFinite(s.y[i]);
          if (!ok) { pen = false; continue; }
          if (pen) ctx.lineTo(toX(s.x[i]), toY(s.y[i])); else ctx.moveTo(toX(s.x[i]), toY(s.y[i]));
          pen = true;
        }
        ctx.stroke();
        ctx.setLineDash([]);
        if (s.markers) {
          for (let i = 0; i < s.x.length; i++) {
            if (!Number.isFinite(s.y[i])) continue;
            ctx.beginPath(); ctx.arc(toX(s.x[i]), toY(s.y[i]), 2.5, 0, Math.PI * 2); ctx.fill();
          }
        }
      }
    }
    ctx.restore();

    if (opts.legend !== false && series.some(s => s.label)) {
      ctx.textAlign = 'left'; ctx.textBaseline = 'middle'; ctx.font = '11px system-ui, sans-serif';
      let x = plot.x + 8;
      for (const s of series.filter(s => s.label)) {
        ctx.fillStyle = s.color; ctx.fillRect(x, plot.y + 8, 10, 3);
        ctx.fillStyle = ink; ctx.fillText(s.label, x + 14, plot.y + 10);
        x += ctx.measureText(s.label).width + 30;
      }
    }
    canvas._chart = { opts, series, plot, xMin, xMax, yMin, yMax, toX, toY, heat };
    canvas.style.cursor = opts.onClick ? 'pointer' : '';
    if (!canvas._hover) {
      canvas._hover = true;
      canvas.addEventListener('mousemove', event => hover(canvas, event));
      canvas.addEventListener('mouseleave', () => document.getElementById('chartTip').classList.add('hidden'));
      canvas.addEventListener('click', event => {
        const c = canvas._chart; if (!c || !c.opts.onClick) return;
        const rect = canvas.getBoundingClientRect();
        const x = c.xMin + (event.clientX - rect.left - c.plot.x) / c.plot.w * (c.xMax - c.xMin);
        c.opts.onClick(x);
      });
    }
  }

  // Save a chart as PNG on the page background (the canvas itself is transparent).
  function download(canvas, name) {
    const out = document.createElement('canvas');
    out.width = canvas.width; out.height = canvas.height;
    const ctx = out.getContext('2d');
    ctx.fillStyle = getComputedStyle(document.body).backgroundColor || '#fff';
    ctx.fillRect(0, 0, out.width, out.height);
    ctx.drawImage(canvas, 0, 0);
    const link = document.createElement('a');
    link.download = (name || 'chart').replace(/[^A-Za-z0-9._-]+/g, '_') + '.png';
    link.href = out.toDataURL('image/png');
    link.click();
  }

  function hover(canvas, event) {
    const c = canvas._chart;
    const tip = document.getElementById('chartTip');
    if (!c) { tip.classList.add('hidden'); return; }
    const rect = canvas.getBoundingClientRect();
    const px = event.clientX - rect.left, py = event.clientY - rect.top;
    if (px < c.plot.x || px > c.plot.x + c.plot.w || py < c.plot.y || py > c.plot.y + c.plot.h) {
      tip.classList.add('hidden'); return;
    }
    const lines = [];
    const xValue = c.xMin + (px - c.plot.x) / c.plot.w * (c.xMax - c.xMin);
    const pointSeries = c.series.filter(s => s.type === 'points');
    if (c.heat && !pointSeries.length) {
      const yValue = c.yMin + (c.plot.y + c.plot.h - py) / c.plot.h * (c.yMax - c.yMin);
      const nearest = (values, v) => values.reduce((b, x, i) => Math.abs(x - v) < Math.abs(values[b] - v) ? i : b, 0);
      const i = nearest(c.heat.x, xValue), j = nearest(c.heat.y, yValue);
      lines.push(`${c.opts.xLabel || 'x'}: ${format(c.heat.x[i])}\n${c.opts.yLabel || 'y'}: ${format(c.heat.y[j])}\npower: ${format(c.heat.z[j][i])}`);
      tip.textContent = lines.join('\n');
      tip.style.left = (event.clientX + 14) + 'px';
      tip.style.top = (event.clientY + 14) + 'px';
      tip.classList.remove('hidden');
      return;
    }
    if (pointSeries.length) {
      let best = null;
      for (const s of pointSeries) {
        for (let i = 0; i < s.x.length; i++) {
          const d = Math.hypot(c.toX(s.x[i]) - px, c.toY(s.y[i]) - py);
          if (d < 12 && (!best || d < best.d)) best = { d, s, i };
        }
      }
      if (!best) { tip.classList.add('hidden'); return; }
      const label = best.s.labels ? best.s.labels[best.i] + '\n' : '';
      lines.push(label + `${c.opts.xLabel || 'x'}: ${format(best.s.x[best.i])}\n${c.opts.yLabel || 'y'}: ${format(best.s.y[best.i])}`);
    } else {
      lines.push(`${c.opts.xLabel || 'x'}: ${format(xValue)}`);
      for (const s of c.series) {
        if (!s.x.length || s.type === 'marks' || s.hover === false) continue;
        let lo = 0, hi = s.x.length - 1;
        while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (s.x[mid] < xValue) lo = mid; else hi = mid; }
        const i = Math.abs(s.x[lo] - xValue) < Math.abs(s.x[hi] - xValue) ? lo : hi;
        lines.push(`${s.label || 'y'}: ${format(s.y[i])}`);
      }
    }
    tip.textContent = lines.join('\n');
    tip.style.left = (event.clientX + 14) + 'px';
    tip.style.top = (event.clientY + 14) + 'px';
    tip.classList.remove('hidden');
  }

  window.Charts = { draw, download, format, COLORS };
})();
