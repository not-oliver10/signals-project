import { useEffect, useMemo, useRef, useState } from 'react';

export interface Trace {
  x: number[]; y: number[]; color: string; label: string;
  mode?: 'line' | 'stem' | 'step'; dashed?: boolean; width?: number;
}
interface Props {
  traces: Trace[]; title?: string; xLabel?: string; yLabel?: string;
  xLog?: boolean; xRange?: [number, number]; height?: number;
}

const M = { l: 58, r: 14, t: 12, b: 34 };

function niceTicks(lo: number, hi: number, count: number): number[] {
  const span = hi - lo;
  if (!(span > 0)) return [lo];
  const raw = span / count, mag = Math.pow(10, Math.floor(Math.log10(raw))), n = raw / mag;
  const step = (n < 1.5 ? 1 : n < 3 ? 2 : n < 7 ? 5 : 10) * mag, out: number[] = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(+v.toPrecision(12));
  return out;
}
const fmt = (v: number) => { const a = Math.abs(v); return a !== 0 && (a < 1e-3 || a >= 1e5) ? v.toExponential(1) : String(+v.toPrecision(4)); };

function nearest(t: Trace, x: number): number {
  let lo = 0, hi = t.x.length - 1;
  while (hi - lo > 1) { const m = (lo + hi) >> 1; if (t.x[m] < x) lo = m; else hi = m; }
  return Math.abs(t.x[lo] - x) < Math.abs(t.x[hi] - x) ? lo : hi;
}

/** Canvas scope: grid, wheel-zoom, drag-pan, hover readout, two click-placed cursors, double-click to reset. */
export function SignalPlotter({ traces, title, xLabel = 't (s)', yLabel = '', xLog = false, xRange, height = 260 }: Props) {
  const wrap = useRef<HTMLDivElement>(null), cv = useRef<HTMLCanvasElement>(null);
  const [w, setW] = useState(600);
  const [view, setView] = useState<[number, number] | null>(null);
  const [hover, setHover] = useState<number | null>(null);
  const [cursors, setCursors] = useState<number[]>([]);
  const drag = useRef<{ px: number; v: [number, number]; moved: boolean } | null>(null);
  const tx = (x: number) => (xLog ? Math.log10(Math.max(x, 1e-12)) : x);
  const ix = (v: number) => (xLog ? Math.pow(10, v) : v);

  const auto = useMemo<[number, number]>(() => {
    if (xRange) return [tx(xRange[0]), tx(xRange[1])];
    let lo = Infinity, hi = -Infinity;
    for (const t of traces) for (const x of t.x) { if (xLog && x <= 0) continue; const v = tx(x); if (v < lo) lo = v; if (v > hi) hi = v; }
    return lo < hi ? [lo, hi] : [0, 1];
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [traces, xRange?.[0], xRange?.[1], xLog]);
  useEffect(() => { setView(null); setCursors([]); }, [auto[0], auto[1]]);

  const v = view ?? auto, pw = w - M.l - M.r, ph = height - M.t - M.b;
  const px2x = (px: number) => v[0] + ((px - M.l) / pw) * (v[1] - v[0]);

  useEffect(() => {
    const el = wrap.current; if (!el) return;
    const ro = new ResizeObserver(() => setW(Math.max(280, el.clientWidth)));
    ro.observe(el); setW(Math.max(280, el.clientWidth));
    return () => ro.disconnect();
  }, []);

  useEffect(() => {  // non-passive wheel listener so the page doesn't scroll while zooming
    const c = cv.current; if (!c) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const rect = c.getBoundingClientRect(), f = Math.exp(e.deltaY * 0.0015);
      const cur = view ?? auto, vx = cur[0] + ((e.clientX - rect.left - M.l) / pw) * (cur[1] - cur[0]);
      setView([vx - (vx - cur[0]) * f, vx + (cur[1] - vx) * f]);
    };
    c.addEventListener('wheel', onWheel, { passive: false });
    return () => c.removeEventListener('wheel', onWheel);
  }, [view, auto, pw]);

  // ---- drawing ----
  useEffect(() => {
    const c = cv.current, el = wrap.current; if (!c || !el) return;
    const dpr = window.devicePixelRatio || 1;
    c.width = w * dpr; c.height = height * dpr;
    const g = c.getContext('2d')!; g.setTransform(dpr, 0, 0, dpr, 0, 0);
    const css = getComputedStyle(el), col = (n: string) => css.getPropertyValue(n).trim();
    g.clearRect(0, 0, w, height);

    // y autoscale on visible data, robust to δ spikes
    const vals: number[] = [];
    for (const t of traces) for (let i = 0; i < t.x.length; i++) { const xv = tx(t.x[i]); if (xv >= v[0] && xv <= v[1] && isFinite(t.y[i])) vals.push(t.y[i]); }
    vals.sort((a, b) => a - b);
    let lo = vals[0] ?? -1, hi = vals[vals.length - 1] ?? 1;
    if (vals.length > 20) {
      const q = (p: number) => vals[Math.floor(p * (vals.length - 1))], plo = q(0.01), phi = q(0.99);
      if (hi - lo > 15 * (phi - plo)) { lo = Math.min(plo, -0.25); hi = Math.max(phi, 1); }
    }
    if (hi - lo < 1e-9) { lo -= 1; hi += 1; }
    const pad = (hi - lo) * 0.08; lo -= pad; hi += pad;

    const X = (x: number) => M.l + ((tx(x) - v[0]) / (v[1] - v[0])) * pw;
    const Y = (y: number) => M.t + (1 - (y - lo) / (hi - lo)) * ph;

    // grid + ticks
    g.font = '11px "Instrument Sans", system-ui, sans-serif'; g.fillStyle = col('--muted'); g.lineWidth = 1;
    const xt: { v: number; major: boolean }[] = [];
    if (xLog) { for (let d = Math.floor(v[0]); d <= Math.ceil(v[1]); d++) for (let m = 1; m < 10; m++) { const lv = Math.log10(m * 10 ** d); if (lv >= v[0] && lv <= v[1]) xt.push({ v: lv, major: m === 1 }); } }
    else niceTicks(v[0], v[1], Math.max(3, Math.floor(pw / 90))).forEach((t) => xt.push({ v: t, major: true }));
    g.textAlign = 'center';
    for (const t of xt) {
      const x = M.l + ((t.v - v[0]) / (v[1] - v[0])) * pw;
      g.strokeStyle = t.major ? col('--grid') : col('--grid-minor'); g.beginPath(); g.moveTo(x, M.t); g.lineTo(x, M.t + ph); g.stroke();
      if (t.major) g.fillText(fmt(ix(t.v)), x, height - M.b + 14);
    }
    g.textAlign = 'right';
    for (const t of niceTicks(lo, hi, Math.max(3, Math.floor(ph / 45)))) {
      const y = Y(t);
      g.strokeStyle = t === 0 ? col('--axis') : col('--grid'); g.beginPath(); g.moveTo(M.l, y); g.lineTo(M.l + pw, y); g.stroke();
      g.fillText(fmt(t), M.l - 6, y + 4);
    }
    g.textAlign = 'center'; g.fillText(xLabel, M.l + pw / 2, height - 4);
    if (yLabel) { g.save(); g.translate(12, M.t + ph / 2); g.rotate(-Math.PI / 2); g.fillText(yLabel, 0, 0); g.restore(); }

    // traces
    g.save(); g.beginPath(); g.rect(M.l, M.t, pw, ph); g.clip();
    for (const t of traces) {
      g.strokeStyle = t.color; g.fillStyle = t.color; g.lineWidth = t.width ?? 1.8; g.setLineDash(t.dashed ? [6, 4] : []);
      const mode = t.mode ?? 'line', y0 = Y(0); let started = false, visible = 0;
      for (let i = 0; i < t.x.length; i++) if (tx(t.x[i]) >= v[0] && tx(t.x[i]) <= v[1]) visible++;
      const dots = pw / Math.max(visible, 1) > 7;
      g.beginPath();
      for (let i = 0; i < t.x.length; i++) {
        const xp = X(t.x[i]), yp = Y(t.y[i]);
        if (xp < M.l - 40 || xp > M.l + pw + 40 || !isFinite(yp)) { started = false; continue; }
        if (mode === 'stem') { g.moveTo(xp, y0); g.lineTo(xp, yp); }
        else if (mode === 'step') { if (!started) g.moveTo(xp, yp); else { g.lineTo(xp, Y(t.y[i - 1])); g.lineTo(xp, yp); } if (i === t.x.length - 1) g.lineTo(xp + (xp - X(t.x[i - 1])), yp); started = true; }
        else { if (!started) g.moveTo(xp, yp); else g.lineTo(xp, yp); started = true; }
      }
      g.stroke();
      if (mode === 'stem' && dots) for (let i = 0; i < t.x.length; i++) { const xp = X(t.x[i]); if (xp < M.l || xp > M.l + pw) continue; g.beginPath(); g.arc(xp, Y(t.y[i]), 3, 0, 7); g.fill(); }
    }
    g.setLineDash([]);
    cursors.forEach((cx, i) => { const x = M.l + ((cx - v[0]) / (v[1] - v[0])) * pw; g.strokeStyle = col('--cursor'); g.setLineDash([4, 3]); g.beginPath(); g.moveTo(x, M.t); g.lineTo(x, M.t + ph); g.stroke(); g.setLineDash([]); g.fillStyle = col('--cursor'); g.fillText(i ? 'B' : 'A', x + 8, M.t + 12); });
    if (hover != null) { const x = M.l + ((hover - v[0]) / (v[1] - v[0])) * pw; g.strokeStyle = col('--axis'); g.beginPath(); g.moveTo(x, M.t); g.lineTo(x, M.t + ph); g.stroke(); }
    g.restore();

    // legend
    g.textAlign = 'left'; let lx = M.l + 8;
    for (const t of traces) { g.fillStyle = t.color; g.fillRect(lx, M.t + 7, 12, 3); g.fillStyle = col('--ink'); g.fillText(t.label, lx + 17, M.t + 12); lx += 28 + g.measureText(t.label).width; }
  });

  const readout = (xv: number) => traces.map((t) => { const i = nearest(t, ix(xv)); return `${t.label}: ${fmt(t.y[i])}`; }).join('   ');
  const pos = (e: React.PointerEvent) => e.clientX - cv.current!.getBoundingClientRect().left;

  return (
    <figure className="plot" ref={wrap}>
      {title && <figcaption>{title}</figcaption>}
      <canvas ref={cv} style={{ width: w, height, touchAction: 'none', cursor: drag.current ? 'grabbing' : 'crosshair' }}
        onPointerDown={(e) => { cv.current!.setPointerCapture(e.pointerId); drag.current = { px: pos(e), v, moved: false }; }}
        onPointerMove={(e) => {
          const px = pos(e), d = drag.current;
          if (d && Math.abs(px - d.px) > 3) { d.moved = true; const dv = ((px - d.px) / pw) * (d.v[1] - d.v[0]); setView([d.v[0] - dv, d.v[1] - dv]); }
          setHover(px2x(px));
        }}
        onPointerUp={(e) => { const d = drag.current; drag.current = null; if (d && !d.moved) setCursors((c) => (c.length >= 2 ? [px2x(pos(e))] : [...c, px2x(pos(e))])); }}
        onPointerLeave={() => setHover(null)}
        onDoubleClick={() => { setView(null); setCursors([]); }} />
      <div className="readout">
        {hover != null ? readout(hover) : 'Hover to read values. Scroll to zoom, drag to pan, click to place cursors A and B, double-click to reset.'}
        {cursors.length === 2 && <b>   Δx = {fmt(ix(cursors[1]) - ix(cursors[0]))}</b>}
      </div>
    </figure>
  );
}
