import { ArithmeticOp, ReconstructionMethod, Signal } from './types';
import { fft, nextPow2 } from './DSPTransforms';

export const timeAxis = (s: Signal): number[] => s.re.map((_, i) => s.t0 + i * s.dt);

/** Value at time t: linear interpolation (continuous) or nearest sample (discrete); zero outside support. */
function valueAt(s: Signal, t: number): [number, number] {
  const x = (t - s.t0) / s.dt, n = s.re.length;
  if (x < -1e-9 || x > n - 1 + 1e-9) return [0, 0];
  if (s.domain === 'discrete') {
    const i = Math.min(n - 1, Math.max(0, Math.round(x)));
    return [s.re[i], s.im?.[i] ?? 0];
  }
  const i = Math.min(n - 2, Math.max(0, Math.floor(x))), f = x - i;
  if (n < 2) return [s.re[0], s.im?.[0] ?? 0];
  return [
    s.re[i] * (1 - f) + s.re[i + 1] * f,
    s.im ? s.im[i] * (1 - f) + s.im[i + 1] * f : 0,
  ];
}

/** g(t) = s(map(t)), evaluated on `target`'s grid (defaults to s's own grid). */
export function transformTime(s: Signal, map: (t: number) => number, target: Signal = s): Signal {
  const re: number[] = [], im: number[] | undefined = s.im ? [] : undefined;
  for (let i = 0; i < target.re.length; i++) {
    const [a, b] = valueAt(s, map(target.t0 + i * target.dt));
    re.push(a); im?.push(b);
  }
  return { t0: target.t0, dt: target.dt, re, im, domain: target.domain };
}

export const timeShift = (s: Signal, t0: number) => transformTime(s, (t) => t - t0);       // f(t - t0)
export const timeScale = (s: Signal, a: number) => transformTime(s, (t) => a * t);         // f(at)
export const timeReverse = (s: Signal) => transformTime(s, (t) => -t);                     // f(-t)

export const amplitudeScale = (s: Signal, k: number): Signal => ({
  ...s, re: s.re.map((v) => k * v), im: s.im?.map((v) => k * v),
});
export const dcOffset = (s: Signal, c: number): Signal => ({ ...s, re: s.re.map((v) => v + c) });

/** Direct convolution y[n] = Σ x[k]h[n-k]; FFT-based for large inputs. Continuous mode scales by dt. */
export function convolve(x: Signal, h: Signal): Signal {
  const N = x.re.length, M = h.re.length, L = N + M - 1;
  let y: number[];
  if (N * M <= 2_000_000) {
    y = new Array<number>(L).fill(0);
    for (let n = 0; n < N; n++) { const xv = x.re[n]; if (xv === 0) continue; for (let k = 0; k < M; k++) y[n + k] += xv * h.re[k]; }
  } else {
    const P = nextPow2(L), pad = (a: number[]) => a.concat(new Array<number>(P - a.length).fill(0));
    const zero = new Array<number>(P).fill(0);
    const X = fft(pad(x.re), zero), H = fft(pad(h.re), zero);
    const re = X.re.map((v, i) => v * H.re[i] - X.im[i] * H.im[i]);
    const im = X.re.map((v, i) => v * H.im[i] + X.im[i] * H.re[i]);
    y = fft(re, im, true).re.slice(0, L);
  }
  const scale = x.domain === 'continuous' ? x.dt : 1;
  return { t0: x.t0 + h.t0, dt: x.dt, re: y.map((v) => v * scale), domain: x.domain };
}

/** f1 ∘ f2 on f1's grid. Convolution is cropped back to f1's window. */
export function combine(a: Signal, b: Signal, op: ArithmeticOp): Signal {
  if (op === 'none') return a;
  if (op === 'conv') return transformTime(convolve(a, b), (t) => t, a);
  const bb = b.re.length === a.re.length && b.t0 === a.t0 && b.dt === a.dt ? b : transformTime(b, (t) => t, a);
  const bi = bb.im ?? new Array<number>(a.re.length).fill(0), ai = a.im ?? bi.map(() => 0);
  const hasIm = !!(a.im || bb.im);
  const re: number[] = [], im: number[] = [];
  for (let i = 0; i < a.re.length; i++) {
    if (op === 'add') { re.push(a.re[i] + bb.re[i]); im.push(ai[i] + bi[i]); }
    else if (op === 'sub') { re.push(a.re[i] - bb.re[i]); im.push(ai[i] - bi[i]); }
    else { re.push(a.re[i] * bb.re[i] - ai[i] * bi[i]); im.push(a.re[i] * bi[i] + ai[i] * bb.re[i]); }
  }
  return { ...a, re, im: hasIm ? im : undefined };
}

/** Ideal C/D conversion: evaluate f(t) at t = n/Fs over [tMin, tMax] (true sampling, no interpolation error). */
export function sampleFunction(f: (t: number) => number, tMin: number, tMax: number, fs: number) {
  const Ts = 1 / fs, n0 = Math.ceil(tMin * fs - 1e-9), n1 = Math.floor(tMax * fs + 1e-9);
  const t: number[] = [], v: number[] = [];
  for (let n = n0; n <= n1; n++) { t.push(n * Ts); v.push(f(n * Ts)); }
  return { t, v, Ts };
}

const sinc = (x: number) => (Math.abs(x) < 1e-12 ? 1 : Math.sin(Math.PI * x) / (Math.PI * x));

/** D/C reconstruction: zero-order hold, first-order hold (linear), or ideal sinc interpolation. */
export function reconstruct(ts: number[], v: number[], Ts: number, tOut: number[], method: ReconstructionMethod): number[] {
  if (ts.length === 0) return tOut.map(() => 0);
  const t0 = ts[0], last = ts.length - 1;
  return tOut.map((t) => {
    const x = (t - t0) / Ts;
    if (method === 'sinc') {
      let s = 0;
      for (let i = 0; i <= last; i++) s += v[i] * sinc(x - i);   // x_r(t)=Σ x[n] sinc((t-nTs)/Ts)
      return s;
    }
    if (method === 'zoh') { const k = Math.floor(x + 1e-9); return k < 0 ? 0 : v[Math.min(k, last)]; }
    if (x < 0 || x > last) return 0;                              // foh
    const k = Math.min(last - 1, Math.floor(x)), f = x - k;
    return last === 0 ? v[0] : v[k] * (1 - f) + v[k + 1] * f;
  });
}
