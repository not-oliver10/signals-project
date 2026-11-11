import { Signal, Spectrum, WindowType } from './types';

export const nextPow2 = (n: number) => { let p = 1; while (p < n) p <<= 1; return p; };

export function logSpace(a: number, b: number, n: number): number[] {
  const la = Math.log10(a), lb = Math.log10(b);
  return Array.from({ length: n }, (_, i) => Math.pow(10, la + ((lb - la) * i) / (n - 1)));
}

export function hann(n: number): number[] {
  return Array.from({ length: n }, (_, i) => 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / Math.max(n - 1, 1)));
}

/** Iterative radix-2 Cooley-Tukey FFT. Length must be a power of two. */
export function fft(re: number[], im: number[], inverse = false): { re: number[]; im: number[] } {
  const n = re.length;
  const r = re.slice(), i = im.slice();
  for (let a = 1, b = 0; a < n; a++) {
    let bit = n >> 1;
    for (; b & bit; bit >>= 1) b ^= bit;
    b ^= bit;
    if (a < b) { [r[a], r[b]] = [r[b], r[a]]; [i[a], i[b]] = [i[b], i[a]]; }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = ((2 * Math.PI) / len) * (inverse ? 1 : -1);
    const wr = Math.cos(ang), wi = Math.sin(ang), half = len >> 1;
    for (let s = 0; s < n; s += len) {
      let cr = 1, ci = 0;
      for (let k = 0; k < half; k++) {
        const a = s + k, b = a + half;
        const xr = r[b] * cr - i[b] * ci, xi = r[b] * ci + i[b] * cr;
        r[b] = r[a] - xr; i[b] = i[a] - xi; r[a] += xr; i[a] += xi;
        const ncr = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = ncr;
      }
    }
  }
  if (inverse) for (let k = 0; k < n; k++) { r[k] /= n; i[k] /= n; }
  return { re: r, im: i };
}

/** Naive O(N^2) DFT for any length (reference / teaching). */
export function dft(re: number[], im: number[]): { re: number[]; im: number[] } {
  const N = re.length;
  const R = new Array<number>(N), I = new Array<number>(N);
  for (let k = 0; k < N; k++) {
    let sr = 0, si = 0;
    for (let n = 0; n < N; n++) {
      const a = (-2 * Math.PI * k * n) / N, c = Math.cos(a), s = Math.sin(a);
      sr += re[n] * c - im[n] * s; si += re[n] * s + im[n] * c;
    }
    R[k] = sr; I[k] = si;
  }
  return { re: R, im: I };
}

/**
 * Two-sided, fft-shifted spectrum. Continuous signals give a Riemann-sum CTFT estimate
 * (X(f) ≈ dt·Σ x[n]e^{-j2πf t_n}); discrete signals give the DTFT sampled on a dense grid.
 * The time origin t0 is accounted for, so phase is referenced to t = 0.
 */
export function spectrum(sig: Signal, opts: { window?: WindowType; zeroPad?: number } = {}): Spectrum {
  const N = sig.re.length;
  const w = opts.window === 'hann' ? hann(N) : new Array<number>(N).fill(1);
  const gain = N / w.reduce((a, b) => a + b, 0);
  const M = nextPow2(N * (opts.zeroPad ?? 2));
  const re = new Array<number>(M).fill(0), im = new Array<number>(M).fill(0);
  for (let n = 0; n < N; n++) { re[n] = sig.re[n] * w[n] * gain; im[n] = (sig.im?.[n] ?? 0) * w[n] * gain; }
  const X = fft(re, im);
  const scale = sig.domain === 'continuous' ? sig.dt : 1;
  const df = 1 / (M * sig.dt);
  const freq = new Array<number>(M), mag = new Array<number>(M), phase = new Array<number>(M);
  let peak = 0;
  for (let j = 0; j < M; j++) {
    const k = (j + M / 2) % M;
    const f = (k < M / 2 ? k : k - M) * df;
    const a = -2 * Math.PI * f * sig.t0, c = Math.cos(a), s = Math.sin(a);
    const xr = (X.re[k] * c - X.im[k] * s) * scale, xi = (X.re[k] * s + X.im[k] * c) * scale;
    freq[j] = f; mag[j] = Math.hypot(xr, xi); phase[j] = Math.atan2(xi, xr);
    if (mag[j] > peak) peak = mag[j];
  }
  for (let j = 0; j < M; j++) if (mag[j] < 1e-4 * peak) phase[j] = 0; // hide numerical-noise phase
  return { freq, mag, phase };
}

/** DTFT/CTFT-style evaluation H(f) = Σ h[n] e^{-j2πf t_n} (· dt for continuous) at arbitrary frequencies. */
export function frequencyResponse(h: Signal, freqs: number[]): { mag: number[]; phase: number[] } {
  const scale = h.domain === 'continuous' ? h.dt : 1;
  const mag: number[] = [], phase: number[] = [];
  for (const f of freqs) {
    let sr = 0, si = 0;
    for (let n = 0; n < h.re.length; n++) {
      const a = -2 * Math.PI * f * (h.t0 + n * h.dt), c = Math.cos(a), s = Math.sin(a);
      sr += h.re[n] * c - (h.im?.[n] ?? 0) * s; si += h.re[n] * s + (h.im?.[n] ?? 0) * c;
    }
    mag.push(Math.hypot(sr, si) * scale); phase.push(Math.atan2(si, sr));
  }
  return { mag, phase };
}

export function unwrap(phase: number[]): number[] {
  const out = phase.slice();
  for (let i = 1; i < out.length; i++) {
    let d = out[i] - out[i - 1];
    while (d > Math.PI) { out[i] -= 2 * Math.PI; d -= 2 * Math.PI; }
    while (d < -Math.PI) { out[i] += 2 * Math.PI; d += 2 * Math.PI; }
  }
  return out;
}

/** Highest positive frequency whose magnitude exceeds `thresh` × peak. */
export function estimateBandwidth(spec: Spectrum, thresh = 0.02): number {
  const peak = Math.max(...spec.mag);
  let hi = 0;
  for (let i = 0; i < spec.freq.length; i++) if (spec.freq[i] > 0 && spec.mag[i] > thresh * peak) hi = Math.max(hi, spec.freq[i]);
  return hi;
}
