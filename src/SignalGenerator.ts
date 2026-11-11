import { Domain, Grid, Signal, SignalParams, SignalType } from './types';

const TWO_PI = 2 * Math.PI;
const frac = (x: number) => x - Math.floor(x);

export const SIGNAL_LABELS: Record<SignalType, string> = {
  sine: 'Sine', cosine: 'Cosine', square: 'Square wave', pulse: 'Pulse train',
  sawtooth: 'Sawtooth', triangle: 'Triangular', step: 'Unit step u(t)',
  impulse: 'Unit impulse δ(t)', expReal: 'Real exponential e^{at}',
  expComplex: 'Complex exponential e^{jωt}', gaussian: 'Gaussian pulse', noise: 'White Gaussian noise',
};

export function defaultParams(type: SignalType = 'sine'): SignalParams {
  return { type, amplitude: 1, frequency: 1, phase: 0, delay: 0, duty: 0.5, alpha: -1, seed: 1 };
}

/** Grid helpers: continuous grids are dense; discrete grids are aligned to multiples of dt. */
export function makeGrid(tMin: number, tMax: number, dt: number, domain: Domain): Grid {
  const start = domain === 'discrete' ? Math.ceil(tMin / dt - 1e-9) * dt : tMin;
  return { tMin: start, dt, n: Math.floor((tMax - start) / dt + 1e-9) + 1, domain };
}

// Deterministic hash -> U(0,1) (mulberry32 step), so noise is a pure function of (t, seed).
function hash01(n: number): number {
  let h = (n | 0) + 0x6d2b79f5;
  h = Math.imul(h ^ (h >>> 15), h | 1);
  h ^= h + Math.imul(h ^ (h >>> 7), h | 61);
  return ((h ^ (h >>> 14)) >>> 0) / 4294967296;
}
function gaussianNoise(key: number, seed: number): number {
  const u1 = Math.max(hash01(key * 2 + seed * 7919), 1e-12);
  const u2 = hash01(key * 2 + 1 + seed * 7919);
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(TWO_PI * u2);
}

/**
 * Evaluate f(t) -> [re, im].
 * Continuous δ(t) is a one-sample spike of height A/dt (area A); discrete δ[n] has height A.
 */
export function evaluate(p: SignalParams, t: number, dt: number, domain: Domain = 'continuous'): [number, number] {
  const A = p.amplitude;
  const u = t - p.delay;
  const w = TWO_PI * p.frequency;
  const ph = frac(p.frequency * u + p.phase / TWO_PI); // normalised phase in [0,1)

  switch (p.type) {
    case 'sine':       return [A * Math.sin(w * u + p.phase), 0];
    case 'cosine':     return [A * Math.cos(w * u + p.phase), 0];
    case 'square':     return [ph < p.duty ? A : -A, 0];
    case 'pulse':      return [ph < p.duty ? A : 0, 0];
    case 'sawtooth':   return [A * (2 * ph - 1), 0];
    case 'triangle':   return [A * (2 * Math.abs(2 * ph - 1) - 1), 0];
    case 'step':       return [u >= -1e-12 ? A : 0, 0];
    case 'impulse': {
      const hit = u >= -dt / 2 && u < dt / 2;
      return [hit ? (domain === 'continuous' ? A / dt : A) : 0, 0];
    }
    case 'expReal':    return [A * Math.exp(p.alpha * u), 0];
    case 'expComplex': return [A * Math.cos(w * u + p.phase), A * Math.sin(w * u + p.phase)];
    case 'gaussian': {
      const sigma = 0.5 / Math.max(p.frequency, 0.05); // width shrinks as "frequency" grows
      return [A * Math.exp(-0.5 * (u / sigma) ** 2), 0];
    }
    case 'noise':      return [A * gaussianNoise(Math.round(t * 1e6), p.seed), 0];
  }
}

export function generateSignal(p: SignalParams, g: Grid): Signal {
  const re = new Array<number>(g.n);
  const complex = p.type === 'expComplex';
  const im = complex ? new Array<number>(g.n) : undefined;
  for (let i = 0; i < g.n; i++) {
    const [a, b] = evaluate(p, g.tMin + i * g.dt, g.dt, g.domain);
    re[i] = a;
    if (im) im[i] = b;
  }
  return { t0: g.tMin, dt: g.dt, re, im, domain: g.domain };
}
