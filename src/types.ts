export type Domain = 'continuous' | 'discrete';

export type SignalType =
  | 'sine' | 'cosine' | 'square' | 'pulse' | 'sawtooth' | 'triangle'
  | 'step' | 'impulse' | 'expReal' | 'expComplex' | 'gaussian' | 'noise';

export interface SignalParams {
  type: SignalType;
  amplitude: number;   // A
  frequency: number;   // f in Hz (for discrete: Hz of the underlying sample clock)
  phase: number;       // phi in radians
  delay: number;       // t0 in seconds
  duty: number;        // 0..1 (square / pulse)
  alpha: number;       // a in A*e^{a t}
  seed: number;        // noise seed
}

/** Uniform sample grid: t[i] = tMin + i*dt */
export interface Grid { tMin: number; dt: number; n: number; domain: Domain }

/** Uniformly sampled (possibly complex) signal. Discrete signals are x[n] = x(n*dt). */
export interface Signal { t0: number; dt: number; re: number[]; im?: number[]; domain: Domain }

export type ArithmeticOp = 'none' | 'add' | 'sub' | 'mul' | 'conv';
export type ReconstructionMethod = 'zoh' | 'foh' | 'sinc';
export type WindowType = 'rect' | 'hann';

export interface Spectrum { freq: number[]; mag: number[]; phase: number[] } // two-sided, fft-shifted

export type SystemType = 'rc' | 'movingAverage' | 'delay' | 'echo' | 'resonator';
export interface SystemConfig {
  type: SystemType;
  tau: number;   // time constant / window / delay (s)
  f0: number;    // resonator frequency (Hz)
  gain: number;  // echo gain / overall gain
}
