import { useMemo, useState } from 'react';
import { SignalPlotter, Trace } from './SignalPlotter';
import { SIGNAL_LABELS, defaultParams, evaluate, generateSignal, makeGrid } from './SignalGenerator';
import {
  amplitudeScale, combine, convolve, dcOffset, reconstruct, sampleFunction, timeAxis, timeScale, timeShift, transformTime,
} from './SignalOperations';
import { estimateBandwidth, frequencyResponse, logSpace, spectrum, unwrap } from './DSPTransforms';
import type {
  ArithmeticOp, Domain, ReconstructionMethod, Signal, SignalParams, SignalType, SystemConfig, SystemType, WindowType,
} from './types';

const TYPES = Object.keys(SIGNAL_LABELS) as SignalType[];
const PERIODIC: SignalType[] = ['sine', 'cosine', 'square', 'pulse', 'sawtooth', 'triangle', 'expComplex'];
const C = { a: 'var(--a)', b: 'var(--b)', out: 'var(--out)' };
// canvas can't resolve var(); the plotter reads colours from CSS so we pass resolved values below
const col = (v: string) => getComputedStyle(document.documentElement).getPropertyValue(v.slice(4, -1)).trim() || '#1f58c7';

function Slider(p: { label: string; value: number; min: number; max: number; step: number; unit?: string; onChange: (v: number) => void }) {
  const dec = p.step < 0.01 ? 3 : p.step < 1 ? 2 : 0;
  return (
    <label className="ctl">
      <span>{p.label}</span>
      <input type="range" min={p.min} max={p.max} step={p.step} value={p.value} onChange={(e) => p.onChange(+e.target.value)} />
      <output>{p.value.toFixed(dec)}{p.unit ?? ''}</output>
    </label>
  );
}

function Seg<T extends string>(p: { value: T; options: [T, string][]; onChange: (v: T) => void }) {
  return (
    <div className="seg">
      {p.options.map(([k, l]) => <button key={k} aria-pressed={p.value === k} onClick={() => p.onChange(k)}>{l}</button>)}
    </div>
  );
}

function SignalEditor(p: { title: string; value: SignalParams; maxF: number; onChange: (v: SignalParams) => void }) {
  const v = p.value, set = (patch: Partial<SignalParams>) => p.onChange({ ...v, ...patch });
  return (
    <fieldset>
      <legend>{p.title}</legend>
      <select value={v.type} onChange={(e) => set({ type: e.target.value as SignalType })}>
        {TYPES.map((t) => <option key={t} value={t}>{SIGNAL_LABELS[t]}</option>)}
      </select>
      <Slider label="Amplitude A" value={v.amplitude} min={0} max={5} step={0.1} onChange={(x) => set({ amplitude: x })} />
      {(PERIODIC.includes(v.type) || v.type === 'gaussian') && (
        <Slider label={v.type === 'gaussian' ? 'Width 1/f' : 'Frequency f'} value={v.frequency} min={0.1} max={p.maxF} step={0.1} unit=" Hz" onChange={(x) => set({ frequency: x })} />
      )}
      {PERIODIC.includes(v.type) && (
        <Slider label="Phase φ" value={Math.round((v.phase * 180) / Math.PI)} min={-180} max={180} step={5} unit="°" onChange={(x) => set({ phase: (x * Math.PI) / 180 })} />
      )}
      {(v.type === 'square' || v.type === 'pulse') && <Slider label="Duty cycle" value={v.duty} min={0.05} max={0.95} step={0.05} onChange={(x) => set({ duty: x })} />}
      {v.type === 'expReal' && <Slider label="Rate a" value={v.alpha} min={-3} max={3} step={0.1} onChange={(x) => set({ alpha: x })} />}
      {v.type !== 'noise' && <Slider label="Delay t₀" value={v.delay} min={-2} max={2} step={0.05} unit=" s" onChange={(x) => set({ delay: x })} />}
    </fieldset>
  );
}

/** Impulse responses h(t). δ terms are expressed as areas; `spike` converts to a one-sample value. */
function impulseResponse(cfg: SystemConfig, dt: number, n: number, domain: Domain): Signal {
  const spike = (t: number, at: number) => (t >= at - dt / 2 && t < at + dt / 2 ? 1 / dt : 0);
  const h = (t: number): number => {
    switch (cfg.type) {
      case 'rc':            return t >= 0 ? (cfg.gain / cfg.tau) * Math.exp(-t / cfg.tau) : 0;
      case 'movingAverage': return t >= 0 && t < cfg.tau ? cfg.gain / cfg.tau : 0;
      case 'delay':         return cfg.gain * spike(t, cfg.tau);
      case 'echo':          return spike(t, 0) + cfg.gain * spike(t, cfg.tau);
      case 'resonator':     return t >= 0 ? cfg.gain * 2 * Math.PI * cfg.f0 * Math.exp(-t / cfg.tau) * Math.sin(2 * Math.PI * cfg.f0 * t) : 0;
    }
  };
  const k = domain === 'discrete' ? dt : 1; // h[n] = Ts·h(nTs) so Σ x[k]h[n-k] approximates the integral
  return { t0: 0, dt, domain, re: Array.from({ length: n }, (_, i) => h(i * dt) * k) };
}

type Tab = 'analysis' | 'sampling' | 'lti';

export default function App() {
  const [tab, setTab] = useState<Tab>('analysis');
  const [domain, setDomain] = useState<Domain>('continuous');
  const [T, setT] = useState(2);          // half-window (s)
  const [fsD, setFsD] = useState(32);     // discrete-time sample rate
  const [pa, setPa] = useState<SignalParams>(defaultParams('sine'));
  const [pb, setPb] = useState<SignalParams>({ ...defaultParams('cosine'), frequency: 3, amplitude: 0.5 });
  const [op, setOp] = useState<ArithmeticOp>('none');
  const [shift, setShift] = useState(0), [scale, setScale] = useState(1), [rev, setRev] = useState(false);
  const [gain, setGain] = useState(1), [dc, setDc] = useState(0);
  // spectrum view
  const [mag, setMag] = useState<'db' | 'lin'>('lin'), [ang, setAng] = useState<'rad' | 'deg'>('rad'), [win, setWin] = useState<WindowType>('rect');
  // sampling
  const [fs, setFs] = useState(6), [method, setMethod] = useState<ReconstructionMethod>('sinc');
  // LTI
  const [sys, setSys] = useState<SystemConfig>({ type: 'rc', tau: 0.3, f0: 2, gain: 1 });

  const colors = { a: col(C.a), b: col(C.b), out: col(C.out) };
  const maxF = domain === 'discrete' ? fsD / 2 : 10;
  const grid = useMemo(() => domain === 'continuous' ? makeGrid(-T, T, (2 * T) / 4096, 'continuous') : makeGrid(-T, T, 1 / fsD, 'discrete'), [domain, T, fsD]);
  const sigA = useMemo(() => generateSignal(pa, grid), [pa, grid]);
  const sigB = useMemo(() => generateSignal(pb, grid), [pb, grid]);
  const result = useMemo(() => {
    let s = timeShift(timeScale(sigA, rev ? -scale : scale), shift);
    s = dcOffset(amplitudeScale(s, gain), dc);
    return combine(s, sigB, op);
  }, [sigA, sigB, op, shift, scale, rev, gain, dc]);

  const mode = domain === 'discrete' ? 'stem' : 'line';
  const t = timeAxis(result);
  const timeTraces: Trace[] = [
    { x: t, y: sigA.re, color: colors.a, label: 'A(t)', mode, width: 1.2 },
    ...(op !== 'none' ? [{ x: t, y: timeAxis(sigB).map((_, i) => sigB.re[i]), color: colors.b, label: 'B(t)', mode, width: 1.2, dashed: true } as Trace] : []),
    { x: t, y: result.re, color: colors.out, label: op === 'conv' ? 'A * B' : 'Result', mode },
    ...(result.im ? [{ x: t, y: result.im, color: colors.out, label: 'Im', mode, dashed: true } as Trace] : []),
  ];

  const nyq = (domain === 'discrete' ? fsD : 1 / grid.dt) / 2;
  const fView = Math.min(nyq, Math.max(8, 6 * pa.frequency));
  const spec = useMemo(() => spectrum(result, { window: win }), [result, win]);
  const specTraces = (kind: 'mag' | 'phase', s = spec, color = colors.out, label = 'Result'): Trace => ({
    x: s.freq, color, label, width: 1.6,
    y: kind === 'mag' ? s.mag.map((m) => (mag === 'db' ? 20 * Math.log10(Math.max(m, 1e-6)) : m)) : s.phase.map((p) => (ang === 'deg' ? (p * 180) / Math.PI : p)),
  });

  // ---- sampling tab: always on a dense continuous grid ----
  const fine = useMemo(() => makeGrid(-T, T, (2 * T) / 4096, 'continuous'), [T]);
  const xc = useMemo(() => generateSignal(pa, fine), [pa, fine]);
  const samp = useMemo(() => sampleFunction((tt) => evaluate(pa, tt, 1 / fs, 'discrete')[0], -T, T, fs), [pa, fs, T]);
  const rec = useMemo(() => reconstruct(samp.t, samp.v, samp.Ts, timeAxis(xc), method), [samp, xc, method]);
  const xcSpec = useMemo(() => spectrum(xc), [xc]);
  const recSpec = useMemo(() => spectrum({ ...xc, re: rec }), [xc, rec]);
  const tonal = pa.type === 'sine' || pa.type === 'cosine' || pa.type === 'expComplex';
  const fmax = tonal ? pa.frequency : estimateBandwidth(xcSpec, 0.02);
  const aliasF = Math.abs(pa.frequency - fs * Math.round(pa.frequency / fs));
  const ok = fs >= 2 * fmax;

  // ---- LTI tab ----
  const h = useMemo(() => impulseResponse(sys, grid.dt, Math.floor(T / grid.dt) + 1, domain), [sys, grid.dt, T, domain]);
  const y = useMemo(() => transformTime(convolve(result, h), (tt) => tt, result), [result, h]);
  const bode = useMemo(() => {
    const f = logSpace(0.05, Math.min(500, 0.5 / grid.dt), 240), H = frequencyResponse(h, f);
    return { f, db: H.mag.map((m) => 20 * Math.log10(Math.max(m, 1e-6))), deg: unwrap(H.phase).map((p) => (p * 180) / Math.PI) };
  }, [h, grid.dt]);

  return (
    <div className="app">
      <header><h1>SignalLab</h1><span>MCT 223 Signals and Systems simulator</span></header>
      <aside>
        <fieldset>
          <legend>Time axis</legend>
          <div className="row"><Seg value={domain} options={[['continuous', 'Continuous x(t)'], ['discrete', 'Discrete x[n]']]} onChange={setDomain} /></div>
          <Slider label="Window ±T" value={T} min={1} max={8} step={0.5} unit=" s" onChange={setT} />
          {domain === 'discrete' && <Slider label="Rate Fs" value={fsD} min={4} max={64} step={1} unit=" Hz" onChange={setFsD} />}
        </fieldset>
        <SignalEditor title="Signal A" value={pa} maxF={maxF} onChange={setPa} />
        <fieldset>
          <legend>Transform A</legend>
          <Slider label="Shift t₀" value={shift} min={-2} max={2} step={0.05} unit=" s" onChange={setShift} />
          <Slider label="Scale a" value={scale} min={0.25} max={4} step={0.05} onChange={setScale} />
          <label className="row"><input type="checkbox" checked={rev} onChange={(e) => setRev(e.target.checked)} /> Reverse time f(−t)</label>
          <Slider label="Gain k" value={gain} min={-3} max={3} step={0.1} onChange={setGain} />
          <Slider label="DC offset C" value={dc} min={-2} max={2} step={0.1} onChange={setDc} />
        </fieldset>
        <fieldset>
          <legend>Combine with B</legend>
          <select value={op} onChange={(e) => setOp(e.target.value as ArithmeticOp)}>
            <option value="none">None</option><option value="add">A + B</option><option value="sub">A − B</option>
            <option value="mul">A · B</option><option value="conv">A * B (convolution)</option>
          </select>
        </fieldset>
        {op !== 'none' && <SignalEditor title="Signal B" value={pb} maxF={maxF} onChange={setPb} />}
      </aside>

      <main>
        <div className="tabs" role="tablist">
          {([['analysis', 'Time and frequency'], ['sampling', 'Sampling and reconstruction'], ['lti', 'LTI system']] as [Tab, string][]).map(([k, l]) =>
            <button key={k} role="tab" aria-selected={tab === k} onClick={() => setTab(k)}>{l}</button>)}
        </div>

        {tab === 'analysis' && (<>
          <SignalPlotter title={domain === 'discrete' ? 'Discrete-time signal x[n]' : 'Continuous-time signal x(t)'} traces={timeTraces} yLabel="amplitude" />
          <div className="toolbar">
            <Seg value={mag} options={[['lin', 'Linear'], ['db', 'dB']]} onChange={setMag} />
            <Seg value={ang} options={[['rad', 'rad'], ['deg', 'deg']]} onChange={setAng} />
            <Seg value={win} options={[['rect', 'Rectangular'], ['hann', 'Hann window']]} onChange={setWin} />
          </div>
          <div className="grid3" style={{ gridTemplateColumns: '1fr 1fr' }}>
            <SignalPlotter title={domain === 'discrete' ? '|X(e^{jω})| (DTFT)' : '|X(f)| (CTFT estimate)'} traces={[specTraces('mag')]} xLabel="f (Hz)" yLabel={mag === 'db' ? 'dB' : 'magnitude'} xRange={[-fView, fView]} height={240} />
            <SignalPlotter title="∠X(f)" traces={[specTraces('phase')]} xLabel="f (Hz)" yLabel={ang} xRange={[-fView, fView]} height={240} />
          </div>
        </>)}

        {tab === 'sampling' && (<>
          <div className="toolbar">
            <label className="ctl" style={{ width: 340 }}><span>Rate Fs</span><input type="range" min={0.5} max={40} step={0.5} value={fs} onChange={(e) => setFs(+e.target.value)} /><output>{fs} Hz</output></label>
            <Seg value={method} options={[['zoh', 'Zero-order hold'], ['foh', 'First-order hold'], ['sinc', 'Sinc']]} onChange={setMethod} />
          </div>
          <div className={'note' + (ok ? '' : ' bad')}>
            {tonal
              ? ok ? `Fs = ${fs} Hz ≥ 2·Fmax = ${(2 * fmax).toFixed(1)} Hz. The Nyquist–Shannon condition holds, so ideal reconstruction recovers x(t).`
                   : `Fs = ${fs} Hz < 2·Fmax = ${(2 * fmax).toFixed(1)} Hz. Aliasing: the ${fmax.toFixed(1)} Hz tone appears at ${aliasF.toFixed(2)} Hz.`
              : `Estimated bandwidth Fmax ≈ ${fmax.toFixed(1)} Hz (frequency content above 2% of the peak), so the Nyquist rate is about ${(2 * fmax).toFixed(1)} Hz. ${ok ? 'Fs is above it.' : 'Fs is below it, so expect aliasing.'} Signals with sharp edges are not strictly band-limited.`}
          </div>
          <SignalPlotter title="Sampling and reconstruction" yLabel="amplitude" traces={[
            { x: timeAxis(xc), y: xc.re, color: colors.a, label: 'x(t)', width: 1.2 },
            { x: timeAxis(xc), y: rec, color: colors.out, label: 'x_r(t)', mode: method === 'zoh' ? 'step' : 'line' },
            { x: samp.t, y: samp.v, color: colors.b, label: 'x[n] = x(nTs)', mode: 'stem' },
          ]} />
          <SignalPlotter title="Spectrum: original vs. reconstruction" xLabel="f (Hz)" yLabel="magnitude" xRange={[-Math.max(fs, 2 * fmax, 6), Math.max(fs, 2 * fmax, 6)]} height={240} traces={[
            { x: xcSpec.freq, y: xcSpec.mag, color: colors.a, label: '|X(f)|', width: 1.2 },
            { x: recSpec.freq, y: recSpec.mag, color: colors.out, label: '|Xr(f)|', dashed: true },
          ]} />
        </>)}

        {tab === 'lti' && (<>
          <fieldset style={{ marginBottom: 12 }}>
            <legend>System h(t)</legend>
            <select value={sys.type} onChange={(e) => setSys({ ...sys, type: e.target.value as SystemType })} style={{ maxWidth: 320 }}>
              <option value="rc">RC low-pass: (1/τ)e^{'{-t/τ}'}u(t)</option><option value="movingAverage">Moving average (rect of width τ)</option>
              <option value="delay">Ideal delay δ(t − τ)</option><option value="echo">Echo δ(t) + g·δ(t − τ)</option><option value="resonator">Damped resonator</option>
            </select>
            <div className="grid3">
              <Slider label="τ" value={sys.tau} min={0.05} max={T} step={0.05} unit=" s" onChange={(x) => setSys({ ...sys, tau: x })} />
              {sys.type === 'resonator' && <Slider label="f₀" value={sys.f0} min={0.5} max={10} step={0.5} unit=" Hz" onChange={(x) => setSys({ ...sys, f0: x })} />}
              <Slider label="Gain" value={sys.gain} min={0.1} max={2} step={0.05} onChange={(x) => setSys({ ...sys, gain: x })} />
            </div>
          </fieldset>
          <SignalPlotter title="Input x vs. output y = x * h" yLabel="amplitude" traces={[
            { x: t, y: result.re, color: colors.a, label: 'x', mode, width: 1.3 },
            { x: t, y: y.re, color: colors.out, label: 'y', mode },
          ]} />
          <div className="grid3">
            <SignalPlotter title="Impulse response h" traces={[{ x: timeAxis(h), y: h.re, color: colors.b, label: 'h', mode }]} height={230} />
            <SignalPlotter title="Bode magnitude" xLog xLabel="f (Hz)" yLabel="dB" traces={[{ x: bode.f, y: bode.db, color: colors.out, label: '|H|' }]} height={230} />
            <SignalPlotter title="Bode phase" xLog xLabel="f (Hz)" yLabel="deg" traces={[{ x: bode.f, y: bode.deg, color: colors.out, label: '∠H' }]} height={230} />
          </div>
        </>)}
      </main>
    </div>
  );
}
