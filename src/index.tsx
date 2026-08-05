import React, {
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
  type ReactElement,
  type ReactNode,
} from "react";
import { createRoot } from "react-dom/client";
import {
  ADVERSARY_WEIGHT_RANGE,
  ADVERSARY_OBJECTIVE_DEFAULTS,
  GALLERY,
  GAME_LEARNING_RATE_RANGE,
  adversaryLossOf,
  adversaryTargetOf,
  resolveAdversary,
  startLoop,
  type AdversaryTelemetry,
  type ColorMode,
  type HeadHealth,
  type LoopHandle,
} from "./main";
import type {
  AdversaryLoss,
  AdversaryTarget,
  TupleEncoding,
} from "./core/gan/adversary";
import type { ColormapName } from "./draw/colormap";
import type { BorderMode } from "./render/webgpu/advect_wgsl";
import type { SplatStyle } from "./render/webgpu/splat";
import "./ui.css";

const container = document.getElementById("app");
if (!container) throw new Error("Container not found");

const PMIN = 200;
const PMAX = 1_000_000;
const SURPRISE_HISTORY = 120;
const CMAPS: ColormapName[] = ["inferno", "viridis", "coolwarm"];
const G_LR_MIN = GAME_LEARNING_RATE_RANGE.generator.min;
const G_LR_MAX = GAME_LEARNING_RATE_RANGE.generator.max;
const D_LR_MIN = GAME_LEARNING_RATE_RANGE.discriminator.min;
const D_LR_MAX = GAME_LEARNING_RATE_RANGE.discriminator.max;
const ADV_WEIGHT_MIN = ADVERSARY_WEIGHT_RANGE.min;
const ADV_WEIGHT_MAX = ADVERSARY_WEIGHT_RANGE.max;

type RelationalView = "rotation" | "rotation-scale-raw" | "rotation-scale-adjusted";
type TupleView = "point" | "pair" | "tri" | "quad-labelled";

interface RuntimeConfig {
  piece: number;
  border: BorderMode;
  encoding: TupleEncoding;
  target: AdversaryTarget;
  loss: AdversaryLoss;
  adversaryKind: "off" | "single" | "wta";
  k: number;
  relaxEps: number;
}

type AdversaryLossTag = AdversaryLoss["tag"];

function lossWithTag(tag: AdversaryLossTag, previous: AdversaryLoss): AdversaryLoss {
  const tau =
    "tau" in previous ? previous.tau : ADVERSARY_OBJECTIVE_DEFAULTS.tau;
  const scaleWeight =
    "scaleWeight" in previous
      ? previous.scaleWeight
      : ADVERSARY_OBJECTIVE_DEFAULTS.scaleWeight;
  const energyWeight =
    "energyWeight" in previous
      ? previous.energyWeight
      : ADVERSARY_OBJECTIVE_DEFAULTS.energyWeight;
  const energyTarget =
    "energyTarget" in previous
      ? previous.energyTarget
      : ADVERSARY_OBJECTIVE_DEFAULTS.energyTarget;
  switch (tag) {
    case "raw-vector":
      return { tag };
    case "soft-angle":
      return { tag, tau };
    case "angle-relative-scale":
    case "angle-scale-hold":
      return { tag, tau, scaleWeight, energyWeight, energyTarget };
  }
}

function lossHasAngle(loss: AdversaryLoss): loss is Exclude<
  AdversaryLoss,
  { readonly tag: "raw-vector" }
> {
  return loss.tag !== "raw-vector";
}

function lossHasScale(loss: AdversaryLoss): loss is Extract<
  AdversaryLoss,
  { readonly tag: "angle-relative-scale" | "angle-scale-hold" }
> {
  return (
    loss.tag === "angle-relative-scale" || loss.tag === "angle-scale-hold"
  );
}

const particleToSlider = (n: number): number =>
  Math.log(Math.min(Math.max(n, PMIN), PMAX) / PMIN) / Math.log(PMAX / PMIN);

const sliderToParticle = (t: number): number => {
  const raw = PMIN * Math.pow(PMAX / PMIN, t);
  const magnitude = Math.pow(10, Math.max(0, Math.floor(Math.log10(raw)) - 1));
  return Math.round(raw / magnitude) * magnitude;
};

const learningRateToSlider = (value: number, min: number, max: number): number =>
  Math.log(Math.min(max, Math.max(min, value)) / min) / Math.log(max / min);

const sliderToLearningRate = (value: number, min: number, max: number): number =>
  min * Math.pow(max / min, value);

function defaultsForPiece(piece: number): RuntimeConfig {
  // URL adversary knobs are intentionally GLOBAL: selecting another gallery
  // piece re-resolves that piece through the same query. This matches
  // startLoop's canonical URL policy and prevents React from masking
  // ?advM/?advK/?advEps by passing the piece defaults back as overrides.
  const adv = resolveAdversary(
    GALLERY[piece].adversary,
    new URLSearchParams(window.location.search)
  );
  return {
    piece,
    border: { tag: "wrap" },
    encoding: adv.tag === "on" ? adv.encoding : ({ tag: "pair-rotation" } as TupleEncoding),
    target: adv.tag === "on" ? adversaryTargetOf(adv) : { tag: "force" },
    loss: adv.tag === "on" ? adversaryLossOf(adv) : { tag: "raw-vector" },
    adversaryKind: adv.tag === "on" ? adv.kind.tag : "off",
    k: adv.tag === "on" && adv.kind.tag === "wta" ? adv.kind.k : 1,
    relaxEps: adv.tag === "on" && adv.kind.tag === "wta" ? adv.kind.relaxEps : 0,
  };
}

function encodingForView(view: RelationalView): TupleEncoding {
  switch (view) {
    case "rotation":
      return { tag: "pair-rotation" };
    case "rotation-scale-raw":
      return { tag: "pair-rotation-scale-raw" };
    case "rotation-scale-adjusted":
      return { tag: "pair-rotation-scale-adjusted" };
  }
}

function viewForEncoding(encoding: TupleEncoding): RelationalView | null {
  switch (encoding.tag) {
    case "pair":
    case "pair-rotation":
      return "rotation";
    case "pair-rotation-scale-raw":
      return "rotation-scale-raw";
    case "pair-rotation-scale-adjusted":
      return "rotation-scale-adjusted";
    default:
      return null;
  }
}

function tupleViewForEncoding(encoding: TupleEncoding): TupleView {
  switch (encoding.tag) {
    case "point":
      return "point";
    case "pair":
    case "pair-rotation":
    case "pair-rotation-scale-raw":
    case "pair-rotation-scale-adjusted":
      return "pair";
    case "tri":
      return "tri";
    case "quad-labelled":
      return "quad-labelled";
  }
}

function encodingForTuple(
  tuple: TupleView,
  current: TupleEncoding
): TupleEncoding {
  switch (tuple) {
    case "point":
      return { tag: "point" };
    case "pair":
      // Preserve the selected pair quotient. Entering pair mode from another
      // arity chooses the scale-adjusted observer—the safe flagship—not the
      // deliberately scale-cheating raw control.
      return viewForEncoding(current)
        ? current
        : { tag: "pair-rotation-scale-adjusted" };
    case "tri":
      return { tag: "tri" };
    case "quad-labelled":
      return { tag: "quad-labelled" };
  }
}

function observerLabel(
  encoding: TupleEncoding,
  target: AdversaryTarget
): string {
  switch (encoding.tag) {
    case "point":
      return target.tag === "post-velocity"
        ? "POINT · absolute x + incoming v → pre-border v+"
        : "POINT · absolute x → F(x)";
    case "pair":
    case "pair-rotation":
      return "PAIR · rotation quotient";
    case "pair-rotation-scale-raw":
      return "PAIR · similarity-blind context, raw preset";
    case "pair-rotation-scale-adjusted":
      return "PAIR · similarity-blind context, angle preset";
    case "tri":
      return "TRI · unordered E(2); ties inactive";
    case "quad-labelled":
      return "QUAD-L · labelled rotation quotient; raw scale";
  }
}

function ControlSection({
  title,
  children,
  testid,
}: {
  title: string;
  children: ReactNode;
  testid?: string;
}): ReactElement {
  return (
    <section className="tui-section" aria-label={title} data-testid={testid}>
      <div className="tui-section-title">{title}</div>
      <div className="tui-section-body">{children}</div>
    </section>
  );
}

function RangeRow({
  label,
  value,
  min,
  max,
  step,
  display,
  onChange,
  testid,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  display: string;
  onChange: (value: number) => void;
  testid: string;
}): ReactElement {
  return (
    <label className="tui-row" data-testid={testid}>
      <span className="tui-label">{label}</span>
      <input
        className="block-slider"
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(event) => onChange(Number(event.currentTarget.value))}
        aria-label={label}
      />
      <output className="tui-value">{display}</output>
    </label>
  );
}

function Segmented<T extends string>({
  label,
  value,
  choices,
  onChange,
  testid,
}: {
  label: string;
  value: T | null;
  choices: readonly { value: T; label: string; title?: string }[];
  onChange: (value: T) => void;
  testid: string;
}): ReactElement {
  return (
    <div className="tui-segment-row" data-testid={testid}>
      <span className="tui-label">{label}</span>
      <div className="tui-segments" role="radiogroup" aria-label={label}>
        {choices.map((choice) => (
          <button
            key={choice.value}
            type="button"
            className="tui-chip"
            role="radio"
            aria-checked={value === choice.value}
            data-active={value === choice.value ? "true" : "false"}
            title={choice.title}
            onClick={() => onChange(choice.value)}
          >
            {choice.label}
          </button>
        ))}
      </div>
    </div>
  );
}

function Sparkline({ data }: { data: readonly number[] }): ReactElement {
  const width = 104;
  const height = 20;
  if (data.length < 2) {
    return (
      <svg
        className="sparkline"
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label="Surprise history awaiting samples"
      />
    );
  }

  const finite = data.filter(Number.isFinite);
  if (!finite.length) {
    return (
      <svg
        className="sparkline"
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label="Surprise history unavailable"
      />
    );
  }

  const lo = Math.min(...finite);
  const hi = Math.max(...finite);
  const flat = hi - lo <= Math.max(Math.abs(hi), 1e-12) * 1e-3;
  const span = Math.max(hi - lo, 1e-30);
  const points = data
    .map((sample, index) => {
      const x = (index / (data.length - 1)) * (width - 2) + 1;
      const y = flat
        ? height / 2
        : height - 2 - ((Number.isFinite(sample) ? sample - lo : 0) / span) * (height - 4);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");

  return (
    <svg
      className="sparkline"
      viewBox={`0 0 ${width} ${height}`}
      role="img"
      aria-label={`Surprise trend from ${lo.toExponential(1)} to ${hi.toExponential(1)}`}
    >
      <polyline points={points} />
    </svg>
  );
}

function HeadBars({ fractions }: { fractions: readonly number[] }): ReactElement {
  const k = Math.max(1, fractions.length);
  const floor = 0.05 / k;
  return (
    <div className="head-bars" aria-label="Predictor-head win fractions">
      {fractions.map((fraction, index) => (
        <span
          key={index}
          className="head-bar"
          data-starved={fraction < floor ? "true" : "false"}
          title={`head ${index}: ${(fraction * 100).toFixed(1)}%`}
          style={{ height: `${Math.max(2, Math.min(18, fraction * k * 9))}px` }}
        />
      ))}
    </div>
  );
}

function healthText(health: HeadHealth): string {
  switch (health.tag) {
    case "pileup":
      return "PILEUP";
    case "separated-unresolved":
      return "SKEW · SUPPORT?";
    case "unresolved":
      return "UNPROBED";
    case "ok":
      return "OK";
  }
}

function App(): ReactElement {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const telemetryHostRef = useRef<HTMLDivElement>(null);
  const handleRef = useRef<LoopHandle | null>(null);
  const cleanupRef = useRef<(() => void) | null>(null);
  const lastStartedPieceRef = useRef<number | null>(null);
  const [runtime, setRuntime] = useState<RuntimeConfig>(() => defaultsForPiece(0));

  const [particles, setParticles] = useState(1_000);
  const [samples, setSamples] = useState(256);
  const [maxVelocity, setMaxVelocity] = useState(24);
  const [drive, setDrive] = useState(0.65);
  const [generatorLearningRate, setGeneratorLearningRate] = useState(1e-3);
  const [discriminatorLearningRate, setDiscriminatorLearningRate] = useState(3e-3);
  const [resetRate, setResetRate] = useState(0.01);
  const [decay, setDecay] = useState(0);
  const [blend, setBlend] = useState(0.5);
  const [strokeStyle, setStrokeStyle] = useState<SplatStyle>("dot");
  const [strokeLength, setStrokeLength] = useState(3);
  const [advWeight, setAdvWeight] = useState(0);
  const [telemetry, setTelemetry] = useState<AdversaryTelemetry>({ tag: "off" });
  const [colorMode, setColorMode] = useState<ColorMode>({ tag: "velocity" });
  const [surpriseSpan, setSurpriseSpan] = useState<{
    lo: number;
    mid: number;
    hi: number;
    covered: number;
    collapsed: boolean;
  } | null>(null);
  const [history, setHistory] = useState<number[]>([]);

  const piece = GALLERY[runtime.piece];
  const adversary = runtime.adversaryKind !== "off";
  const hasField = !!piece.createField;
  const isAgreeDisagree = piece.mode === "agree-disagree";
  const isWta = runtime.adversaryKind === "wta";
  const hasStructuralFieldLoss =
    !!piece.fieldLoss &&
    (piece.fieldLoss.W_CHAOS !== 0 ||
      piece.fieldLoss.W_ISO !== 0 ||
      piece.fieldLoss.W_DIV !== 0 ||
      piece.fieldLoss.W_SPIRAL !== 0);
  const relationalView = viewForEncoding(runtime.encoding);

  useLayoutEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    // A compile-time control rebuilds the loop for the SAME piece. Preserve
    // every live dial across that rebuild. Selecting a DIFFERENT gallery piece
    // intentionally loads the new piece's defaults.
    const preserveLiveControls = lastStartedPieceRef.current === runtime.piece;
    lastStartedPieceRef.current = runtime.piece;
    cleanupRef.current?.();
    handleRef.current = null;
    let current = true;
    cleanupRef.current = startLoop(
      canvas,
      runtime.piece,
      (handle) => {
        if (!current) return;
        handleRef.current = handle;
        if (preserveLiveControls) {
          handle.setParticleCount(particles);
          handle.setSampleRate(samples);
          handle.setMaxVelocity(maxVelocity);
          handle.setDrive(drive);
          handle.setGeneratorLearningRate(generatorLearningRate);
          handle.setDiscriminatorLearningRate(discriminatorLearningRate);
          handle.setResetRate(resetRate);
          handle.setDecay(decay);
          handle.setBlend(blend);
          handle.setStrokeStyle(strokeStyle);
          handle.setStrokeLength(strokeLength);
          handle.setAdversaryWeight(advWeight);
          handle.setColorMode(colorMode);
        } else {
          setParticles(handle.getParticleCount());
          setSamples(handle.getSampleRate());
          setMaxVelocity(handle.getMaxVelocity());
          setDrive(handle.getDrive());
          setGeneratorLearningRate(handle.getGeneratorLearningRate());
          setDiscriminatorLearningRate(handle.getDiscriminatorLearningRate());
          setResetRate(handle.getResetRate());
          setDecay(handle.getDecay());
          setBlend(handle.getBlend());
          setStrokeStyle(handle.getStrokeStyle());
          setStrokeLength(handle.getStrokeLength());
          setAdvWeight(handle.getAdversaryWeight());
          setColorMode(handle.getColorMode());
        }
      },
      {
        telemetryHost: telemetryHostRef.current ?? undefined,
        overrides: {
          border: runtime.border,
          adversaryEncoding: runtime.encoding,
          adversaryTarget: runtime.target,
          adversaryLoss: runtime.loss,
          k: runtime.k,
          relaxEps: runtime.relaxEps,
        },
      }
    );

    return () => {
      current = false;
      cleanupRef.current?.();
      cleanupRef.current = null;
      handleRef.current = null;
    };
  }, [runtime]);

  useEffect(() => {
    setHistory([]);
    setTelemetry({ tag: "off" });
    setSurpriseSpan(null);
    const poll = window.setInterval(() => {
      const handle = handleRef.current;
      if (!handle) return;
      const next = handle.getAdversaryTelemetry();
      setTelemetry(next);
      setSurpriseSpan(handle.getSurpriseSpan());
      setColorMode(handle.getColorMode());
      if (next.tag === "on") {
        setHistory((previous) => {
          const updated = previous.concat(next.surprise);
          return updated.slice(-SURPRISE_HISTORY);
        });
      }
    }, 200);
    return () => window.clearInterval(poll);
  }, [runtime]);

  const updateColor = (next: ColorMode): void => {
    handleRef.current?.setColorMode(next);
    setColorMode(next);
  };

  return (
    <main className="art-shell">
      <canvas ref={canvasRef} id="myCanvas" aria-label="Neural force-field artwork" />

      <aside className="hud-stack" aria-label="Performance and piece controls">
        <div
          ref={telemetryHostRef}
          className="telemetry-host"
          data-testid="telemetry-host"
          aria-live="polite"
        />

        <div className="config-dock" data-testid="piece-config-dock">
          <header className="dock-header" aria-label={`Controls for ${piece.name}`}>
            <span className="dock-piece" title={piece.name}>
              {piece.name}
            </span>
          </header>

          <ControlSection title="simulation" testid="simulation-controls">
            <RangeRow
              label="particles"
              value={particleToSlider(particles)}
              min={0}
              max={1}
              step={0.002}
              display={particles.toLocaleString()}
              testid="particles-control"
              onChange={(value) => {
                const count = sliderToParticle(value);
                setParticles(count);
                handleRef.current?.setParticleCount(count);
              }}
            />
            <RangeRow
              label="train B"
              value={samples}
              min={16}
              max={4096}
              step={16}
              display={`${samples}`}
              testid="samples-control"
              onChange={(value) => {
                setSamples(value);
                handleRef.current?.setSampleRate(value);
              }}
            />
            <RangeRow
              label="max vel"
              value={maxVelocity}
              min={0.25}
              max={80}
              step={0.25}
              display={maxVelocity.toFixed(1)}
              testid="max-velocity-control"
              onChange={(value) => {
                setMaxVelocity(value);
                handleRef.current?.setMaxVelocity(value);
              }}
            />
            {adversary && piece.drive !== undefined && (
              <RangeRow
                label="drive"
                value={drive}
                min={0}
                max={1}
                step={0.01}
                display={`${drive.toFixed(2)}× clip`}
                testid="drive-control"
                onChange={(value) => {
                  setDrive(value);
                  handleRef.current?.setDrive(value);
                }}
              />
            )}
            <RangeRow
              label="respawn"
              value={resetRate}
              min={0}
              max={0.05}
              step={0.001}
              display={`${(resetRate * 100).toFixed(1)}%`}
              testid="random-reset-control"
              onChange={(value) => {
                setResetRate(value);
                handleRef.current?.setResetRate(value);
              }}
            />
            <Segmented
              label="border"
              value={runtime.border.tag}
              testid="border-control"
              choices={[
                { value: "wrap", label: "WRAP", title: "Periodic torus" },
                { value: "bounce", label: "BOUNCE", title: "Reflect at the box edge" },
                { value: "reset", label: "RESET", title: "Respawn when leaving the box" },
              ]}
              onChange={(tag) =>
                setRuntime((previous) => ({ ...previous, border: { tag } }))
              }
            />
            <p className="tui-note restart-note">border is compiled · changing it restarts</p>
          </ControlSection>

          <ControlSection title="ink" testid="ink-controls">
            <RangeRow
              label="trails"
              value={decay}
              min={0}
              max={0.99}
              step={0.005}
              display={decay.toFixed(2)}
              testid="trails-control"
              onChange={(value) => {
                setDecay(value);
                handleRef.current?.setDecay(value);
              }}
            />
            <Segmented
              label="stroke"
              value={strokeStyle}
              testid="stroke-style-control"
              choices={[
                { value: "dot", label: "DOT" },
                { value: "vel", label: "VEL" },
                { value: "curl", label: "CURL" },
              ]}
              onChange={(value) => {
                setStrokeStyle(value);
                handleRef.current?.setStrokeStyle(value);
              }}
            />
            {strokeStyle !== "dot" && (
              <RangeRow
                label="length"
                value={strokeLength}
                min={0.5}
                max={16}
                step={0.5}
                display={strokeLength.toFixed(1)}
                testid="stroke-length-control"
                onChange={(value) => {
                  setStrokeLength(value);
                  handleRef.current?.setStrokeLength(value);
                }}
              />
            )}
          </ControlSection>

          {hasField && (
            <ControlSection
              title={isAgreeDisagree ? "A/B/C roles" : "two-head field"}
              testid="field-controls"
            >
              <RangeRow
                label={isAgreeDisagree ? "blend C" : "blend A/B"}
                value={blend}
                min={0}
                max={1}
                step={0.01}
                display={blend.toFixed(2)}
                testid="head-blend-control"
                onChange={(value) => {
                  setBlend(value);
                  handleRef.current?.setBlend(value);
                }}
              />
              {isAgreeDisagree ? (
                <div className="rgb-role-legend" data-testid="rgb-role-legend">
                  <span className="role-a">R · A disagree</span>
                  <span className="role-b">G · B agree</span>
                  <span className="role-c">B · C blend (no loss)</span>
                </div>
              ) : (
                <p className="tui-note">neutral output mix — not order ↔ chaos</p>
              )}
            </ControlSection>
          )}

          {adversary && (
            <ControlSection title="adversary" testid="adversary-controls">
              <div className="tui-static-row" data-testid="objective-contract">
                <span>objective</span>
                <strong>
                  {runtime.loss.tag === "angle-scale-hold"
                    ? "ANGLE OPPOSE · SCALE AGREE"
                    : runtime.loss.tag === "angle-relative-scale"
                      ? "ANGLE + SCALE OPPOSE · ENERGY HOLD"
                    : isAgreeDisagree
                      ? "A OPPOSE · B COOPERATE"
                      : hasStructuralFieldLoss
                        ? "GAME + MAX CHAOS"
                        : runtime.loss.tag === "raw-vector"
                          ? "RAW VECTOR · CHEAT CONTROL"
                          : "ANGLE OPPONENT GAME"}
                </strong>
              </div>
              <Segmented
                label="target"
                value={runtime.target.tag}
                testid="adversary-target-control"
                choices={[
                  {
                    value: "force",
                    label: "FORCE",
                    title: "Predict raw neural field output F(x)",
                  },
                  {
                    value: "post-velocity",
                    label: "POST-V",
                    title:
                      "Predict normalized velocity after force, friction and clip; context includes incoming velocity",
                  },
                ]}
                onChange={(tag) =>
                  setRuntime((previous) => {
                    const target: AdversaryTarget = { tag };
                    if (tag === "post-velocity") {
                      return {
                        ...previous,
                        target,
                        encoding: { tag: "point" },
                        loss: lossHasScale(previous.loss)
                          ? lossWithTag("soft-angle", previous.loss)
                          : previous.loss,
                      };
                    }
                    return { ...previous, target };
                  })
                }
              />
              <Segmented
                label="loss"
                value={runtime.loss.tag}
                testid="adversary-loss-control"
                choices={[
                  {
                    value: "raw-vector",
                    label: "RAW",
                    title: "Euclidean vector error; explicit amplitude-cheat control",
                  },
                  {
                    value: "soft-angle",
                    label: "ANGLE",
                    title: "Exact smooth S² chord loss with bounded Jacobian",
                  },
                  {
                    value: "angle-relative-scale",
                    label: "A+S ADV",
                    title: "Adversarial direction and relative magnitude contrast",
                  },
                  {
                    value: "angle-scale-hold",
                    label: "A+S HOLD",
                    title:
                      "Direction adversarial; relative scale cooperative; absolute energy held",
                  },
                ]}
                onChange={(tag) =>
                  setRuntime((previous) => {
                    const nextLoss = lossWithTag(tag, previous.loss);
                    if (lossHasScale(nextLoss)) {
                      return {
                        ...previous,
                        target: { tag: "force" },
                        encoding:
                          previous.encoding.tag === "point"
                            ? { tag: "pair-rotation-scale-adjusted" }
                            : previous.encoding,
                        loss: nextLoss,
                      };
                    }
                    return { ...previous, loss: nextLoss };
                  })
                }
              />
              {lossHasAngle(runtime.loss) && (
                <RangeRow
                  label="soft τ"
                  value={runtime.loss.tau}
                  min={0.005}
                  max={0.25}
                  step={0.005}
                  display={runtime.loss.tau.toFixed(3)}
                  testid="adversary-angle-tau-control"
                  onChange={(tau) =>
                    setRuntime((previous) => ({
                      ...previous,
                      loss:
                        previous.loss.tag === "raw-vector"
                          ? previous.loss
                          : { ...previous.loss, tau },
                    }))
                  }
                />
              )}
              {lossHasScale(runtime.loss) && (
                <>
                  <RangeRow
                    label="scale w"
                    value={runtime.loss.scaleWeight}
                    min={0}
                    max={2}
                    step={0.05}
                    display={runtime.loss.scaleWeight.toFixed(2)}
                    testid="adversary-scale-weight-control"
                    onChange={(scaleWeight) =>
                      setRuntime((previous) => ({
                        ...previous,
                        loss: lossHasScale(previous.loss)
                          ? { ...previous.loss, scaleWeight }
                          : previous.loss,
                      }))
                    }
                  />
                  <RangeRow
                    label="energy w"
                    value={runtime.loss.energyWeight}
                    min={0}
                    max={1}
                    step={0.01}
                    display={runtime.loss.energyWeight.toFixed(2)}
                    testid="adversary-energy-weight-control"
                    onChange={(energyWeight) =>
                      setRuntime((previous) => ({
                        ...previous,
                        loss: lossHasScale(previous.loss)
                          ? { ...previous.loss, energyWeight }
                          : previous.loss,
                      }))
                    }
                  />
                  <RangeRow
                    label="energy"
                    value={runtime.loss.energyTarget}
                    min={0.02}
                    max={1}
                    step={0.01}
                    display={runtime.loss.energyTarget.toFixed(2)}
                    testid="adversary-energy-target-control"
                    onChange={(energyTarget) =>
                      setRuntime((previous) => ({
                        ...previous,
                        loss: lossHasScale(previous.loss)
                          ? { ...previous.loss, energyTarget }
                          : previous.loss,
                      }))
                    }
                  />
                </>
              )}
              {runtime.target.tag === "post-velocity" && (
                <p className="tui-note" data-testid="post-velocity-contract">
                  context = x + incoming v · target is pre-border normalized v+
                </p>
              )}
              {lossHasScale(runtime.loss) && (
                <p className="tui-note" data-testid="relative-scale-contract">
                  scale = within-tuple contrast · energy fixes absolute RMS
                </p>
              )}
              {runtime.loss.tag === "angle-scale-hold" && (
                <p className="tui-note" data-testid="scale-hold-diagnostic-contract">
                  displayed D-joint is not G reward · G raises angle, lowers scale error
                </p>
              )}
              <Segmented
                label="tuple"
                value={tupleViewForEncoding(runtime.encoding)}
                testid="adversary-tuple-control"
                choices={[
                  { value: "point", label: "1 · POINT" },
                  { value: "pair", label: "2 · PAIR" },
                  { value: "tri", label: "3 · TRI" },
                  {
                    value: "quad-labelled",
                    label: "4L · QUAD",
                    title: "Four labelled points; translation+rotation quotient only",
                  },
                ]}
                onChange={(tuple) =>
                  setRuntime((previous) => {
                    const encoding = encodingForTuple(tuple, previous.encoding);
                    return {
                      ...previous,
                      encoding,
                      target:
                        tuple === "point"
                          ? previous.target
                          : { tag: "force" },
                      loss:
                        tuple === "point" && lossHasScale(previous.loss)
                          ? lossWithTag("soft-angle", previous.loss)
                          : previous.loss,
                    };
                  })
                }
              />
              {relationalView ? (
                <Segmented
                  label="observer"
                  value={relationalView}
                  testid="adversary-view-control"
                  choices={[
                    {
                      value: "rotation",
                      label: "R",
                      title: "Quotient translation and rotation; keep scale observable",
                    },
                    {
                      value: "rotation-scale-raw",
                      label: "R+S RAW",
                      title:
                        "Scale-blind context; selecting it also chooses the raw-vector cheat control",
                    },
                    {
                      value: "rotation-scale-adjusted",
                      label: "R+S ADJ",
                      title:
                        "Same scale-blind context; selecting it chooses exact smooth soft-angle",
                    },
                  ]}
                  onChange={(view) =>
                    setRuntime((previous) => ({
                      ...previous,
                      encoding: encodingForView(view),
                      loss:
                        view === "rotation-scale-raw"
                          ? lossWithTag("raw-vector", previous.loss)
                          : view === "rotation-scale-adjusted"
                            ? lossWithTag("soft-angle", previous.loss)
                            : previous.loss,
                    }))
                  }
                />
              ) : (
                <div className="tui-static-row" data-testid="adversary-view-control">
                  <span>observer</span>
                  <strong>{observerLabel(runtime.encoding, runtime.target)}</strong>
                </div>
              )}
              <RangeRow
                label={runtime.loss.tag === "angle-scale-hold" ? "game w" : "reward"}
                value={advWeight}
                min={ADV_WEIGHT_MIN}
                max={ADV_WEIGHT_MAX}
                step={0.0005}
                display={advWeight.toFixed(3)}
                testid="adversary-reward-control"
                onChange={(value) => {
                  setAdvWeight(value);
                  handleRef.current?.setAdversaryWeight(value);
                }}
              />
              <RangeRow
                label="G lr"
                value={learningRateToSlider(
                  generatorLearningRate,
                  G_LR_MIN,
                  G_LR_MAX
                )}
                min={0}
                max={1}
                step={0.005}
                display={generatorLearningRate.toExponential(1)}
                testid="generator-learning-rate-control"
                onChange={(value) => {
                  const next = sliderToLearningRate(value, G_LR_MIN, G_LR_MAX);
                  setGeneratorLearningRate(next);
                  handleRef.current?.setGeneratorLearningRate(next);
                }}
              />
              <RangeRow
                label="D lr"
                value={learningRateToSlider(
                  discriminatorLearningRate,
                  D_LR_MIN,
                  D_LR_MAX
                )}
                min={0}
                max={1}
                step={0.005}
                display={discriminatorLearningRate.toExponential(1)}
                testid="discriminator-learning-rate-control"
                onChange={(value) => {
                  const next = sliderToLearningRate(value, D_LR_MIN, D_LR_MAX);
                  setDiscriminatorLearningRate(next);
                  handleRef.current?.setDiscriminatorLearningRate(next);
                }}
              />
              <div className="tui-static-row" data-testid="learning-rate-ratio">
                <span>D / G</span>
                <strong>
                  {(discriminatorLearningRate / generatorLearningRate).toFixed(2)}×
                </strong>
              </div>
              {isWta ? (
                <>
                  <RangeRow
                    label="guesses K"
                    value={runtime.k}
                    min={2}
                    max={12}
                    step={1}
                    display={`${runtime.k}`}
                    testid="adversary-k-control"
                    onChange={(value) =>
                      setRuntime((previous) => ({ ...previous, k: Math.round(value) }))
                    }
                  />
                  <RangeRow
                    label="relax ε"
                    value={runtime.relaxEps}
                    min={0}
                    max={0.45}
                    step={0.01}
                    display={runtime.relaxEps.toFixed(2)}
                    testid="adversary-epsilon-control"
                    onChange={(value) =>
                      setRuntime((previous) => ({ ...previous, relaxEps: value }))
                    }
                  />
                </>
              ) : (
                <div className="tui-static-row">
                  <span>predictor</span>
                  <strong>single-head control</strong>
                </div>
              )}
              <p className="tui-note restart-note">
                target, loss, tuple, observer, K and ε rebuild GPU pipelines
              </p>
              {isAgreeDisagree ? (
                <div className="tui-static-row" data-testid="color-mode-control">
                  <span>color</span>
                  <strong>RGB · A / B / derived C</strong>
                </div>
              ) : (
                <Segmented
                  label="color"
                  value={colorMode.tag}
                  testid="color-mode-control"
                  choices={[
                    { value: "velocity", label: "VEL" },
                    { value: "surprise-raw", label: "RAW" },
                    { value: "surprise-per-unit", label: "PER UNIT" },
                  ]}
                  onChange={(tag) =>
                    updateColor(
                      tag === "velocity"
                        ? { tag: "velocity" }
                        : {
                            tag,
                            colormap:
                              colorMode.tag !== "velocity" ? colorMode.colormap : "inferno",
                          }
                    )
                  }
                />
              )}
              {!isAgreeDisagree && colorMode.tag !== "velocity" && (
                <Segmented
                  label="map"
                  value={colorMode.colormap}
                  testid="colormap-control"
                  choices={CMAPS.map((name) => ({ value: name, label: name.toUpperCase() }))}
                  onChange={(colormap) => updateColor({ ...colorMode, colormap })}
                />
              )}
            </ControlSection>
          )}

          {adversary && (
            <ControlSection title="diagnostics" testid="adversary-diagnostics">
              {telemetry.tag === "on" ? (
                <>
                  <div className="diagnostic-row">
                    <span className="diagnostic-name">
                      {runtime.loss.tag === "angle-scale-hold"
                        ? `D joint · ${telemetry.variant}`
                        : telemetry.variant}
                    </span>
                    <Sparkline data={history} />
                    <strong>{telemetry.surprise.toExponential(1)}</strong>
                  </div>
                  <div className="diagnostic-row">
                    <span>heads</span>
                    <HeadBars fractions={telemetry.winFractions} />
                    <strong
                      className={`health health-${telemetry.health.tag}`}
                      data-testid="head-health"
                    >
                      {healthText(telemetry.health)}
                    </strong>
                  </div>
                  {telemetry.branches && (
                    <>
                      <div className="diagnostic-row" data-testid="disagree-head-health">
                        <span>A disagree</span>
                        <HeadBars fractions={telemetry.branches.disagree.winFractions} />
                        <strong
                          className={`health health-${telemetry.branches.disagree.health.tag}`}
                        >
                          {healthText(telemetry.branches.disagree.health)}
                        </strong>
                      </div>
                      <div className="diagnostic-row" data-testid="agree-head-health">
                        <span>B agree</span>
                        <HeadBars fractions={telemetry.branches.agree.winFractions} />
                        <strong
                          className={`health health-${telemetry.branches.agree.health.tag}`}
                        >
                          {healthText(telemetry.branches.agree.health)}
                        </strong>
                      </div>
                    </>
                  )}
                </>
              ) : (
                <div className="tui-static-row">
                  <span>game</span>
                  <strong>initialising…</strong>
                </div>
              )}
              {colorMode.tag !== "velocity" && surpriseSpan && (
                <div className="span-readout" data-testid="surprise-span">
                  {colorMode.tag === "surprise-raw" ? "RAW" : "PER UNIT"} · p2{" "}
                  {surpriseSpan.lo.toExponential(1)} · p98{" "}
                  {surpriseSpan.hi.toExponential(1)} ·{" "}
                  {Math.round(surpriseSpan.covered * 100)}%
                  {surpriseSpan.collapsed ? " · FLAT" : ""}
                </div>
              )}
            </ControlSection>
          )}
        </div>
      </aside>

      <nav
        className="gallery-radio"
        role="radiogroup"
        aria-label="Art piece"
        data-testid="art-piece-gallery"
      >
        {GALLERY.map((galleryPiece, index) => (
          <button
            key={galleryPiece.name}
            type="button"
            role="radio"
            aria-checked={index === runtime.piece}
            data-active={index === runtime.piece ? "true" : "false"}
            onClick={() => setRuntime(defaultsForPiece(index))}
          >
            {galleryPiece.name}
          </button>
        ))}
      </nav>
    </main>
  );
}

createRoot(container).render(<App />);
