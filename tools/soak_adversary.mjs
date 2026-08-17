/**
 * Browser release soak for every adversarial gallery piece.
 *
 *   node tools/soak_adversary.mjs <pieceKey|all> [baseURL] [durationSec] [sampleSec]
 *   node tools/soak_adversary.mjs --self-test
 *
 * Examples:
 *   node tools/soak_adversary.mjs agree http://localhost:1234/index.html 300 10
 *   node tools/soak_adversary.mjs all   http://localhost:1234/index.html 300 10
 *
 * Each piece runs in a fresh page; `all` serializes them because they share one
 * GPU. Screenshots, the complete sample stream and the summary are written
 * under output/playwright/adversary-soak/. Nothing is written to docs/ or a
 * machine/session-specific scratch directory.
 *
 * Current release gates:
 *   - the selected piece is the one producing telemetry, on the fused path;
 *   - zero page errors and console errors;
 *   - no NaN/Infinity in HUD or diagnostic text;
 *   - every required fused datum parses and remains finite;
 *   - the FPS HUD keeps mutating throughout the run (not merely still visible);
 *   - tensor count does not grow beyond a small post-warmup allowance;
 *   - confirmed predictor pileup / a flat surprise span hard-fails;
 *   - when speed is exposed, its trailing mean must remain below the configured
 *     fraction of the componentwise velocity clip;
 *   - when surprise coverage is exposed, final coverage must reach the
 *     configured floor.
 *
 * Deliberately NOT asserted:
 *   - no tfjs `learn N ms` line (the shipped path is fused);
 *   - no obsolete "K>modes" label;
 *   - no claim that the single predictor's residual must decay. The strict
 *     force target moves while the generator trains, so decay is not a release
 *     invariant.
 */

import puppeteer from "puppeteer";
import path from "node:path";
import fs from "node:fs";
import { fileURLToPath } from "node:url";

export const PIECES = Object.freeze({
  single: "Adversary · Single (control)",
  wta8: "Adversary · WTA K=8",
  pair4: "Adversary · Pair WTA K=4",
  tri6: "Adversary · Tri WTA K=6",
  quad6: "Adversary · Quad WTA K=6",
  agree: "Adversary · Agree + Disagree RGB",
  weave: "Adversary · Chaos Weave",
  // The DEFAULT piece. Only fused since the hashgrid adversary port, so the
  // "every sample reports the fused adversary" gate below is the end-to-end
  // proof of that work — it fails on any build where fusedAdvOk still excludes
  // hashgrid encodings.
  hashgrid: "Adversary · Pair · HashGrid · Curl",
});

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, "..");
const DEFAULT_BASE = "http://localhost:1234/index.html";

const TENSOR_SLACK = numberEnv("SOAK_TENSOR_SLACK", 12);
const MAX_SPEED_FRACTION = numberEnv("SOAK_MAX_SPEED_FRACTION", 0.95);
const MIN_COVERAGE = numberEnv("SOAK_MIN_COVERAGE", 0.9);
const WARMUP_MS = numberEnv("SOAK_WARMUP_MS", 4000);

function numberEnv(name, fallback) {
  const raw = process.env[name];
  if (raw === undefined) return fallback;
  const value = Number(raw);
  if (!Number.isFinite(value)) {
    throw new Error(`${name} must be finite, got ${JSON.stringify(raw)}`);
  }
  return value;
}

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const median = (values) => {
  const sorted = [...values].sort((a, b) => a - b);
  if (!sorted.length) return NaN;
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2
    ? sorted[mid]
    : 0.5 * (sorted[mid - 1] + sorted[mid]);
};
const finite = (value) => typeof value === "number" && Number.isFinite(value);
const slugTime = () => new Date().toISOString().replace(/[:.]/g, "-");

const NUMBER = String.raw`([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)`;
const extract = (text, expression) => {
  const match = text.match(expression);
  return match ? Number(match[1]) : null;
};

/**
 * Parse the current fused HUD plus the React diagnostic span.
 * Exported so `--self-test` and future tooling can gate the parser without
 * booting a browser.
 */
export function parseTelemetry(hud, diagnostic = "") {
  const all = `${hud}\n${diagnostic}`;
  const healthMatch = hud.match(/heads\s+([▁▂▃▄▅▆▇█]+)([^\n]*)/);
  const healthText = healthMatch?.[2]?.trim() ?? "";
  const coverage =
    extract(hud, /covered\s+([\d.]+)%/) ??
    extract(diagnostic, /(?:^|[·\s])([\d.]+)%\s*(?:·|$)/);
  const spanLo = extract(all, new RegExp(String.raw`p2\s+${NUMBER}`));
  const spanHi = extract(all, new RegExp(String.raw`p98\s+${NUMBER}`));
  const speed = extract(hud, new RegExp(String.raw`speed\s+${NUMBER}\s+px\/f`));
  const speedClip = extract(hud, new RegExp(String.raw`\(clip\s+${NUMBER}\)`));
  return {
    piece: hud.split("\n")[0]?.trim() || null,
    fused: /\(fused\)/.test(hud),
    fps: extract(hud, new RegExp(String.raw`FPS\s+${NUMBER}`)),
    frameMs: extract(
      hud,
      /FPS\s+[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?\s+\(([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s+ms\)/
    ),
    rolloutMs: extract(hud, new RegExp(String.raw`rollout\s+${NUMBER}\s+ms`)),
    optimMs: extract(hud, new RegExp(String.raw`optim\s+${NUMBER}\s+ms`)),
    loss: extract(hud, new RegExp(String.raw`\bloss\s+${NUMBER}`)),
    surprise: extract(hud, new RegExp(String.raw`surprise\s+${NUMBER}`)),
    pred: extract(hud, new RegExp(String.raw`\bpred\s+${NUMBER}`)),
    bars: healthMatch?.[1] ?? null,
    health: /HEAD COLLAPSE/.test(healthText)
      ? "pileup"
      : /skew \+ separated/.test(healthText)
        ? "separated-unresolved"
        : /skew \(unprobed\)/.test(healthText)
          ? "unresolved"
          : "ok",
    speed,
    speedClip,
    spanLo,
    spanHi,
    coverage,
    flatSpan: /ADVERSARY COLLAPSED|\bFLAT\b/.test(all),
    tensors: extract(hud, /tensors\s+(\d+)/),
    branchA: {
      surprise: extract(
        hud,
        new RegExp(String.raw`A\s+disagree\s+sur\s+${NUMBER}`)
      ),
      pred: extract(
        hud,
        new RegExp(String.raw`A\s+disagree[^\n]*\bpred\s+${NUMBER}`)
      ),
    },
    branchB: {
      surprise: extract(
        hud,
        new RegExp(String.raw`B\s+agree\s+sur\s+${NUMBER}`)
      ),
      pred: extract(
        hud,
        new RegExp(String.raw`B\s+agree[^\n]*\bpred\s+${NUMBER}`)
      ),
    },
    cDisplayOnly: /C\s+blend\s+display only \(zero direct loss\)/.test(hud),
    rawHasNonFinite: /(^|[^\w])(?:NaN|[+-]?Infinity)(?=$|[^\w])/i.test(all),
  };
}

function parserSelfTest() {
  const ordinary = [
    "Adversary · Pair WTA K=4",
    "backend webgpu  render=90000 train=512",
    "FPS     60.0  (16.7 ms)",
    "rollout 1.11 ms  optim 0.72 ms  loss 0.000",
    "advect  0.42 ms  render 1.30 ms  (gpu)",
    "adv     wta  w 0.015  (fused)",
    "surprise 2.10e-2  pred 0.0210",
    "heads   ▂█▁▁  skew + separated (support unprobed, 2.20·rms)",
    "speed   8.10 px/f  (clip 22.0)",
    "span    p2 1.00e-3 p50 2.00e-2 p98 8.00e-2",
    "tensors 52",
  ].join("\n");
  const a = parseTelemetry(ordinary, "p2 1.0e-3 · p98 8.0e-2 · 97%");
  const agree = [
    "Adversary · Agree + Disagree RGB",
    "backend webgpu  render=90000 train=512",
    "FPS     58.2  (17.2 ms)",
    "rollout 1.20 ms  optim 0.80 ms  loss 0.000",
    "advect  0.50 ms  render 1.50 ms  (gpu)",
    "adv     agree+disagree  w 0.012  (fused)",
    "surprise 3.00e-2  pred 0.0610",
    "heads   ▄▄▄▄",
    "A disagree  sur 4.00e-2 pred 4.10e-2",
    "B agree     sur 2.00e-2 pred 2.00e-2",
    "C blend     display only (zero direct loss)",
    "speed   7.00 px/f  (clip 22.0)",
    "tensors 58",
  ].join("\n");
  const b = parseTelemetry(agree);
  const checks = [
    [a.fused && a.fps === 60 && a.rolloutMs === 1.11, "ordinary fused timings"],
    [
      a.health === "separated-unresolved" && a.coverage === 97,
      "current health + diagnostic coverage",
    ],
    [a.speed === 8.1 && a.speedClip === 22, "speed/clip"],
    [
      b.branchA.surprise === 0.04 &&
        b.branchB.pred === 0.02 &&
        b.cDisplayOnly,
      "Agree A/B/C telemetry",
    ],
    [
      parseTelemetry(ordinary.replace("loss 0.000", "loss NaN")).rawHasNonFinite,
      "NaN detection",
    ],
  ];
  let failed = 0;
  for (const [condition, message] of checks) {
    console.log(`${condition ? "PASS" : "FAIL"}  parser: ${message}`);
    if (!condition) failed++;
  }
  console.log(failed ? `\n${failed} PARSER FAILURE(S)` : "\nPARSER SELF-TEST PASS");
  return failed;
}

function validateArgs(durationSec, sampleSec) {
  if (!(Number.isFinite(durationSec) && durationSec >= 1)) {
    throw new Error(`durationSec must be finite and >= 1, got ${durationSec}`);
  }
  if (!(Number.isFinite(sampleSec) && sampleSec >= 0.5)) {
    throw new Error(`sampleSec must be finite and >= 0.5, got ${sampleSec}`);
  }
  if (durationSec < sampleSec) {
    throw new Error(
      `durationSec (${durationSec}) must be >= sampleSec (${sampleSec}) so ` +
        "telemetry progress can be observed"
    );
  }
}

async function installTelemetryObserver(page) {
  await page.evaluate(() => {
    const hud = document.querySelector('[data-testid="fps-hud"]');
    if (!hud) throw new Error("missing [data-testid=fps-hud]");
    window.__adversarySoakObserver?.observer?.disconnect?.();
    const state = {
      hud,
      mutations: 0,
      lastMutationMs: performance.now(),
      observer: null,
    };
    state.observer = new MutationObserver(() => {
      state.mutations++;
      state.lastMutationMs = performance.now();
    });
    state.observer.observe(hud, {
      subtree: true,
      childList: true,
      characterData: true,
    });
    window.__adversarySoakObserver = state;
  });
}

async function readTelemetry(page) {
  return page.evaluate(() => {
    const hud = document.querySelector('[data-testid="fps-hud"]');
    const diagnostic = document.querySelector('[data-testid="surprise-span"]');
    const health = document.querySelector('[data-testid="head-health"]');
    const selected = document.querySelector(
      '[data-testid="art-piece-gallery"] [role="radio"][aria-checked="true"]'
    );
    const state = window.__adversarySoakObserver;
    return {
      hud: hud?.textContent ?? "",
      diagnostic: diagnostic?.textContent ?? "",
      health: health?.textContent ?? "",
      selected: selected?.textContent?.trim() ?? "",
      mutations: state?.mutations ?? -1,
      staleMs: state ? performance.now() - state.lastMutationMs : Infinity,
    };
  });
}

async function exerciseColorDiagnostics(page, key) {
  return page.evaluate(async (pieceKey) => {
    const sleepInPage = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
    const row = document.querySelector('[data-testid="color-mode-control"]');
    if (!row) throw new Error("missing color-mode control");
    if (pieceKey === "agree") {
      return {
        fixedRgb: /RGB\s*·\s*A\s*\/\s*B\s*\/\s*derived C/.test(row.textContent || ""),
        choices: [...row.querySelectorAll("button")].map((button) =>
          (button.textContent || "").trim()
        ),
      };
    }

    window.__diagnosticCanvasIdentity = document.querySelector("canvas");
    const click = async (label) => {
      const button = [...row.querySelectorAll("button")].find(
        (candidate) => (candidate.textContent || "").trim() === label
      );
      if (!button) throw new Error(`missing ${label} color choice`);
      button.click();
      for (let i = 0; i < 80; i++) {
        const active = button.getAttribute("aria-checked") === "true";
        const span = document.querySelector('[data-testid="surprise-span"]')?.textContent || "";
        if (
          active &&
          (label === "VEL" ||
            (label === "RAW" && /\bRAW\b/.test(span)) ||
            (label === "PER UNIT" && /\bPER UNIT\b/.test(span)))
        ) {
          return span;
        }
        await sleepInPage(50);
      }
      throw new Error(`${label} diagnostic did not become active`);
    };

    await click("VEL");
    const raw = await click("RAW");
    await sleepInPage(400);
    const rawSettled =
      document.querySelector('[data-testid="surprise-span"]')?.textContent || raw;
    const unit = await click("PER UNIT");
    await sleepInPage(400);
    const unitSettled =
      document.querySelector('[data-testid="surprise-span"]')?.textContent || unit;
    return {
      fixedRgb: false,
      choices: [...row.querySelectorAll("button")].map((button) =>
        (button.textContent || "").trim()
      ),
      canvasStable:
        window.__diagnosticCanvasIdentity === document.querySelector("canvas"),
      raw: rawSettled,
      unit: unitSettled,
    };
  }, key);
}

async function runPiece(browser, key, pieceName, options) {
  const pieceDir = path.join(options.runDir, key);
  fs.mkdirSync(pieceDir, { recursive: true });
  const page = await browser.newPage();
  await page.setViewport({ width: 1280, height: 800 });

  const pageErrors = [];
  const consoleErrors = [];
  page.on("pageerror", (error) => pageErrors.push(error.message));
  page.on("console", (message) => {
    if (message.type() === "error") consoleErrors.push(message.text());
  });

  const samples = [];
  let displayChecks = null;
  let thrown = null;
  const startedAt = new Date().toISOString();
  try {
    await page.goto(options.base, {
      waitUntil: "domcontentloaded",
      timeout: 30_000,
    });
    await page.waitForSelector('[data-testid="art-piece-gallery"]', {
      timeout: 30_000,
    });
    await page.evaluate((name) => {
      const button = [...document.querySelectorAll("button")].find((candidate) =>
        (candidate.textContent || "").includes(name)
      );
      if (!button) throw new Error(`no gallery button containing '${name}'`);
      button.click();
    }, pieceName);
    await page.waitForFunction(
      (name) =>
        document.querySelector('[data-testid="fps-hud"]')?.textContent?.includes(name),
      { timeout: 30_000 },
      pieceName
    );
    await sleep(WARMUP_MS);
    displayChecks = await exerciseColorDiagnostics(page, key);
    await installTelemetryObserver(page);
    await sleep(500);

    const sampleCount = Math.floor(options.durationSec / options.sampleSec) + 1;
    const midpoint = Math.floor((sampleCount - 1) / 2);
    const t0 = Date.now();
    for (let i = 0; i < sampleCount; i++) {
      const raw = await readTelemetry(page);
      const parsed = parseTelemetry(raw.hud, `${raw.diagnostic}\n${raw.health}`);
      const sample = {
        ...parsed,
        selected: raw.selected,
        mutations: raw.mutations,
        staleMs: raw.staleMs,
        tSec: (Date.now() - t0) / 1000,
      };
      samples.push(sample);
      console.log(
        `[${key}] t=${sample.tSec.toFixed(1).padStart(6)}s ` +
          `fps=${sample.fps ?? "-"} rollout=${sample.rolloutMs ?? "-"}ms ` +
          `optim=${sample.optimMs ?? "-"}ms sur=${sample.surprise ?? "-"} ` +
          `pred=${sample.pred ?? "-"} speed=${sample.speed ?? "-"}/${sample.speedClip ?? "-"} ` +
          `cov=${sample.coverage ?? "-"}% health=${sample.health} ` +
          `mut=${sample.mutations} stale=${Math.round(sample.staleMs)}ms`
      );
      if (i === 0 || i === midpoint || i === sampleCount - 1) {
        const label = i === 0 ? "start" : i === midpoint ? "mid" : "end";
        await page.screenshot({
          path: path.join(pieceDir, `${label}_${Math.round(sample.tSec)}s.png`),
        });
      }
      if (i < sampleCount - 1) await sleep(options.sampleSec * 1000);
    }
  } catch (error) {
    thrown = error instanceof Error ? error.stack ?? error.message : String(error);
    try {
      await page.screenshot({ path: path.join(pieceDir, "failure.png") });
    } catch (_) {}
  } finally {
    await page.close();
  }

  const failures = [];
  const fail = (condition, message) => {
    console.log(`${condition ? "PASS" : "FAIL"}  [${key}] ${message}`);
    if (!condition) failures.push(message);
  };

  console.log(`\n=== gates: ${pieceName} ===`);
  fail(thrown === null, `runner completed${thrown ? `: ${thrown.split("\n")[0]}` : ""}`);
  fail(
    pageErrors.length === 0,
    `zero page errors${pageErrors.length ? `: ${pageErrors[0]}` : ""}`
  );
  fail(
    consoleErrors.length === 0,
    `zero console errors${consoleErrors.length ? `: ${consoleErrors[0]}` : ""}`
  );
  fail(samples.length >= 2, `at least two telemetry samples (${samples.length})`);
  fail(samples.every((sample) => sample.selected.includes(pieceName)), "gallery selection stayed active");
  fail(samples.every((sample) => sample.piece === pieceName), "HUD stayed on the selected piece");
  fail(samples.every((sample) => sample.fused), "every sample reports the fused adversary");
  fail(samples.every((sample) => !sample.rawHasNonFinite), "no NaN/Infinity in telemetry");
  fail(
    samples.every(
      (sample) =>
        finite(sample.fps) &&
        sample.fps > 0 &&
        finite(sample.frameMs) &&
        finite(sample.rolloutMs) &&
        finite(sample.optimMs) &&
        finite(sample.loss) &&
        finite(sample.surprise) &&
        finite(sample.pred) &&
        finite(sample.tensors)
    ),
    "every current fused HUD datum parsed and is finite"
  );

  const staleLimitMs = Math.max(3000, options.sampleSec * 2000);
  fail(
    samples.slice(1).every(
      (sample, index) =>
        sample.mutations > samples[index].mutations && sample.staleMs < staleLimitMs
    ),
    `telemetry kept advancing (mutation delta > 0, stale < ${staleLimitMs} ms)`
  );

  if (samples.length) {
    const tensorBase = samples[1]?.tensors ?? samples[0].tensors;
    const tensorEnd = samples.at(-1).tensors;
    fail(
      finite(tensorBase) &&
        finite(tensorEnd) &&
        tensorEnd - tensorBase <= TENSOR_SLACK,
      `tensor count stable: ${tensorBase} -> ${tensorEnd} (slack ${TENSOR_SLACK})`
    );
  }
  fail(
    samples.every((sample) => sample.health !== "pileup"),
    "no confirmed predictor-head pileup"
  );
  fail(samples.every((sample) => !sample.flatSpan), "surprise span never reports FLAT");

  if (key === "agree") {
    fail(
      displayChecks?.fixedRgb === true &&
        Array.isArray(displayChecks?.choices) &&
        displayChecks.choices.length === 0,
      "Agree keeps fixed RGB A/B/derived-C output and exposes no generic color radios"
    );
  } else {
    const rawParsed = parseTelemetry("", displayChecks?.raw ?? "");
    const unitParsed = parseTelemetry("", displayChecks?.unit ?? "");
    fail(
      displayChecks?.choices?.join("|") === "VEL|RAW|PER UNIT",
      "color control exposes VEL, RAW and PER UNIT"
    );
    fail(displayChecks?.canvasStable === true, "diagnostic toggles do not restart the artwork");
    fail(
      finite(rawParsed.spanLo) &&
        finite(rawParsed.spanHi) &&
        rawParsed.spanHi >= rawParsed.spanLo &&
        !rawParsed.flatSpan &&
        finite(unitParsed.spanLo) &&
        finite(unitParsed.spanHi) &&
        unitParsed.spanHi >= unitParsed.spanLo &&
        !unitParsed.flatSpan,
      // `>=` + not-FLAT, NOT strict `>`: the HUD prints these at 2 significant
      // figures, so a healthy-but-tight distribution rounds p2 == p98 and a
      // strict `>` fails on rounding noise (hashgrid failed 6/9 runs on this
      // while the app's exact SPAN_FLOOR test passed every time — see
      // agent_notes/2026-08-17_soak_flake_attribution.md §2/§5). The app's own
      // `· FLAT` flag is the authoritative collapse test; defer to it.
      "RAW and PER UNIT each expose a finite non-flat percentile span"
    );
    fail(
      /\bRAW\b/.test(displayChecks?.raw ?? "") &&
        /\bPER UNIT\b/.test(displayChecks?.unit ?? ""),
      "diagnostic span names the selected metric"
    );
  }

  const speedSamples = samples.filter(
    (sample) => finite(sample.speed) && finite(sample.speedClip) && sample.speedClip > 0
  );
  if (speedSamples.length) {
    const trailing = speedSamples.slice(-Math.min(3, speedSamples.length));
    const ratio = median(trailing.map((sample) => sample.speed / sample.speedClip));
    fail(
      ratio < MAX_SPEED_FRACTION,
      `trailing mean-speed/clip ${ratio.toFixed(3)} < ${MAX_SPEED_FRACTION}`
    );
  } else {
    console.log(`INFO  [${key}] speed/clip is not exposed; saturation gate not applicable`);
  }

  const coverageSamples = samples.filter((sample) => finite(sample.coverage));
  if (coverageSamples.length) {
    const finalCoverage = coverageSamples.at(-1).coverage / 100;
    fail(
      finalCoverage >= MIN_COVERAGE,
      `final surprise coverage ${(100 * finalCoverage).toFixed(1)}% >= ${(100 * MIN_COVERAGE).toFixed(1)}%`
    );
  } else {
    console.log(`INFO  [${key}] coverage is not exposed in this render mode`);
  }

  const spanSamples = samples.filter(
    (sample) => finite(sample.spanLo) && finite(sample.spanHi)
  );
  if (spanSamples.length) {
    fail(
      // `>=` + not-FLAT for the same reason as the RAW/PER-UNIT gate above:
      // 2-sig-fig HUD strings cannot support a strict inequality (2026-08-17
      // flake attribution). A genuinely collapsed span still fails via the
      // app's exact `· FLAT` flag.
      spanSamples.every(
        (sample) => sample.spanHi >= sample.spanLo && !sample.flatSpan
      ),
      "exposed surprise percentile span has p98 >= p2 and never FLAT"
    );
  }

  if (key === "agree") {
    fail(
      samples.every(
        (sample) =>
          finite(sample.branchA.surprise) &&
          finite(sample.branchA.pred) &&
          finite(sample.branchB.surprise) &&
          finite(sample.branchB.pred) &&
          sample.cDisplayOnly
      ),
      "Agree exposes finite A/B metrics and labels C display-only"
    );
  }

  if (samples.length >= 2) {
    const window = Math.min(3, samples.length);
    const first = median(samples.slice(0, window).map((sample) => sample.surprise));
    const last = median(samples.slice(-window).map((sample) => sample.surprise));
    console.log(
      `INFO  [${key}] surprise trajectory ${first.toExponential(3)} -> ` +
        `${last.toExponential(3)} (reported, no obsolete decay/floor assertion)`
    );
    console.log(
      `INFO  [${key}] median FPS ${median(samples.map((sample) => sample.fps)).toFixed(1)}, ` +
        `rollout ${median(samples.map((sample) => sample.rolloutMs)).toFixed(3)} ms, ` +
        `optim ${median(samples.map((sample) => sample.optimMs)).toFixed(3)} ms`
    );
  }

  const result = {
    key,
    pieceName,
    startedAt,
    base: options.base,
    durationSec: options.durationSec,
    sampleSec: options.sampleSec,
    thresholds: {
      tensorSlack: TENSOR_SLACK,
      maxSpeedFraction: MAX_SPEED_FRACTION,
      minCoverage: MIN_COVERAGE,
      warmupMs: WARMUP_MS,
    },
    assumptions: [
      "FPS HUD mutates at least once per sample interval on a running render loop.",
      "mean speed is |v| while maxVelocity is the componentwise clip; the trailing ratio threshold is an operational saturation alarm, not a proof every component is clipped.",
      "coverage is gated only when the selected render mode exposes it.",
      "single-predictor surprise decay is not a release invariant.",
      "Agree+Disagree must expose two finite branch metrics and a display-only C label.",
      "Display diagnostics are output-only: the Metal kernel suite proves toggling planes leaves gradients and weights bit-identical; the browser gate proves the canvas/runtime is not restarted.",
    ],
    displayChecks,
    samples,
    pageErrors,
    consoleErrors,
    thrown,
    failures,
  };
  fs.writeFileSync(
    path.join(pieceDir, "samples.json"),
    JSON.stringify(result, null, 2)
  );
  console.log(
    failures.length
      ? `\n${failures.length} GATE FAILURE(S) [${key}]`
      : `\nALL GATES PASS [${key}]`
  );
  return result;
}

async function main() {
  const requested = process.argv[2];
  if (requested === "--self-test") {
    process.exitCode = parserSelfTest() ? 1 : 0;
    return;
  }
  if (!requested || (requested !== "all" && !PIECES[requested])) {
    console.error(
      `usage: node tools/soak_adversary.mjs ` +
        `<${[...Object.keys(PIECES), "all"].join("|")}> ` +
        `[baseURL] [durationSec] [sampleSec]`
    );
    process.exitCode = 2;
    return;
  }

  const base = process.argv[3] || DEFAULT_BASE;
  const durationSec = Number(process.argv[4] || 300);
  const sampleSec = Number(process.argv[5] || 10);
  validateArgs(durationSec, sampleSec);

  const runDir = path.join(
    ROOT,
    "output",
    "playwright",
    "adversary-soak",
    slugTime()
  );
  fs.mkdirSync(runDir, { recursive: true });

  const browser = await puppeteer.launch({
    headless: "new",
    args: [
      "--enable-unsafe-webgpu",
      "--enable-features=Vulkan,WebGPU",
      "--use-angle=metal",
      "--window-size=1280,800",
    ],
  });
  const keys = requested === "all" ? Object.keys(PIECES) : [requested];
  const results = [];
  try {
    for (const key of keys) {
      results.push(
        await runPiece(browser, key, PIECES[key], {
          base,
          durationSec,
          sampleSec,
          runDir,
        })
      );
    }
  } finally {
    await browser.close();
  }

  const totalFailures = results.reduce(
    (sum, result) => sum + result.failures.length,
    0
  );
  const summary = {
    runDir,
    base,
    durationSec,
    sampleSec,
    pieces: results.map((result) => ({
      key: result.key,
      pieceName: result.pieceName,
      passed: result.failures.length === 0,
      failures: result.failures,
    })),
    totalFailures,
  };
  fs.writeFileSync(
    path.join(runDir, "summary.json"),
    JSON.stringify(summary, null, 2)
  );
  console.log(`\nARTIFACTS ${runDir}`);
  console.log(
    totalFailures
      ? `${totalFailures} TOTAL GATE FAILURE(S)`
      : `ALL REQUESTED ADVERSARY SOAKS PASS`
  );
  process.exitCode = totalFailures ? 1 : 0;
}

await main();
