#!/usr/bin/env node
/**
 * Capture a Chrome DevTools trace for the WebGPU 3D splat page.
 *
 * This is complementary to the Bun/WebGPU timestamp profilers. It does not
 * expose Metal occupancy/cache counters, but it can show Dawn/GPU scheduling,
 * command-buffer behavior, queue gaps, and pipeline/compile events.
 *
 * Usage:
 *   node tools/webgpu_trace.mjs [url] [durationMs] [outDir]
 *   GRID9=1 DIRECT_GRID=1 node tools/webgpu_trace.mjs http://localhost:1234/splat3d.html 12000 /tmp/nffa_trace
 *   HEADED=1 node tools/webgpu_trace.mjs
 */
import fs from "node:fs";
import path from "node:path";
import puppeteer from "puppeteer";

const url = process.argv[2] || "http://localhost:1234/splat3d.html";
const durationMs = Number.parseInt(process.argv[3] || "12000", 10);
const outDir = process.argv[4] || "/tmp/nffa_webgpu_trace";
const headed = process.env.HEADED === "1";
const grid9 = process.env.GRID9 === "1";
const directGrid = process.env.DIRECT_GRID === "1";
const prompt = process.env.PROMPT || "a photo of a cat";
const setupTimeoutMs = Number.parseInt(process.env.SETUP_TIMEOUT_MS || "180000", 10);
const profileDir = process.env.TRACE_PROFILE_DIR || "/tmp/nffa_webgpu_trace_profile";

const chromePath =
  process.env.CHROME_PATH ||
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";

const categories = [
  "benchmark",
  "gpu",
  "gpu.angle",
  "disabled-by-default-gpu.dawn",
  "disabled-by-default-gpu.graphite.dawn",
  "disabled-by-default-webgpu",
  "disabled-by-default-skia.gpu",
  "disabled-by-default-viz.gpu_composite_time",
  "devtools.timeline",
  "disabled-by-default-devtools.timeline",
  "blink",
  "blink.user_timing",
  "renderer.scheduler",
  "disabled-by-default-renderer.scheduler",
  "v8",
  "v8.execute",
  "toplevel",
];

fs.mkdirSync(outDir, { recursive: true });
const stamp = new Date().toISOString().replace(/[:.]/g, "-");
const tracePath = path.join(outDir, `webgpu_trace_${stamp}.json`);
const shotPath = path.join(outDir, `webgpu_trace_${stamp}.png`);

const launchOptions = {
  headless: headed ? false : "new",
  userDataDir: profileDir,
  args: [
    "--no-sandbox",
    "--use-angle=metal",
    "--ignore-gpu-blocklist",
    "--enable-webgpu-developer-features",
    "--enable-unsafe-webgpu",
    "--disable-background-timer-throttling",
    "--disable-renderer-backgrounding",
    "--window-size=1280,900",
  ],
};
if (fs.existsSync(chromePath)) {
  launchOptions.executablePath = chromePath;
}

const browser = await puppeteer.launch(launchOptions);
let page;

const logs = [];

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

async function status() {
  if (!page) return { error: "page not created" };
  return await page.evaluate(() => {
    const s = window.__splat3d || {};
    return {
      gpu: s.gpu ?? null,
      ready: s.ready ?? null,
      running: s.running ?? null,
      step: s.step ?? null,
      phase: s.phase ?? null,
      clipLayout: s.clipLayout ?? null,
      gridDirectRaster: s.gridDirectRaster ?? null,
      viewsPerStep: s.viewsPerStep ?? null,
      clipBatchSize: s.clipBatchSize ?? null,
      error: s.error ?? null,
    };
  });
}

async function waitForIdleReady() {
  await page.waitForFunction(
    () => {
      const s = window.__splat3d;
      return s && s.ready && !s.error && (s.phase === "idle" || s.phase === "run");
    },
    { timeout: setupTimeoutMs }
  );
}

async function selectAndWait(selector, value) {
  await page.select(selector, value);
  await page.waitForFunction(
    () => {
      const s = window.__splat3d;
      return s && s.ready && !s.error && s.phase === "idle";
    },
    { timeout: setupTimeoutMs }
  );
}

try {
  page = await browser.newPage();
  await page.setViewport({ width: 1280, height: 900 });

  page.on("console", (m) => logs.push(`[${m.type()}] ${m.text()}`));
  page.on("pageerror", (e) => logs.push(`[pageerror] ${e.message}`));
  page.on("requestfailed", (r) => logs.push(`[reqfail] ${r.url()} ${r.failure()?.errorText}`));

  await page.goto(url, { waitUntil: "domcontentloaded", timeout: 60000 });
  await waitForIdleReady();

  await page.$eval("#prompt", (el, value) => {
    el.value = value;
    el.dispatchEvent(new Event("input", { bubbles: true }));
  }, prompt);

  if (grid9) {
    await selectAndWait("#clipLayout", "grid9_close2");
    if (directGrid) await selectAndWait("#gridRasterMode", "direct80");
  }

  await page.click("#optimize");
  await page.waitForFunction(
    () => {
      const s = window.__splat3d;
      return s && s.ready && s.running && s.phase === "run" && s.step >= 2;
    },
    { timeout: setupTimeoutMs }
  );

  const before = await status();
  await page.tracing.start({ path: tracePath, categories });
  await sleep(Number.isFinite(durationMs) ? Math.max(1000, durationMs) : 12000);
  await page.tracing.stop();
  const after = await status();
  await page.screenshot({ path: shotPath });

  let summary = null;
  try {
    const trace = JSON.parse(fs.readFileSync(tracePath, "utf8"));
    const events = Array.isArray(trace.traceEvents) ? trace.traceEvents : [];
    const interesting = new Map();
    for (const ev of events) {
      const name = String(ev.name || "");
      const cat = String(ev.cat || "");
      if (!/dawn|gpu|webgpu|command|queue|pipeline|compute/i.test(`${cat} ${name}`)) continue;
      const key = `${cat} :: ${name}`.slice(0, 160);
      interesting.set(key, (interesting.get(key) || 0) + 1);
    }
    summary = {
      events: events.length,
      interesting: [...interesting.entries()]
        .sort((a, b) => b[1] - a[1])
        .slice(0, 20),
    };
  } catch (e) {
    summary = { error: String(e?.message ?? e) };
  }

  console.log(`URL ${url}`);
  console.log(`TRACE ${tracePath}`);
  console.log(`SCREENSHOT ${shotPath}`);
  console.log(`BEFORE ${JSON.stringify(before)}`);
  console.log(`AFTER ${JSON.stringify(after)}`);
  console.log(`CATEGORIES ${categories.join(",")}`);
  console.log(`SUMMARY ${JSON.stringify(summary)}`);
  console.log(`--- CONSOLE (${logs.length}) ---`);
  for (const line of logs.slice(0, 80)) console.log(line);
} catch (e) {
  const failureShot = shotPath.replace(/\.png$/, "_failed.png");
  let pageStatus = null;
  let body = "";
  try {
    pageStatus = await status();
    body = await page.evaluate(() => document.body?.innerText?.slice(0, 1200) || "");
    await page.screenshot({ path: failureShot });
  } catch (inner) {
    pageStatus = { error: String(inner?.message ?? inner) };
  }
  console.error(`TRACE_CAPTURE_FAILED ${String(e?.message ?? e)}`);
  console.error(`URL ${url}`);
  console.error(`SCREENSHOT ${failureShot}`);
  console.error(`STATUS ${JSON.stringify(pageStatus)}`);
  console.error(`BODY ${JSON.stringify(body)}`);
  console.error(`--- CONSOLE (${logs.length}) ---`);
  for (const line of logs.slice(0, 120)) console.error(line);
  process.exitCode = 1;
} finally {
  await browser.close();
}
