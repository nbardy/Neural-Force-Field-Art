// tools/splat/page_smoke.mjs — ACCEPTANCE GATE for the prompt→splats page
// (src/splat.html + src/splat_page.ts), against the ACTUALLY-RUNNING page on
// real headless-Metal Chrome (adapted from tools/qa_browser.mjs).
//
// It: (1) serves the repo root (so /dist/splat.html + the 82 MB vision weights
// are same-origin — see tools/splat/serve.mjs); (2) loads the page, asserts no
// page errors and navigator.gpu present; (3) types "a photo of a cat", clicks
// Optimize; (4) waits for the text model to load + ~150 optimize steps; (5)
// asserts the cosine readout ROSE from its initial value; (6) screenshots the
// canvas and asserts it is NON-BLANK (pixel variance above a floor). Prints the
// SCREENSHOT path + the cos trajectory.
//
// Prereqs:
//   uv run --with onnx --with numpy python tools/clip/compile_plan.py --train
//   npx parcel build --no-scope-hoist --public-url ./ src/splat.html
// Run:
//   node tools/splat/page_smoke.mjs
import { existsSync } from "node:fs";
import { join } from "node:path";
import puppeteer from "puppeteer";
import sharp from "sharp";
import { createRepoServer, ROOT } from "./serve.mjs";

const OUT = process.env.OUT_DIR || "/tmp";
const TARGET_STEPS = Number(process.env.STEPS ?? 150);
const COS_MARGIN = Number(process.env.COS_MARGIN ?? 0.005); // rose by at least this
const VAR_FLOOR = Number(process.env.VAR_FLOOR ?? 2.0); // luminance variance floor
// First run downloads the 161 MB fp32 text model; a persistent profile caches it.
const PROFILE = process.env.CHROME_PROFILE || "/tmp/splat_smoke_chrome_profile";
const BOOT_TIMEOUT = 90_000; // page ready (fetch 82 MB weights + build pipelines)
const RUN_TIMEOUT = Number(process.env.RUN_TIMEOUT ?? 360_000); // model dl + steps

if (!existsSync(join(ROOT, "dist", "splat.html"))) {
  console.error("FATAL: dist/splat.html missing. Build first:");
  console.error("  npx parcel build --no-scope-hoist --public-url ./ src/splat.html");
  process.exit(1);
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const fail = (msg) => {
  console.error(`\nGATE FAIL: ${msg}`);
  process.exitCode = 1;
};

const { server, base } = await createRepoServer();
const url = `${base}/dist/splat.html`;

const browser = await puppeteer.launch({
  headless: "new",
  userDataDir: PROFILE,
  args: [
    "--no-sandbox",
    "--use-angle=metal",
    "--ignore-gpu-blocklist",
    "--enable-webgpu-developer-features",
    "--window-size=800,900",
    // Keep the rAF optimize loop running at full speed — headless Chrome
    // otherwise throttles a "backgrounded" page's rAF to ~1/s after ~30s,
    // which stretches the 150-step gate from seconds to minutes.
    "--disable-background-timer-throttling",
    "--disable-backgrounding-occluded-windows",
    "--disable-renderer-backgrounding",
  ],
});

const page = await browser.newPage();
await page.setViewport({ width: 800, height: 900 });
const consoleLines = [];
const pageErrors = [];
page.on("console", (m) => consoleLines.push(`[${m.type()}] ${m.text()}`));
page.on("pageerror", (e) => pageErrors.push(e.message));

let ok = true;
try {
  console.log(`serving ${ROOT}\nloading ${url}`);
  await page.goto(url, { waitUntil: "domcontentloaded", timeout: 60_000 });

  // (1) navigator.gpu present + page reached ready (weights fetched, optimizer built)
  const gpu = await page.evaluate(() => !!navigator.gpu);
  if (!gpu) throw new Error("navigator.gpu absent in page");
  console.log(`navigator.gpu: present`);

  await page
    .waitForFunction(() => window.__splat && (window.__splat.ready || window.__splat.error), {
      timeout: BOOT_TIMEOUT,
      polling: 250,
    })
    .catch(() => {});
  let st = await page.evaluate(() => window.__splat);
  if (st?.error) throw new Error(`page boot error: ${st.error}`);
  if (!st?.ready) throw new Error(`page never became ready (phase=${st?.phase})`);
  console.log(`page ready (phase=${st.phase})`);

  // (2) type the prompt + click Optimize
  await page.click("#prompt", { clickCount: 3 });
  await page.type("#prompt", "a photo of a cat");
  await page.click("#optimize");
  console.log(`clicked Optimize — waiting for text model + ${TARGET_STEPS} steps…`);

  // (3) wait for the text model to load + TARGET_STEPS to run, sampling the cos
  const traj = [];
  const t0 = Date.now();
  let last = null;
  while (Date.now() - t0 < RUN_TIMEOUT) {
    st = await page.evaluate(() => window.__splat);
    if (st?.error) throw new Error(`page error during run: ${st.error}`);
    if (st?.cos !== null && st?.cos !== undefined) {
      if (!last || st.step !== last.step || st.cos !== last.cos) {
        traj.push({ step: st.step, cos: st.cos });
        last = { step: st.step, cos: st.cos };
      }
    }
    const el = Math.round((Date.now() - t0) / 1000);
    process.stdout.write(
      `\r  phase=${st?.phase} step=${st?.step} cos=${st?.cos?.toFixed?.(4) ?? "…"} (${el}s)   `
    );
    if (st?.step >= TARGET_STEPS + 8 && st?.initialCos !== null) break;
    await sleep(1000);
  }
  process.stdout.write("\n");
  if (pageErrors.length) throw new Error(`pageerror: ${pageErrors.join(" | ")}`);
  if (!st || st.step < TARGET_STEPS) throw new Error(`only ${st?.step} steps ran (need ${TARGET_STEPS})`);

  const initialCos = st.initialCos;
  const finalCos = st.cos;
  console.log(`cos trajectory (${traj.length} samples):`);
  for (const p of traj) console.log(`  step ${String(p.step).padStart(4)}  cos ${p.cos.toFixed(5)}`);
  console.log(`initial cos = ${initialCos?.toFixed(5)}   final cos = ${finalCos?.toFixed(5)}   Δ = ${(finalCos - initialCos).toFixed(5)}`);

  // (4) assert the cos ROSE
  if (!(finalCos - initialCos > COS_MARGIN)) {
    ok = false;
    fail(`cos did not rise enough: Δ=${(finalCos - initialCos).toFixed(5)} (need > ${COS_MARGIN})`);
  } else {
    console.log(`cos ROSE by ${(finalCos - initialCos).toFixed(5)} (> ${COS_MARGIN}) — PASS`);
  }

  // (5) screenshot the canvas + NON-BLANK assertion. We measure the ACTUAL
  // displayed pixels via the puppeteer screenshot (the browser compositor path),
  // NOT canvas.drawImage — a WebGPU canvas's drawing buffer is not preserved, so
  // copying it into a 2D canvas in a later task reads empty even when the display
  // is correct. sharp decodes the PNG; luminance variance over a blank/constant
  // canvas is ~0.
  const shot = join(OUT, `splat_page_smoke_${Date.now()}.png`);
  const el = await page.$("#splat");
  const png = await el.screenshot({ path: shot });
  const { data, info } = await sharp(png).raw().toBuffer({ resolveWithObject: true });
  const ch = info.channels;
  let n = 0, mean = 0, m2 = 0;
  for (let i = 0; i + 2 < data.length; i += ch) {
    const lum = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
    n++;
    const delta = lum - mean;
    mean += delta / n;
    m2 += delta * (lum - mean);
  }
  const variance = m2 / n;
  console.log(`SCREENSHOT ${shot}  (${info.width}x${info.height}, ${ch}ch)`);
  console.log(`canvas luminance: mean=${mean.toFixed(2)} variance=${variance.toFixed(2)} (floor ${VAR_FLOOR})`);
  if (!(variance > VAR_FLOOR)) {
    ok = false;
    fail(`canvas is blank/constant: variance ${variance.toFixed(3)} <= ${VAR_FLOOR}`);
  } else {
    console.log(`canvas NON-BLANK (variance ${variance.toFixed(2)} > ${VAR_FLOOR}) — PASS`);
  }
} catch (e) {
  ok = false;
  fail(e.message);
} finally {
  console.log(`\n--- CONSOLE (${consoleLines.length}) ---`);
  for (const l of consoleLines.slice(-25)) console.log(l);
  if (pageErrors.length) {
    console.log(`--- PAGEERRORS ---`);
    for (const l of pageErrors) console.log(l);
  }
  await browser.close();
  server.close();
}

if (ok && !process.exitCode) console.log("\nGATE PASS: page optimizes toward the prompt on the real GPU, canvas is live.");
