/**
 * Real-GPU browser QA driver. Unlike tools/smoke.mjs (which forces the
 * SwiftShader software adapter so CI-ish boxes can at least boot the page),
 * this launches Chrome with the REAL GPU enabled (new headless supports
 * hardware Metal on macOS), clicks through gallery pieces, probes the HUD +
 * console, and screenshots each state so the driver (human or agent) can
 * eyeball the render.
 *
 *   node tools/qa_browser.mjs [baseUrl] [outDir]
 *   HEADED=1 node tools/qa_browser.mjs        # visible window fallback
 */
import puppeteer from "puppeteer";
import path from "node:path";

const base = process.argv[2] || "http://localhost:8798/index.html";
const outDir = process.argv[3] || "/tmp";
const headed = !!process.env.HEADED;

const browser = await puppeteer.launch({
  headless: headed ? false : "new",
  args: [
    "--no-sandbox",
    "--use-angle=metal",
    "--ignore-gpu-blocklist",
    "--enable-webgpu-developer-features",
    "--window-size=1280,800",
  ],
});

const page = await browser.newPage();
await page.setViewport({ width: 1280, height: 720 });
const consoleLines = [];
page.on("console", (m) => consoleLines.push(`[${m.type()}] ${m.text()}`));
page.on("pageerror", (e) => consoleLines.push(`[pageerror] ${e.message}`));

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function probe(label) {
  const p = await page.evaluate(async () => {
    const a = navigator.gpu ? await navigator.gpu.requestAdapter() : null;
    const hud = [...document.querySelectorAll("div")]
      .map((d) => d.textContent || "")
      .find((t) => t.includes("FPS") && t.includes("backend"));
    return {
      webgpu: !!navigator.gpu,
      adapter: a ? `${a.info?.vendor ?? "?"} ${a.info?.architecture ?? "?"}` : null,
      hud: hud ? hud.slice(0, 300) : null,
      warning: !!document.body.textContent.match(/needs WebGPU/i),
    };
  });
  const shot = path.join(outDir, `qa_${label}_${Date.now()}.png`);
  await page.screenshot({ path: shot });
  console.log(`\n=== ${label} ===`);
  console.log(`SCREENSHOT ${shot}`);
  console.log(`PROBE ${JSON.stringify(p)}`);
  return p;
}

async function clickPiece(name) {
  await page.evaluate((n) => {
    const btn = [...document.querySelectorAll("button")].find((b) =>
      (b.textContent || "").includes(n)
    );
    if (!btn) throw new Error(`no button containing '${n}'`);
    btn.click();
  }, name);
}

// ---- pass 1: default load (piece 0, legacy MLP on the fused advect path) ----
const runUrl = process.argv[4] || base;
await page.goto(runUrl, { waitUntil: "domcontentloaded" });
await sleep(12000);
await probe("piece0_spiral");

// ---- Helmholtz · Chaos (fused trainer, particle batches) ----
await clickPiece("Helmholtz · Chaos");
await sleep(12000);
await probe("helmholtz_chaos");

// ---- Helmholtz · Species (classes=3) ----
await clickPiece("Helmholtz · Species");
await sleep(12000);
await probe("helmholtz_species");

// ---- the selectable model types — all FUSED-trained since M4 (train_wgsl
// generates each backward; verified vs tfjs fixtures on Metal, see
// tools/train_types_test.ts). QA: they must render, move, and show the
// fused rollout/optim HUD lines without console errors. ----
await clickPiece("Helmholtz · SIREN");
await sleep(12000);
await probe("helmholtz_siren");

await clickPiece("Helmholtz · Fourier");
await sleep(12000);
await probe("helmholtz_fourier");

await clickPiece("Helmholtz · HashGrid");
await sleep(12000);
await probe("helmholtz_hashgrid");

console.log(`\n--- CONSOLE (${consoleLines.length}) ---`);
for (const l of consoleLines.slice(0, 40)) console.log(l);

await browser.close();
