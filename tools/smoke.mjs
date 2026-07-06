#!/usr/bin/env node
/**
 * smoke.mjs — reusable WebGPU smoke harness.
 *
 * Drives a URL in headless Chrome WITH WebGPU enabled (SwiftShader software
 * adapter, so it works on CI / headless boxes with no GPU), captures all console
 * output + page errors + failed requests, probes `navigator.gpu` / the tfjs
 * backend / the on-screen HUD, and writes a screenshot. Prints the screenshot
 * PATH and the console log so an agent can "see" a WebGPU page it otherwise
 * can't render.
 *
 * Usage:
 *   node tools/smoke.mjs [url] [waitMs] [outDir]
 *   node tools/smoke.mjs http://localhost:8080 8000 /tmp
 *   node tools/smoke.mjs https://nbardy.github.io/Neural-Force-Field-Art/
 *
 * Exit code 0 always (it's a probe, not a test); read PROBE/CONSOLE to judge.
 */
import puppeteer from "puppeteer";
import path from "node:path";

const url =
  process.argv[2] || "https://nbardy.github.io/Neural-Force-Field-Art/";
const waitMs = parseInt(process.argv[3] || "7000", 10);
const outDir = process.argv[4] || "/tmp";

const browser = await puppeteer.launch({
  headless: true,
  args: [
    "--no-sandbox",
    "--use-angle=swiftshader", // software GL/Vulkan
    "--enable-unsafe-swiftshader", // software WebGPU when no hardware adapter
    "--enable-features=Vulkan",
    "--ignore-gpu-blocklist",
    "--enable-webgpu-developer-features",
  ],
});

const page = await browser.newPage();
await page.setViewport({ width: 1280, height: 720 });

// Headless Chrome has no hardware WebGPU adapter; the default requestAdapter()
// returns null. Force the SwiftShader software fallback so the page's WebGPU
// path actually runs here (real browsers keep using their real adapter). This
// is the only reason a WebGPU app is verifiable in a headless sandbox.
await page.evaluateOnNewDocument(() => {
  if (navigator.gpu && navigator.gpu.requestAdapter) {
    const orig = navigator.gpu.requestAdapter.bind(navigator.gpu);
    navigator.gpu.requestAdapter = (opts = {}) =>
      orig({ ...opts, forceFallbackAdapter: true });
  }
});

const logs = [];
page.on("console", (m) => logs.push(`[${m.type()}] ${m.text()}`));
page.on("pageerror", (e) => logs.push(`[pageerror] ${e.message}`));
page.on("requestfailed", (r) =>
  logs.push(`[reqfail] ${r.url()} ${r.failure()?.errorText}`)
);

try {
  await page.goto(url, { waitUntil: "networkidle2", timeout: 45000 });
} catch (e) {
  logs.push(`[goto] ${e.message}`);
}
await new Promise((r) => setTimeout(r, waitMs));

const probe = await page.evaluate(async () => {
  const out = {
    webgpu: !!navigator.gpu,
    adapter: null,
    adapterErr: null,
    hud: null,
    warning: false,
  };
  try {
    if (navigator.gpu) {
      const a = await navigator.gpu.requestAdapter();
      out.adapter = a ? "yes" : "null";
    }
  } catch (e) {
    out.adapterErr = String(e);
  }
  const divs = [...document.querySelectorAll("div")];
  const hud = divs.find((d) => d.textContent && d.textContent.includes("backend"));
  out.hud = hud ? hud.textContent.replace(/\n/g, " | ") : null;
  out.warning = divs.some(
    (d) => d.textContent && d.textContent.includes("This needs WebGPU")
  );
  return out;
});

const shot = path.join(outDir, `smoke_${Date.now()}.png`);
await page.screenshot({ path: shot });

console.log("URL        " + url);
console.log("SCREENSHOT " + shot);
console.log("PROBE      " + JSON.stringify(probe));
console.log(`--- CONSOLE (${logs.length}) ---`);
console.log(logs.join("\n"));

await browser.close();
