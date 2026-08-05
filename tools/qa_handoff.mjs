// A/B the `?handoff=N` dial (tfjs WEBGPU_CPU_HANDOFF_SIZE_THRESHOLD) on the
// ADVERSARY pieces' learn cost, read from the live HUD in headless Chrome.
//
//   node tools/qa_handoff.mjs [baseUrl] [outDir]
//
// Why this exists: adversary pieces train on the tfjs webgpu backend with
// B=256 tensors — tiny-op, dispatch-bound work. tfjs can forward ops whose
// tensors are below the handoff threshold to the CPU backend; whether that
// helps the REAL frame (not a synthetic bench) is only answerable from the
// HUD's learn line on real hardware. Companion to tools/handoff_bench.ts
// (bun-webgpu, aesthetic loss only, no adversary).
import puppeteer from "puppeteer";

const base = process.argv[2] || "http://localhost:1234/index.html";
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const CONFIGS = [
  { label: "default (threshold 1000)", qs: "" },
  { label: "handoff=0   (all ops GPU)", qs: "?handoff=0" },
  { label: "handoff=1e6 (tiny ops CPU)", qs: "?handoff=1000000" },
];
const PIECES = ["Adversary · Pair WTA K=4", "Adversary · Tri WTA K=6"];

const browser = await puppeteer.launch({
  headless: "new",
  args: [
    "--enable-unsafe-webgpu",
    "--enable-features=Vulkan,WebGPU",
    "--use-angle=metal",
    "--window-size=1280,800",
  ],
});

function median(xs) {
  const s = [...xs].sort((a, b) => a - b);
  return s.length ? s[Math.floor(s.length / 2)] : NaN;
}

async function hudSample(page) {
  return page.evaluate(() => {
    const texts = [...document.querySelectorAll("div")].map((d) => d.textContent || "");
    const hud = texts.find((t) => t.includes("FPS") && t.includes("backend"));
    if (!hud) return null;
    const learn = hud.match(/learn\s+([\d.]+)\s*ms/);
    const fps = hud.match(/FPS\s+([\d.]+)/);
    return {
      learn: learn ? parseFloat(learn[1]) : NaN,
      fps: fps ? parseFloat(fps[1]) : NaN,
      raw: hud.slice(0, 260),
    };
  });
}

const results = [];
for (const cfg of CONFIGS) {
  const page = await browser.newPage();
  await page.setViewport({ width: 1280, height: 800 });
  const consoleLines = [];
  page.on("console", (m) => consoleLines.push(m.text()));
  await page.goto(base + cfg.qs, { waitUntil: "domcontentloaded" });
  await sleep(9000);
  for (const piece of PIECES) {
    await page.evaluate((n) => {
      const btn = [...document.querySelectorAll("button")].find((b) =>
        (b.textContent || "").includes(n)
      );
      if (!btn) throw new Error(`no button containing '${n}'`);
      btn.click();
    }, piece);
    await sleep(12000); // warm-up: backend settle + EMA fill
    const learns = [];
    const fpss = [];
    let raw = "";
    for (let i = 0; i < 8; i++) {
      const s = await hudSample(page);
      if (s && Number.isFinite(s.learn)) {
        learns.push(s.learn);
        fpss.push(s.fps);
        raw = s.raw;
      }
      await sleep(1000);
    }
    results.push({
      config: cfg.label,
      piece,
      learnMs: median(learns),
      fps: median(fpss),
      n: learns.length,
      raw,
    });
    console.log(
      `${cfg.label} | ${piece} | learn ${median(learns).toFixed(1)} ms ` +
        `(median of ${learns.length}) | FPS ${median(fpss).toFixed(1)}`
    );
  }
  const handoffLine = consoleLines.find((l) => l.includes("CPU handoff threshold"));
  if (handoffLine) console.log(`  [console] ${handoffLine}`);
  await page.close();
}

console.log("\n=== SUMMARY (median HUD values) ===");
for (const r of results) {
  console.log(
    `${r.config.padEnd(28)} ${r.piece.padEnd(26)} learn ${String(r.learnMs).padStart(6)} ms   FPS ${r.fps}`
  );
}
console.log("\nlast HUD sample per row available in `raw` — first row:", results[0]?.raw);
await browser.close();
