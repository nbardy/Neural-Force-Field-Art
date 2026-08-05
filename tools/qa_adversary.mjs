import puppeteer from "puppeteer";
import path from "node:path";
const base = process.argv[2] || "http://localhost:8799/index.html";
const outDir = process.argv[3] || "/private/tmp/claude-501/-Users-nicholasbardy-git/92bb4ee1-c21e-4d63-b291-4b07196d2780/scratchpad";
const browser = await puppeteer.launch({
  headless: "new",
  args: [
    "--enable-unsafe-webgpu",
    "--enable-features=Vulkan,WebGPU",
    "--use-angle=metal",
    "--window-size=1280,800",
  ],
});
const page = await browser.newPage();
await page.setViewport({ width: 1280, height: 800 });
const consoleLines = [];
page.on("console", (m) => consoleLines.push(`[${m.type()}] ${m.text()}`));
page.on("pageerror", (e) => consoleLines.push(`[pageerror] ${e.message}`));
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function probe(label) {
  const p = await page.evaluate(async () => {
    const texts = [...document.querySelectorAll("div")].map((d) => d.textContent || "");
    const hud = texts.find((t) => t.includes("FPS") && t.includes("backend"));
    const panel = texts.find((t) => t.includes("surprise") && t.includes("heads"));
    return {
      hud: hud ? hud.slice(0, 500) : null,
      panel: panel ? panel.slice(0, 300) : null,
      warning: !!document.body.textContent.match(/needs WebGPU/i),
    };
  });
  const shot = path.join(outDir, `adv_${label}.png`);
  await page.screenshot({ path: shot });
  console.log(`\n=== ${label} ===\nSCREENSHOT ${shot}\nHUD ${JSON.stringify(p.hud)}\nPANEL ${JSON.stringify(p.panel)}\nWARN ${p.warning}`);
  return p;
}
async function clickPiece(name) {
  await page.evaluate((n) => {
    const btn = [...document.querySelectorAll("button")].find((b) => (b.textContent || "").includes(n));
    if (!btn) throw new Error(`no button containing '${n}'`);
    btn.click();
  }, name);
}
await page.goto(base, { waitUntil: "domcontentloaded" });
await sleep(9000);
await probe("boot");
for (const n of ["Adversary · Single (control)", "Adversary · WTA K=8", "Adversary · Pair WTA K=4", "Adversary · Tri WTA K=6", "Adversary · Chaos Weave"]) {
  await clickPiece(n);
  await sleep(14000);
  await probe(n.replace(/[^a-zA-Z0-9]+/g, "_"));
}
console.log(`\n--- CONSOLE (${consoleLines.length}) ---`);
for (const l of consoleLines.slice(0, 60)) console.log(l);
await browser.close();
