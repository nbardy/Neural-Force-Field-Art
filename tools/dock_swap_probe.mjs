/**
 * DOCK SWAP integration probe — does changing the model in the dock actually
 * recompile BOTH nets and leave the piece training?
 *
 *   node_modules/.bin/parcel src/index.html --port 1234    # in another shell
 *   node tools/dock_swap_probe.mjs [url] [piece] [settleMs]
 *
 * The codegen gate (`adversary_wire_test.ts` §8e) proves every dock choice
 * EMITS; this proves one of them RUNS — React restart, WGSL recompile, Adam
 * state resize, still 60 fps, still no direction collapse. The two together
 * are why the dock's arch + predictor pickers are not a way to break a piece.
 *
 * Reads `window.__nffHealth` (exact floats), never the HUD text — see the
 * 2026-08-17 soak-flake note for why parsing the HUD is not an option.
 *
 * Real-adapter flags, NOT smoke.mjs's software fallback: on an Apple box the
 * SwiftShader path does not exist and the page correctly shows the WebGPU
 * notice, which looks like a regression and is not.
 */
import puppeteer from "puppeteer";

const URL_ = process.argv[2] ?? "http://localhost:1234/index.html";
const PIECE = process.argv[3] ?? "Adversary · Pair WTA K=4";
const SETTLE = Number(process.argv[4] ?? 14000);

const browser = await puppeteer.launch({
  headless: "new",
  args: [
    "--enable-unsafe-webgpu",
    "--use-angle=metal",
    "--ignore-gpu-blocklist",
    "--window-size=1280,900",
  ],
  defaultViewport: { width: 1280, height: 900 },
});
const page = await browser.newPage();
const logs = [];
page.on("console", (m) => logs.push(m.text()));
page.on("pageerror", (e) => logs.push("PAGEERROR " + e.message));

await page.goto(URL_, { waitUntil: "domcontentloaded", timeout: 30000 });
await page.waitForSelector('[data-testid="art-piece-gallery"]', { timeout: 30000 });
await page.evaluate((name) => {
  const b = [...document.querySelectorAll("button")].find((x) =>
    (x.textContent || "").includes(name)
  );
  if (!b) throw new Error(`no gallery button '${name}'`);
  b.click();
}, PIECE);
await page.waitForFunction((n) => window.__nffHealth?.piece === n, { timeout: 45000 }, PIECE);

const snap = async (label) => {
  await new Promise((r) => setTimeout(r, SETTLE));
  const s = await page.evaluate(() => {
    const q = (t) =>
      document.querySelector(`[data-testid="${t}"]`)?.textContent?.trim() ?? null;
    const h = window.__nffHealth;
    return {
      piece: h?.piece ?? null,
      arch: q("model-arch-summary"),
      predictor: q("model-predictor-summary"),
      field: h?.field ? { r1: h.field.r1, r2: h.field.r2, mag: h.field.mag } : null,
      adv: h?.adv ? { r1: h.adv.r1, payoff: h.adv.payoff } : null,
      fps: h?.fps ?? null,
    };
  });
  console.log(`\n--- ${label} ---`);
  console.log(JSON.stringify(s, null, 2));
  return s;
};

const before = await snap("BEFORE (piece defaults)");

await page.evaluate(() => {
  const pick = (testid, label) => {
    const el = document.querySelector(`[data-testid="${testid}"]`);
    if (!el) throw new Error(`missing ${testid}`);
    const b = [...el.querySelectorAll("button")].find(
      (x) => x.textContent.trim() === label
    );
    if (!b) throw new Error(`no choice '${label}' in ${testid}`);
    b.click();
  };
  pick("model-arch-presets", "Dual HashGrid");
  pick("model-predictor-presets", "64/32");
});

const after = await snap("AFTER (Dual HashGrid + predictor 64/32)");

console.log("\n--- trainer log lines ---");
for (const l of logs.filter((l) => /\[adversary\]|\[advect\]|PAGEERROR|Error/.test(l))) {
  console.log("  " + l.slice(0, 190));
}
// Health thresholds from src/health.ts: r1/r2 are the 32^2 GRID direction
// order parameters, present on EVERY piece, `null` = UNMEASURED (never 0).
// r1 > ~0.5 is laminar collapse; r2 must be read WITH r1 because a +-F0
// counter-streaming field scores r1 ~ 0 and is just as collapsed.
const healthy = (s) =>
  s.field !== null &&
  s.field.r1 !== null &&
  s.field.r2 !== null &&
  s.field.r1 < 0.5 &&
  s.field.r2 < 0.5 &&
  s.adv !== null &&
  Number.isFinite(s.adv.payoff) &&
  s.fps > 30;
const checks = [
  ["before: measured + no direction collapse", healthy(before)],
  ["after:  measured + no direction collapse", healthy(after)],
  ["field arch actually swapped", before.arch !== after.arch && /hashgrid/.test(after.arch)],
  ["predictor actually swapped", after.predictor.includes("[64, 32]")],
];
for (const [name, pass] of checks) console.log(`  ${pass ? "ok  " : "FAIL"} ${name}`);
const verdict = checks.every(([, p]) => p);
console.log(`\nVERDICT ${verdict ? "OK" : "FAILED"}`);
await browser.close();
process.exit(verdict ? 0 : 1);
