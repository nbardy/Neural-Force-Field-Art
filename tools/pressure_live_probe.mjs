/**
 * ANTI-COLLAPSE PRESSURE — live browser probe on a REAL adapter.
 *
 *   node tools/pressure_live_probe.mjs <url> [seconds]
 *   SHOT=/tmp/a.png SCROLL_DIAG=1 node tools/pressure_live_probe.mjs <url> 60
 *
 * Complements tools/soak_adversary.mjs (which gates the release invariants) by
 * gating the things only this feature has: the compiled pressure, the R₁ chart,
 * and the √2 payoff rule. The A/B that made the fix visible is
 *
 *   <url>                              pressure ON  (the gallery default)
 *   <url>?advPolar=1e-9&advNematic=1e-9  gradient inert, R₁ still MEASURED —
 *                                        this is how to see the baseline
 *                                        collapse, because ?advPolar=0 turns
 *                                        the moments off along with the term
 *   <url>?advPolar=0&advNematic=0       fully off (R₁ reports "—")
 *
 * NOTE tools/smoke.mjs cannot be used for this on an Apple box: it forces a
 * software fallback adapter that does not exist there. These flags
 * (--use-angle=metal) are the ones soak_adversary.mjs already uses.
 *
 * Asserts, on the DEFAULT gallery piece:
 *   - the loop logs the FUSED adversary WITH an anti-collapse pressure;
 *   - the dock's R1 direction-order chart exists, shows a finite number, and
 *     MOVES (a frozen chart would mean the moments are not being read back);
 *   - the payoff chart carries the √2 reference rule under soft-angle;
 *   - fps is in the fused ballpark and there are no page/console errors.
 */
import puppeteer from "puppeteer";

const url = process.argv[2] ?? "http://localhost:8811/index.html";
const seconds = Number(process.argv[3] ?? 25);

const browser = await puppeteer.launch({
  headless: "new",
  args: [
    "--enable-unsafe-webgpu",
    "--enable-features=Vulkan,WebGPU",
    "--use-angle=metal",
    "--window-size=1280,900",
  ],
});
const page = await browser.newPage();
await page.setViewport({ width: 1280, height: 900 });
const logs = [];
const errors = [];
page.on("console", (m) => logs.push(`${m.type()}: ${m.text()}`));
page.on("pageerror", (e) => errors.push(String(e)));
page.on("requestfailed", (r) => {
  if (!/favicon/.test(r.url())) errors.push(`requestfailed ${r.url()}`);
});

await page.goto(url, { waitUntil: "networkidle2", timeout: 60000 });
await page.waitForSelector('[data-testid="fps-hud"]', { timeout: 30000 });

const samples = [];
for (let i = 0; i < seconds; i++) {
  await new Promise((r) => setTimeout(r, 1000));
  samples.push(
    await page.evaluate(() => {
      const txt = (sel) =>
        document.querySelector(sel)?.textContent?.trim() ?? null;
      const row = document.querySelector('[data-testid="direction-order-chart"]');
      const payoffRow = document.querySelector('[data-testid="disc-loss-chart"]');
      const poly = row?.querySelector(".sparkline-raw");
      return {
        hud: txt('[data-testid="fps-hud"]')?.replace(/\s+/g, " ").slice(0, 400),
        r1: row?.querySelector("strong")?.textContent ?? null,
        r1Label: row?.querySelector(".diagnostic-name")?.textContent ?? null,
        payoff: payoffRow?.querySelector("strong")?.textContent ?? null,
        rule: !!payoffRow?.querySelector(".sparkline-rule"),
        ruleTitle:
          payoffRow?.querySelector(".sparkline-rule title")?.textContent ?? null,
        points: poly?.getAttribute("points")?.length ?? 0,
        polyHash: poly?.getAttribute("points")?.slice(-40) ?? null,
      };
    })
  );
}
if (process.env.SCROLL_DIAG) {
  await page.evaluate(() => {
    document
      .querySelector('[data-testid="adversary-diagnostics"]')
      ?.scrollIntoView({ block: "center" });
  });
  await new Promise((r) => setTimeout(r, 800));
}
await page.screenshot({ path: process.env.SHOT ?? "/tmp/live_pressure_probe.png" });
await browser.close();

const advLines = logs.filter((l) => /\[adversary\]/.test(l));
const errLines = logs.filter(
  (l) => /^error:/.test(l) && !/favicon/.test(l) && !/404 \(File not found\)/.test(l)
);
const r1s = samples
  .map((s) => Number(s.r1))
  .filter((x) => Number.isFinite(x));
const moved = new Set(samples.map((s) => s.polyHash)).size > 1;
const fps = samples.map((s) => /FPS\s+([\d.]+)/.exec(s.hud ?? "")?.[1]).filter(Boolean);

console.log("--- adversary log lines ---");
advLines.forEach((l) => console.log("  " + l));
console.log("--- page errors ---");
errors.forEach((e) => console.log("  " + e));
errLines.forEach((e) => console.log("  " + e));
console.log("--- samples ---");
console.log("  R1 label   :", samples[0]?.r1Label);
console.log("  R1 values  :", samples.map((s) => s.r1).join(" "));
console.log("  payoff     :", samples.map((s) => s.payoff).join(" "));
console.log("  √2 rule    :", samples[0]?.rule, samples[0]?.ruleTitle);
console.log("  R1 chart moves:", moved);
console.log("  fps        :", fps.join(" "));
console.log("  hud tail   :", samples[samples.length - 1]?.hud);

const gates = [
  [advLines.some((l) => /FUSED/.test(l) && /pressure=anti-collapse/.test(l)),
    "loop logs FUSED adversary with anti-collapse pressure"],
  [errors.length === 0 && errLines.length === 0,
    "no page/console errors (the favicon 404 from python3 -m http.server is filtered)"],
  [r1s.length >= samples.length - 2 && r1s.every((x) => x >= 0 && x <= 1),
    `R1 reported as a finite number in [0,1] (${r1s.length}/${samples.length} samples)`],
  [moved, "R1 chart series is MOVING (moments read back every poll)"],
  [samples[0]?.rule === true, "payoff chart carries the √2 reference rule"],
  [fps.length > 0 && Number(fps[fps.length - 1]) > 40,
    `fps stays fused-fast (${fps[fps.length - 1]})`],
];
let bad = 0;
console.log("--- gates ---");
for (const [pass, msg] of gates) {
  console.log(`${pass ? "  ok  " : " FAIL "} ${msg}`);
  if (!pass) bad++;
}
console.log(bad === 0 ? "\nLIVE PRESSURE PROBE PASS" : `\n${bad} GATE(S) FAILED`);
process.exit(bad === 0 ? 0 : 1);
