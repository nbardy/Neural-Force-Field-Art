/**
 * Live gate for src/url_guard.ts, against the BUILT bundle in dist/.
 *
 *   node tools/percent_url_gate.mjs [baseUrl]
 *
 * Not tools/smoke.mjs: that one forces a software fallback adapter that does
 * not exist on this box, so the app shows the "needs WebGPU" notice and the
 * boot path under test never runs. This uses the real Metal adapter.
 *
 * Three assertions, matching the three ways this can regress:
 *   1. `?dock=%%%` BOOTS (was: URIError at tfjs import time, blank page).
 *   2. A VALID `?dock=` share link still ingests — minted from the app's own
 *      persisted dock, so it exercises the real parseDockParam, not a fixture.
 *   3. A plain no-query load is UNTOUCHED — replaceState is monkey-patched
 *      before any page script runs, so "we did not repair" is asserted
 *      directly rather than inferred from the URL.
 */
import { readFileSync, readdirSync } from "node:fs";
import puppeteer from "puppeteer";

const base = process.argv[2] ?? "http://localhost:8799/index.html";
const DIST = new URL("../dist/", import.meta.url).pathname;
const SETTLE_MS = 12000;

let failures = 0;
function check(name, condition, detail) {
  if (!condition) failures += 1;
  console.log(`${condition ? "ok  " : "FAIL"}  ${name}`);
  if (!condition) console.log(`        ${detail}`);
}

// ── Gate 0: the guard is required BEFORE the module that pulls tfjs ──────────
// Static tripwire. The live gates below would also catch a reorder, but only
// by blanking; this says *why*. Parcel preserves import order in the emitted
// factory, so the entry's require sequence is the evaluation order.
{
  console.log("=".repeat(72));
  console.log("GATE 0 — bundle require order");
  const bundle = readdirSync(DIST)
    .filter((f) => /^index\.[0-9a-f]+\.js$/.test(f))
    .map((f) => ({ f, src: readFileSync(DIST + f, "utf8") }))
    .find((b) => b.src.includes("undecodable query parameter"));
  if (!bundle) {
    check("guard is present in a built bundle", false, `no match in ${DIST}`);
  } else {
    const factory = bundle.src.slice(
      bundle.src.lastIndexOf(":[function(", bundle.src.indexOf('"./url_guard":"')),
      bundle.src.indexOf('"./url_guard":"')
    );
    const requires = [...factory.matchAll(/e\("([^"]{1,60})"\)/g)].map((m) => m[1]);
    const guardAt = requires.indexOf("./url_guard");
    const mainAt = requires.indexOf("./main");
    // Anything evaluated before the guard must be incapable of reaching tfjs.
    const SAFE_BEFORE = new Set([
      "@parcel/transformer-js/src/esmodule-helpers.js",
      "react/jsx-runtime",
    ]);
    const before = requires.slice(0, guardAt);
    console.log(`  bundle ${bundle.f}`);
    console.log(`  entry requires: ${JSON.stringify(requires.slice(0, 8))}`);
    check(
      "./url_guard is required before ./main (which imports tfjs)",
      guardAt >= 0 && mainAt >= 0 && guardAt < mainAt,
      `url_guard@${guardAt} main@${mainAt}`
    );
    check(
      "nothing that could reach tfjs is evaluated before the guard",
      before.every((id) => SAFE_BEFORE.has(id)),
      `before guard: ${JSON.stringify(before.filter((id) => !SAFE_BEFORE.has(id)))}` +
        ` — if this is genuinely tfjs-free, add it to SAFE_BEFORE; otherwise move` +
        ` the import below "./url_guard" in src/index.tsx`
    );
  }
}

const browser = await puppeteer.launch({
  headless: "new",
  args: [
    "--no-sandbox",
    "--enable-unsafe-webgpu",
    "--enable-webgpu-developer-features",
    "--ignore-gpu-blocklist",
    "--use-angle=metal",
  ],
});

/** Load `base + query` and report everything the gates need to judge it. */
async function load(query) {
  const page = await browser.newPage();
  const logs = [];
  const errors = [];
  page.on("console", (m) => logs.push(`${m.type()}: ${m.text()}`));
  page.on("pageerror", (e) => errors.push(`${e.name}: ${e.message}`));
  // Count repairs at the source, before the guard (or anything else) can run.
  await page.evaluateOnNewDocument(() => {
    window.__replaceStateCalls = [];
    const real = History.prototype.replaceState;
    History.prototype.replaceState = function (state, title, url) {
      window.__replaceStateCalls.push(String(url));
      return real.call(this, state, title, url);
    };
  });
  await page.goto(base + query, { waitUntil: "domcontentloaded", timeout: 60000 });
  await new Promise((r) => setTimeout(r, SETTLE_MS));
  const probe = await page.evaluate(() => ({
    href: window.location.href,
    search: window.location.search,
    replaceStateCalls: window.__replaceStateCalls,
    appChildren: document.getElementById("app")?.childElementCount ?? -1,
    canvases: document.querySelectorAll("canvas").length,
    dockBlob: window.localStorage.getItem("nffa.dock.v2"),
  }));
  return { page, logs, errors, probe, url: base + query };
}

function report(label, run) {
  console.log("=".repeat(72));
  console.log(`${label}\n  ${run.url}`);
  console.log(`  search=${JSON.stringify(run.probe.search)}`);
  console.log(`  replaceState=${JSON.stringify(run.probe.replaceStateCalls)}`);
  console.log(`  appChildren=${run.probe.appChildren} canvases=${run.probe.canvases}`);
  console.log(`  pageerrors=${JSON.stringify(run.errors)}`);
  for (const line of run.logs.filter((l) => /\[url\]|\[dock\]|\[adversary\]/.test(l))) {
    console.log(`  ${line}`);
  }
}

// ── Gate 3 first: a plain load also mints the dock blob the share link needs ──
const plain = await load("");
report("GATE 3 — plain load, no query", plain);
check(
  "plain load boots",
  plain.probe.appChildren > 0 && plain.probe.canvases > 0,
  JSON.stringify(plain.probe)
);
check(
  "plain load calls replaceState ZERO times",
  plain.probe.replaceStateCalls.length === 0,
  JSON.stringify(plain.probe.replaceStateCalls)
);
check(
  "plain load URL unchanged",
  plain.probe.href === base && plain.probe.search === "",
  `href=${plain.probe.href}`
);
check(
  "plain load emits no [url] warning",
  !plain.logs.some((l) => l.includes("[url]")),
  plain.logs.filter((l) => l.includes("[url]")).join(" | ")
);

const dockBlob = plain.probe.dockBlob;
await plain.page.close();

// ── Gate 1: the reported bug ─────────────────────────────────────────────────
const broken = await load("?dock=%%%");
report("GATE 1 — malformed query ?dock=%%%", broken);
check(
  "?dock=%%% BOOTS (app mounted + canvas)",
  broken.probe.appChildren > 0 && broken.probe.canvases > 0,
  JSON.stringify(broken.probe)
);
check(
  "no URIError pageerror",
  broken.errors.length === 0,
  JSON.stringify(broken.errors)
);
check(
  "[url] warning names the dropped parameter",
  broken.logs.some((l) => l.includes("[url]") && l.includes("dock=%%%")),
  broken.logs.filter((l) => l.includes("[url]")).join(" | ")
);
check(
  "URL repaired to an empty query",
  broken.probe.search === "" && broken.probe.replaceStateCalls.length === 1,
  `search=${JSON.stringify(broken.probe.search)} calls=${JSON.stringify(broken.probe.replaceStateCalls)}`
);
check(
  "default piece still runs FUSED",
  broken.logs.some((l) => l.includes("[adversary] FUSED")),
  broken.logs.filter((l) => l.includes("[adversary]")).join(" | ")
);
await broken.page.close();

// ── Gate 2: a VALID share link still ingests ─────────────────────────────────
if (!dockBlob) {
  failures += 1;
  console.log("FAIL  could not mint a share link: no nffa.dock.v2 in localStorage");
} else {
  // Same transport as src/share.ts encodeDockParam: UTF-8 → base64 → base64url.
  const json = JSON.stringify(JSON.parse(dockBlob));
  const bytes = new TextEncoder().encode(json);
  let binary = "";
  for (const byte of bytes) binary += String.fromCharCode(byte);
  const param = Buffer.from(binary, "binary")
    .toString("base64")
    .replace(/\+/g, "-")
    .replace(/\//g, "_")
    .replace(/=+$/, "");

  const shared = await load(`?dock=${param}`);
  report("GATE 2 — valid ?dock= share link", shared);
  check(
    "valid share link boots",
    shared.probe.appChildren > 0 && shared.probe.canvases > 0,
    JSON.stringify(shared.probe)
  );
  check(
    "share link is ADOPTED",
    shared.logs.some((l) => l.includes("[dock] adopted ?dock= share link")),
    shared.logs.filter((l) => l.includes("[dock]")).join(" | ")
  );
  check(
    "guard left the valid link alone (no replaceState, no [url] warning)",
    shared.probe.replaceStateCalls.length === 0 &&
      !shared.logs.some((l) => l.includes("[url]")),
    `calls=${JSON.stringify(shared.probe.replaceStateCalls)}`
  );
  check(
    "query survived byte-identical",
    shared.probe.search === `?dock=${param}`,
    `search=${shared.probe.search}`
  );
  await shared.page.close();
}

await browser.close();
console.log("=".repeat(72));
console.log(failures === 0 ? "ALL GATES PASS" : `${failures} GATE CHECK(S) FAILED`);
process.exit(failures === 0 ? 0 : 1);
