// tools/clip/ort_web_bench.mjs — ORT-WebGPU baseline for the fused-WGSL comparison.
//
// Run:  node tools/clip/ort_web_bench.mjs [runs=50]
//
// Measures what a STANDARD runtime does with the same model on the same GPU:
// onnxruntime-web's WebGPU EP in real-Metal headless Chrome (same launch
// flags as tools/qa_browser.mjs). WebGPU EP only — no wasm fallback listed,
// so if it runs, it ran on the GPU. Correctness is checked against the same
// golden embedding fixture the fused suite uses (cosine), so the timing is
// for a verified-correct forward, not a silently-broken one.
import { createServer } from "node:http";
import { readFileSync, existsSync } from "node:fs";
import { extname, join, normalize } from "node:path";
import { fileURLToPath } from "node:url";
import puppeteer from "puppeteer";

const ROOT = fileURLToPath(new URL("../..", import.meta.url));
const RUNS = Number(process.argv[2] ?? 50);
const MIME = {
  ".mjs": "text/javascript", ".js": "text/javascript", ".wasm": "application/wasm",
  ".onnx": "application/octet-stream", ".bin": "application/octet-stream",
  ".html": "text/html", ".json": "application/json",
};

const server = createServer((req, res) => {
  const path = normalize(join(ROOT, decodeURIComponent(new URL(req.url, "http://x").pathname)));
  if (!path.startsWith(ROOT) || !existsSync(path)) {
    res.writeHead(404).end();
    return;
  }
  res.writeHead(200, { "Content-Type": MIME[extname(path)] ?? "application/octet-stream" });
  res.end(readFileSync(path));
});
await new Promise((r) => server.listen(0, r));
const base = `http://localhost:${server.address().port}`;

const browser = await puppeteer.launch({
  headless: "new",
  args: [
    "--no-sandbox",
    "--use-angle=metal",
    "--ignore-gpu-blocklist",
    "--enable-webgpu-developer-features",
  ],
});
const page = await browser.newPage();
page.on("console", (m) => process.env.VERBOSE && console.log(`[page:${m.type()}] ${m.text()}`));
page.on("pageerror", (e) => console.error(`[pageerror] ${e.message}`));
await page.goto(`${base}/tools/clip/ort_web_bench.html`, { waitUntil: "domcontentloaded" });

const result = await page.evaluate(
  async (base, runs) => {
    const ort = await import(`${base}/node_modules/onnxruntime-web/dist/ort.all.min.mjs`);
    ort.env.wasm.wasmPaths = `${base}/node_modules/onnxruntime-web/dist/`;
    const adapter = await navigator.gpu.requestAdapter();
    const adapterInfo = `${adapter?.info?.vendor ?? "?"} ${adapter?.info?.architecture ?? "?"}`;

    const t0 = performance.now();
    const session = await ort.InferenceSession.create(
      `${base}/models/mobileclip_s0/vision_model.onnx`,
      { executionProviders: ["webgpu"], graphOptimizationLevel: "all" }
    );
    const createMs = performance.now() - t0;

    const buf = await (await fetch(`${base}/models/mobileclip_s0/fixtures/input_1x3x256x256.f32.bin`)).arrayBuffer();
    const feeds = {
      pixel_values: new ort.Tensor("float32", new Float32Array(buf), [1, 3, 256, 256]),
    };

    for (let i = 0; i < 10; i++) await session.run(feeds); // warm (shader JIT)
    const times = [];
    let out;
    for (let i = 0; i < runs; i++) {
      const t = performance.now();
      out = await session.run(feeds);
      times.push(performance.now() - t);
    }
    times.sort((a, b) => a - b);
    return {
      adapterInfo,
      createMs,
      version: ort.env.versions?.web ?? "?",
      mean: times.reduce((a, b) => a + b, 0) / times.length,
      median: times[Math.floor(times.length / 2)],
      min: times[0],
      embedsHead: [...out.image_embeds.data.slice(0, 8)],
      embeds: [...out.image_embeds.data],
    };
  },
  base,
  RUNS
);

// Node pools small Buffers (<4KB): .buffer is the shared 8KB pool, NOT this
// file's bytes — Float32Array(buf.buffer) silently reads pool garbage. Always
// slice via byteOffset/length. (Bun's readFileSync is exact-sized; the bun
// test suite can use the short form, Node tools cannot.)
const goldenBuf = readFileSync(join(ROOT, "models/mobileclip_s0/fixtures/image_embeds_512.f32.bin"));
const golden = new Float32Array(goldenBuf.buffer, goldenBuf.byteOffset, goldenBuf.length / 4);
let dot = 0, na = 0, nb = 0;
for (let i = 0; i < golden.length; i++) {
  dot += result.embeds[i] * golden[i];
  na += result.embeds[i] ** 2;
  nb += golden[i] ** 2;
}

console.log(`ort-web ${result.version} · webgpu EP · adapter: ${result.adapterInfo}`);
console.log(`session create (incl. graph opt): ${result.createMs.toFixed(0)} ms`);
console.log(`embedding cosine vs ORT-CPU golden: ${(dot / Math.sqrt(na * nb)).toFixed(6)}`);
console.log(`  first 8: ${result.embedsHead.map((x) => x.toFixed(5)).join(", ")}`);
console.log(
  `ORT WebGPU (${RUNS} runs): mean ${result.mean.toFixed(2)} ms · ` +
    `median ${result.median.toFixed(2)} ms · min ${result.min.toFixed(2)} ms`
);

await browser.close();
server.close();
