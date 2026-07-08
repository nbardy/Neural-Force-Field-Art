// tools/splat/serve.mjs — tiny static server over the REPO ROOT for the
// prompt→splats page (src/splat_page.ts). Everything the page needs is then
// same-origin off ONE server:
//   /dist/splat.html                          the parcel-built page
//   /models/mobileclip_s0/plan_train.json      CLIP vision plan (90 KB)
//   /models/mobileclip_s0/weights_train.bin    CLIP vision train-weights (82 MB)
// (the text model + tokenizer + transformers.js load from the HF hub / jsdelivr
// CDN at runtime and are cached by the browser's Cache Storage API — only the
// vision weights need local serving.)
//
// CACHING (why the 82 MB weights don't re-download every reload): each response
// carries a strong ETag (mtimeMs-size) + Last-Modified + Cache-Control:
// no-cache. "no-cache" means the browser MAY store the body but MUST revalidate
// — so a reload sends a conditional request and we answer 304 Not Modified
// (no re-download) when the file is unchanged, and a full 200 when it changed
// (ETag differs), i.e. always fresh AND never re-downloads a stale-free file.
//
// Same base pattern as tools/clip/ort_web_bench.mjs. Used for manual dev and by
// the puppeteer gate (tools/splat/page_smoke.mjs imports createRepoServer).
//
//   node tools/splat/serve.mjs [port=8799]
//     → build first:  npx parcel build --no-scope-hoist --public-url ./ src/splat.html
//     → then open:     http://localhost:8799/dist/splat.html
import { createServer } from "node:http";
import { readFileSync, existsSync, statSync } from "node:fs";
import { extname, join, normalize } from "node:path";
import { fileURLToPath } from "node:url";

export const ROOT = fileURLToPath(new URL("../..", import.meta.url));

const MIME = {
  ".mjs": "text/javascript",
  ".js": "text/javascript",
  ".ts": "text/javascript",
  ".wasm": "application/wasm",
  ".onnx": "application/octet-stream",
  ".bin": "application/octet-stream",
  ".html": "text/html",
  ".css": "text/css",
  ".json": "application/json",
  ".map": "application/json",
  ".png": "image/png",
  ".svg": "image/svg+xml",
};

/** Resolve a URL path to an on-disk file under ROOT, or null if invalid/missing. */
function resolvePath(urlPath) {
  let path = normalize(join(ROOT, urlPath));
  if (!path.startsWith(ROOT)) return { forbidden: true };
  if (existsSync(path) && statSync(path).isDirectory()) path = join(path, "index.html");
  if (!existsSync(path)) return { missing: true };
  return { path };
}

/** One shared, cache-aware request handler. Conditional 304 spares the big
 *  weights blob from re-downloading on every reload (see file header). */
function handle(req, res) {
  const urlPath = decodeURIComponent(new URL(req.url, "http://x").pathname);
  if (urlPath === "/favicon.ico") return void res.writeHead(204).end();
  const r = resolvePath(urlPath);
  if (r.forbidden) return void res.writeHead(403).end();
  if (r.missing) return void res.writeHead(404).end(`not found: ${urlPath}`);

  const st = statSync(r.path);
  const etag = `"${Math.round(st.mtimeMs)}-${st.size}"`;
  const lastModified = new Date(st.mtimeMs).toUTCString();
  const cacheHeaders = {
    ETag: etag,
    "Last-Modified": lastModified,
    "Cache-Control": "no-cache",
    "Access-Control-Allow-Origin": "*",
  };

  // Conditional request → 304 if the file is unchanged (no body re-sent).
  const inm = req.headers["if-none-match"];
  const ims = req.headers["if-modified-since"];
  if ((inm && inm === etag) || (ims && Date.parse(ims) >= Math.floor(st.mtimeMs / 1000) * 1000)) {
    return void res.writeHead(304, cacheHeaders).end();
  }

  res.writeHead(200, {
    ...cacheHeaders,
    "Content-Type": MIME[extname(r.path)] ?? "application/octet-stream",
    "Content-Length": st.size,
  });
  if (req.method === "HEAD") return void res.end();
  res.end(readFileSync(r.path));
}

/** Start a static server rooted at the repo root. Resolves { server, base, port }. */
export function createRepoServer() {
  const server = createServer(handle);
  return new Promise((resolve) => {
    server.listen(0, () => {
      const port = server.address().port;
      resolve({ server, port, base: `http://localhost:${port}` });
    });
  });
}

// CLI: serve on a fixed port for manual browsing.
if (import.meta.url === `file://${process.argv[1]}`) {
  const port = Number(process.argv[2] ?? 8799);
  createServer(handle).listen(port, () => {
    console.log(`serving repo root at http://localhost:${port}`);
    console.log(`open  http://localhost:${port}/dist/splat.html`);
    if (!existsSync(join(ROOT, "dist", "splat.html"))) {
      console.warn("WARNING: dist/splat.html missing — run:");
      console.warn("  npx parcel build --no-scope-hoist --public-url ./ src/splat.html");
    }
  });
}
