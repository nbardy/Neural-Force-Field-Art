/**
 * fetch_progress — stream a fetch() body and report download progress so the
 * pages can show a loading bar for the 82 MB CLIP weights. Content-Length gives
 * the total (a CORS-safelisted header, so it's readable even from the HF
 * cross-origin fetch); the stream reader gives bytes-so-far. Falls back to a
 * plain arrayBuffer() if the body isn't streamable, and reports total=0 when
 * Content-Length is absent (caller then shows an indeterminate readout).
 *
 * Shared by src/splat_page.ts and src/splat3d_page.ts (same loader).
 */
export interface DownloadProgress {
  received: number; // bytes so far
  total: number; // bytes (0 = unknown)
  elapsedMs: number;
}

export async function fetchArrayBufferWithProgress(
  url: string,
  onProgress: (p: DownloadProgress) => void,
  init?: RequestInit
): Promise<ArrayBuffer> {
  const start = performance.now();
  const res = await fetch(url, init);
  if (!res.ok) throw new Error(`fetch ${res.status} ${url}`);
  const total = Number(res.headers.get("content-length")) || 0;
  const reader = res.body?.getReader();
  if (!reader) {
    const buf = await res.arrayBuffer();
    onProgress({ received: buf.byteLength, total: buf.byteLength || total, elapsedMs: performance.now() - start });
    return buf;
  }
  const chunks: Uint8Array[] = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.byteLength;
    onProgress({ received, total, elapsedMs: performance.now() - start });
  }
  const out = new Uint8Array(received);
  let off = 0;
  for (const c of chunks) {
    out.set(c, off);
    off += c.byteLength;
  }
  return out.buffer;
}

/** Compact text bar: "loading CLIP weights [████░░░░] 52% · 43/82 MB · 3.1s · 14 MB/s". */
export function formatProgress(label: string, p: DownloadProgress): string {
  const mb = (p.received / 1e6).toFixed(1);
  const secs = (p.elapsedMs / 1000).toFixed(1);
  const speed = p.elapsedMs > 0 ? (p.received / (p.elapsedMs / 1000) / 1e6).toFixed(1) : "0.0";
  if (p.total > 0) {
    const pct = Math.min(100, Math.round((p.received / p.total) * 100));
    const totMb = (p.total / 1e6).toFixed(0);
    const W = 16;
    const filled = Math.round((pct / 100) * W);
    const bar = "█".repeat(filled) + "░".repeat(W - filled);
    return `${label}  [${bar}] ${pct}%  ·  ${mb}/${totMb} MB  ·  ${secs}s  ·  ${speed} MB/s`;
  }
  return `${label}  ${mb} MB  ·  ${secs}s  ·  ${speed} MB/s`;
}
