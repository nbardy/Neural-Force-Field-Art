/**
 * GpuTimer — real per-pass GPU wall-clock via WebGPU timestamp-query.
 *
 * The CPU HUD numbers only ever measured ENCODE time; the GPU work runs async
 * after submit, so they hid the real cost. This times passes on the GPU: a
 * timestamp per pass begin/end, resolved into a buffer and read OFF the hot
 * path. 8 timestamps, one begin+end pair per span:
 *   0,1 trainer pass A (rollout+backward) -> rollout
 *   2,3 trainer pass B (grads+Adam)       -> optim
 *   4,5 advect pass                       -> advect
 *   6,7 render (splat: decay.begin..tonemap.end; quads: whole pass) -> render
 *
 * Reads are amortised: every PERIOD frames the query set is resolved into a
 * staging buffer inside the SAME frame encoder, then mapAsync'd on a small ring
 * so an in-flight map never stalls the frame. u64 ns in, ms out. Requires the
 * device's "timestamp-query" feature (main.ts's requestDevice shim appends it);
 * create() returns null otherwise → caller keeps the CPU-encode HUD lines.
 */

const TS_COUNT = 8;
const RING = 3; // staging buffers — enough that a map is always done before reuse
const PERIOD = 15; // resolve+read once every N frames

/**
 * Current-API pass timestamp descriptor. The installed @webgpu/types (0.1.30)
 * still declares the OLD array-of-{queryIndex,location} form, but every live
 * runtime (Chrome/Dawn) uses this object form — so we carry our own type and
 * cast at the beginComputePass/beginRenderPass boundary.
 */
export interface PassTimestampWrites {
  querySet: GPUQuerySet;
  beginningOfPassWriteIndex?: number;
  endOfPassWriteIndex?: number;
}

export interface GpuTimings {
  rollout: number;
  optim: number;
  advect: number;
  render: number;
}

export class GpuTimer {
  /** null when the device lacks "timestamp-query" (→ CPU-encode HUD fallback). */
  static create(device: GPUDevice): GpuTimer | null {
    return device.features.has("timestamp-query") ? new GpuTimer(device) : null;
  }

  private readonly querySet: GPUQuerySet;
  private readonly resolveBuf: GPUBuffer;
  private readonly ring: GPUBuffer[] = [];
  private readonly inFlight: boolean[] = [];
  private ringIdx = 0;
  private pending = false;
  /** Latest per-pass GPU times in ms, or null until the first read lands. */
  timings: GpuTimings | null = null;

  private constructor(private readonly device: GPUDevice) {
    this.querySet = device.createQuerySet({ type: "timestamp", count: TS_COUNT });
    this.resolveBuf = device.createBuffer({
      size: TS_COUNT * 8, // u64 per timestamp
      usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
    });
    for (let i = 0; i < RING; i++) {
      this.ring.push(
        device.createBuffer({
          size: TS_COUNT * 8,
          usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        })
      );
      this.inFlight.push(false);
    }
  }

  /** Pass timestampWrites spanning a begin/end index pair (single-pass spans). */
  writes(begin: number, end: number): PassTimestampWrites {
    return {
      querySet: this.querySet,
      beginningOfPassWriteIndex: begin,
      endOfPassWriteIndex: end,
    };
  }

  /**
   * Record a query-set resolve + copy into `encoder` on sample frames only.
   * Runs inside the frame's own encoder so no extra submit is created.
   */
  maybeResolve(encoder: GPUCommandEncoder, frame: number): void {
    if (frame % PERIOD !== 0) return;
    if (this.inFlight[this.ringIdx]) return; // previous map not back yet — skip
    encoder.resolveQuerySet(this.querySet, 0, TS_COUNT, this.resolveBuf, 0);
    encoder.copyBufferToBuffer(
      this.resolveBuf,
      0,
      this.ring[this.ringIdx],
      0,
      TS_COUNT * 8
    );
    this.pending = true;
  }

  /**
   * Call after queue.submit(): kicks off the async read of this frame's staging
   * buffer (OFF the hot path) and advances the ring. No-op on non-sample frames.
   */
  afterSubmit(): void {
    if (!this.pending) return;
    this.pending = false;
    const idx = this.ringIdx;
    this.ringIdx = (this.ringIdx + 1) % RING;
    const buf = this.ring[idx];
    this.inFlight[idx] = true;
    buf
      .mapAsync(GPUMapMode.READ)
      .then(() => {
        const ts = new BigUint64Array(buf.getMappedRange());
        const ms = (a: number, b: number) => Number(ts[b] - ts[a]) / 1e6;
        this.timings = {
          rollout: ms(0, 1),
          optim: ms(2, 3),
          advect: ms(4, 5),
          render: ms(6, 7),
        };
        buf.unmap();
        this.inFlight[idx] = false;
      })
      .catch(() => {
        this.inFlight[idx] = false;
      });
  }

  destroy(): void {
    try {
      this.querySet.destroy();
    } catch (_) {}
    try {
      this.resolveBuf.destroy();
    } catch (_) {}
    for (const b of this.ring) {
      try {
        b.destroy();
      } catch (_) {}
    }
  }
}
