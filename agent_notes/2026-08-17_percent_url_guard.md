# Percent-URL page-blanking landmine — fixed at the import boundary

Date: 2026-08-17 KST
Branch: worktree off `main` @ `c342cb7`. **Not committed.**
Predecessor: `agent_notes/2026-08-17_120215_KST_fused_audit_export_button_collapse_fix.md`
(workstream B recorded the hole; this note closes it).

## Goal

Any URL whose query string contains a `%` that is not a valid percent-escape
(`?dock=%%%`, `?x=100%`, a share link truncated mid-`%E2%82%AC` by a chat
client) blanked the entire page. Close it.

## Verified facts

### 1. The blanking is real and reproduces on the built bundle

Built `npm run build`, served `dist/` with `python3 -m http.server 8799`, drove
headless Chrome with a **real Metal adapter**
(`--enable-unsafe-webgpu --enable-webgpu-developer-features
--ignore-gpu-blocklist --use-angle=metal`).

> `tools/smoke.mjs` is the wrong tool here: it forces a software fallback
> adapter that does not exist on this box, so the app correctly shows the
> "needs WebGPU" notice and the boot path under test never runs.

Baseline, before the fix:

```
URL       http://localhost:8799/index.html?dock=%%%
BOOTED    false
PROBE     {"search":"?dock=%%%","appChildren":0,"bodyText":"","canvases":0}
PAGEERROR ["URIError: URI malformed"]
CONSOLE   (0 app lines)
```

Control (no query) on the same build: `appChildren:1`, `canvases:1`,
`[adversary] FUSED …` logged. So the query string alone is the difference.

**Zero console lines** on the failing load is the load-bearing observation: not
one line of app code ran.

### 2. Mechanism — confirmed by stack, not inferred

```
URIError: URI malformed
    at <anonymous>              index.js:977:992   ← decodeURIComponent
    at l                        index.js:977:889   ← getQueryParams
    at populateURLFlags         index.js:977:623
    at i                        index.js:974:28    ← new Environment
    at v                        index.js:921:334
    at 5PvxX../backends/backend index.js:923:95    ← tfjs MODULE FACTORY
    at u / d                    index.js:16        ← parcelRequire
    at iVT0P../engine           index.js:743:7
```

Source (`node_modules/@tensorflow/tfjs-core/dist/environment.js`): the
`Environment` constructor calls `this.populateURLFlags()`, which calls
`getQueryParams(this.global.location.search)`, whose body is

```js
queryString.replace(/[?&]([^=?&]+)(?:=([^&]*))?/g, (s, ...t) => { decodeParam(params, t[0], t[1]); … });
function decodeParam(params, name, value) {
  params[decodeURIComponent(name)] = decodeURIComponent(value || '');
}
```

and `ENV` is built at **module scope**. So the throw happens while
`import * as tf` is still being evaluated, before the app exists. No try/catch
in application code can catch it. This is why `src/index.tsx`'s `readQuery`
guard — which was already there — did not help.

### 3. CORRECTION: `URLSearchParams` does NOT throw

The comment on `readQuery` in `src/index.tsx` asserted
*"`new URLSearchParams("?%%%")` THROWS URIError"*. **Measured false** in both
Node 24 and Chrome:

| input | `new URLSearchParams(s)` | `decodeURIComponent(s)` |
|---|---|---|
| `?dock=%%%` | ok → `[["dock","%%%"]]` | **URIError** |
| `?x=100%` | ok → `[["x","100%"]]` | **URIError** |
| `?%%%` | ok → `[["%%%",""]]` | **URIError** |
| `?a=%2&b=c` | ok → `[["a","%2"],["b","c"]]` | **URIError** |

WHATWG urlencoded parsing is lenient by spec and hands malformed escapes back
verbatim. Every reader in this app goes through `URLSearchParams`, so **the app
was never the thing that broke** — tfjs's hand-rolled parser was, and it was
always the *only* thing that broke. `readQuery`'s `malformed` branch is
therefore unreachable in practice.

Left in place (it is cheap, and dock ingestion runs inside a render where a
throw unmounts the tree), but the false premise in its doc comment is now
corrected in-place so the next reader is not misled.

## Design

`src/url_guard.ts` — dependency-free, ~40 lines of logic, side-effect import,
made the FIRST import of `src/index.tsx`.

Canonical type + κ + one clean path, per root CLAUDE.md:

```ts
type Segment =
  | { tag: "decodable";   raw: string }
  | { tag: "undecodable"; raw: string; reason: string };

function classifySegment(raw: string): Segment          // κ, the only classifier
export function sanitizeSearch(search: string): { clean: string; dropped: readonly string[] }
function repairLocationSearch(): void                   // the side effect
```

Decisions worth keeping:

- **Per-parameter, not whole-query.** Split on `&`, then test name and value
  with `decodeURIComponent` separately. Only undecodable parameters are
  dropped; the rest of the link keeps working. Dropping the whole query would
  be a silent fallback that opens a mistyped share link on the wrong artwork —
  exactly what the project's "deep links stay honest" rule forbids.
- **Announced, never silent.** `console.warn` names each dropped parameter, its
  raw text, which half was bad, and the resulting query. That warn *is* the
  typed-error path.
- **Byte-identity on the happy path.** When nothing is dropped, `sanitizeSearch`
  returns the **original string object**, not a rebuilt one. A rebuilt string
  could silently normalize `?a=b&&c=` or re-encode a valid `%2F`, changing what
  every downstream `URLSearchParams` reader sees. This is asserted by the unit
  gate over 13 valid inputs.
- **`replaceState`, not `pushState`.** The malformed URL is not somewhere the
  user navigated and must not become a Back target. `history.state`, pathname
  and hash are all preserved.

### Deviations from the brief

1. **No `decodeURIComponent(location.search)` fast-path probe.** The brief
   proposed try-whole-string, then repair on `URIError`. Dropped in favour of
   always calling `sanitizeSearch` and acting on `dropped.length === 0`. The
   two-path version is only sound if "whole string decodes ⟺ every component
   decodes" — true (a multi-byte UTF-8 sequence can never validly span a
   literal `&` or `=`, both of which are ASCII and not continuation bytes), but
   it is a load-bearing argument for zero benefit: the one-path version costs
   two `decodeURIComponent` calls per parameter on a cold load and needs no
   argument at all. One clean path beat the micro-optimisation.
2. **`replaceState` is not wrapped in try/catch.** For a same-origin,
   same-document URL it cannot throw on the origins this ships to (https
   GitHub Pages, http localhost). A branch for a state that cannot occur is
   dead code pretending to be a safety net. Reasoning recorded in the module
   doc comment.
3. **Scope stayed on `index.html`.** Checked `splat.html` / `splat3d.html`:
   neither bundle contains `populateURLFlags` (verified with `rg -c` against
   the built `dist/splat*.js`), so neither entry has this landmine and neither
   needed the import.

## Gate results — all pass

### Unit — `bun tools/url_guard_test.ts` → ALL PASS (0 failures)

Stubs `window` and **records** `replaceState` calls, so "did not repair" is
asserted directly rather than inferred.

- 13 valid queries pass through **byte-identical**, incl. valid `%2F`, valid
  multi-byte `%E2%82%AC`, `+`, empty segments, repeated keys, bare keys,
  `?tf_flags=WEBGL_VERSION:2`, and a base64url `?dock=` blob.
- 9 malformed classes each dropped with a reason naming the offending half:
  `%%%` value, trailing bare `%`, truncated `%2`, non-hex `%zz`, malformed
  **key** with and without `=`, empty key + malformed value, truncated
  multi-byte `%E2%82`, lone surrogate `%ED%A0%80`.
- 6 mixed queries keep every valid parameter
  (`?dock=%%%&piece=3` → `?piece=3`; `?a=b&x=%2&y=%3&c=d` → `?a=b&c=d`).
- **Contract check:** every repaired output is re-run through *tfjs's own
  regex + `decodeURIComponent`* and through `URLSearchParams`. A fix that
  dropped only some offenders would pass the other groups and still blank.
- Idempotence: sanitizing an already-clean string is a no-op.
- Import-time side effect did not fire for the valid query it was loaded with.

### Live — `node tools/percent_url_gate.mjs` → ALL GATES PASS

Against the built `dist/`, real Metal adapter, 12 s settle. `replaceState` is
monkey-patched via `evaluateOnNewDocument` *before any page script runs*, so
repair counts are observed at the source.

| Gate | Result |
|---|---|
| 0 · bundle order | `./url_guard` required before `./main`; only the parcel helper + `react/jsx-runtime` precede it |
| 3 · plain load | boots, canvas present, **replaceState called 0 times**, URL unchanged, no `[url]` warning |
| 1 · `?dock=%%%` | **boots** (app mounted + canvas), **no pageerror**, `[url]` warning names `dock=%%%`, query repaired to empty, `[adversary] FUSED` logged |
| 2 · valid `?dock=` | boots, `[dock] adopted ?dock= share link · piece "Adversary · Pair · HashGrid · Curl"`, guard left it alone (0 replaceState, no warning), query **byte-identical** |

Gate 2's link is minted from the app's own persisted `nffa.dock.v2` blob using
the same base64url transport as `src/share.ts`, so it exercises the real
`parseDockParam`, not a fixture.

Emitted order proven in the bundle (`GATE 0`):

```
["@parcel/transformer-js/src/esmodule-helpers.js","react/jsx-runtime",
 "./url_guard","react","react-dom/client","./main","./core/gan/adversary","./share"]
```

Gate 0 keeps an allowlist of what may precede the guard; adding an import above
it fails with a message saying to move it below `"./url_guard"`.

### Build — `npm run build` clean, no new deps

`✨ Built in 3.45s`, exit 0, all three entries emitted. `package.json`
untouched.

## Open items for the next agent

1. **Lockfiles are modified in this worktree and I could not revert them.**
   `npm install` (needed — the worktree had no `node_modules`) synced
   `package-lock.json` (−34 lines) and `yarn.lock` (−17) to catch up with
   commit `8b60c53`, which dropped `onnxruntime-web` and `webgpu` from
   `package.json` but left both lockfiles stale. **Deletions only; nothing
   added.** This is pre-existing drift, not part of this fix — the guard rules
   blocked `git checkout --` on them, so they are still dirty. Either keep them
   (they are correct) or restore them, but decide deliberately before
   committing.
2. **`readQuery` in `src/index.tsx` is now provably dead defence.** Its
   `malformed` branch cannot be reached because `URLSearchParams` does not
   throw. Removing it would delete the `Query` sum type and simplify
   `parseDockParam` / `initialDock`, but that is a real refactor of dock
   ingestion with its own risk; deliberately left alone here. The misleading
   comment is fixed.
3. The guard only repairs `location.search`. `location.hash` is never parsed by
   tfjs, so it is out of scope today — but if anything starts decoding the
   hash, it needs the same treatment.

## Files

| File | Change |
|---|---|
| `src/url_guard.ts` | **new** — the guard: `sanitizeSearch` + import-time repair |
| `src/index.tsx` | `import "./url_guard"` as the first import; corrected the false `URLSearchParams`-throws claim on `readQuery` |
| `tools/url_guard_test.ts` | **new** — unit gate (`bun tools/url_guard_test.ts`) |
| `tools/percent_url_gate.mjs` | **new** — live gate incl. bundle-order tripwire |
