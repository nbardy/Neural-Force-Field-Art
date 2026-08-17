/**
 * FIRST import of the entry file. Repairs a query string containing a percent
 * sign that is not a valid escape — `?dock=%%%`, `?x=100%`, a link truncated
 * mid-`%E2%82%AC` by a chat client — which otherwise BLANKS THE WHOLE PAGE.
 *
 * MEASURED MECHANISM (2026-08-17, built bundle, headless Chrome/Metal):
 * tfjs-core builds its `Environment` at MODULE SCOPE, and that constructor
 * calls `populateURLFlags()` → `getQueryParams(location.search)` →
 * `decodeURIComponent(value)`. On `?dock=%%%` that throws
 *
 *   URIError: URI malformed
 *       at getQueryParams … at populateURLFlags … at new Environment
 *       at 5PvxX../backends/backend   ← tfjs module FACTORY, i.e. import time
 *
 * so it fires while `import * as tf` is still being evaluated. Not one line of
 * app code has run: the page logs NOTHING, `#app` stays empty, no canvas is
 * created. No try/catch anywhere in the app can catch it, because the app does
 * not exist yet. The only fix is to make `location.search` decodable BEFORE
 * that import is evaluated — which is all this module does.
 *
 * NOT the same hazard as `URLSearchParams`: that one is lenient by spec and
 * hands back `%%%` verbatim without throwing (measured in both Node 24 and
 * Chrome). Every reader in this app goes through `URLSearchParams`, so the app
 * itself was never the thing that broke. tfjs's hand-rolled parser is.
 *
 * REPAIR, NOT SUPPRESSION: only the parameters that cannot decode are dropped,
 * the rest of the link keeps working, and the drop is announced on the console
 * with the exact text and reason. Silently discarding the whole query would
 * turn a mistyped share link into a piece that quietly opens on the wrong
 * artwork — the failure mode this project's "deep links stay honest" rule
 * exists to prevent.
 *
 * Dependency-free on purpose: it must not drag in anything that could itself
 * pull tfjs and lose the race. Pure half is unit-testable without a browser:
 *
 *   bun tools/url_guard_test.ts
 */

/** One `&`-separated parameter, classified by whether it can be decoded. */
type Segment =
  | { readonly tag: "decodable"; readonly raw: string }
  | { readonly tag: "undecodable"; readonly raw: string; readonly reason: string };

/** The query string with every undecodable parameter removed. */
export interface SanitizedSearch {
  /** Byte-identical to the input when `dropped` is empty. */
  readonly clean: string;
  /** One `raw — reason` line per dropped parameter; empty means untouched. */
  readonly dropped: readonly string[];
}

/**
 * `decodeURIComponent` throws `URIError` and nothing else, so the failure is
 * total information: this reports it as data rather than swallowing it.
 */
function decodes(text: string): boolean {
  try {
    decodeURIComponent(text);
    return true;
  } catch {
    return false;
  }
}

/**
 * κ — the ONE place a raw parameter becomes a classified `Segment`. Name and
 * value are tested separately because tfjs decodes them separately, and
 * because naming the offending half is what makes the console warning
 * actionable ("the value of `dock`", not "somewhere in your URL").
 */
function classifySegment(raw: string): Segment {
  const split = raw.indexOf("=");
  const name = split < 0 ? raw : raw.slice(0, split);
  const value = split < 0 ? "" : raw.slice(split + 1);
  if (decodes(name) && decodes(value)) return { tag: "decodable", raw };
  const offender = decodes(name)
    ? `value of "${name}"`
    : `parameter name "${name}"`;
  return {
    tag: "undecodable",
    raw,
    reason: `${offender} is not valid percent-encoding`,
  };
}

/**
 * Pure. `search` is `location.search` (with or without the leading "?").
 *
 * When nothing is undecodable this returns the ORIGINAL string by identity
 * rather than a rebuilt one, so a valid link — including valid `%xx` escapes,
 * base64url `?dock=` blobs, empty segments, repeated keys — can never be
 * silently normalized into a different-but-equivalent URL.
 */
export function sanitizeSearch(search: string): SanitizedSearch {
  const body = search.startsWith("?") ? search.slice(1) : search;
  const segments = body.split("&").map(classifySegment);
  const dropped = segments.filter(
    (segment): segment is Extract<Segment, { tag: "undecodable" }> =>
      segment.tag === "undecodable"
  );
  if (dropped.length === 0) return { clean: search, dropped: [] };

  const kept = segments
    .filter((segment) => segment.tag === "decodable")
    .map((segment) => segment.raw)
    .join("&");
  return {
    clean: kept === "" ? "" : `?${kept}`,
    dropped: dropped.map((segment) => `${segment.raw} — ${segment.reason}`),
  };
}

/**
 * The side effect. Runs at import; must complete before tfjs is evaluated.
 *
 * `replaceState` (not `pushState`): the malformed URL is not a place the user
 * navigated to and must not become a Back target. `history.state` is carried
 * through unchanged, and the path and hash are preserved — only the query is
 * touched, and only when something in it is genuinely undecodable.
 *
 * `replaceState` is not wrapped: for a same-origin, same-document URL it
 * cannot throw on the origins this ships to (https GitHub Pages, http
 * localhost), and a branch for a state that cannot occur would be dead code
 * pretending to be a safety net.
 */
function repairLocationSearch(): void {
  const { clean, dropped } = sanitizeSearch(window.location.search);
  if (dropped.length === 0) return;

  console.warn(
    `[url] dropped ${dropped.length} undecodable query parameter(s) — a "%" ` +
      `that is not a valid escape crashes tfjs-core's flag parser at IMPORT ` +
      `time and blanks the page, so the link is repaired before that runs:` +
      dropped.map((line) => `\n  • ${line}`).join("") +
      `\n  query is now ${clean === "" ? "(empty)" : clean}`
  );
  window.history.replaceState(
    window.history.state,
    "",
    window.location.pathname + clean + window.location.hash
  );
}

repairLocationSearch();
