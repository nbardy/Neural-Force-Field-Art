/**
 * Unit gate for `sanitizeSearch` — the pure half of src/url_guard.ts.
 *
 *   bun tools/url_guard_test.ts
 *
 * Importing the module runs its side effect, which touches `window`. Bun has no
 * DOM, so a minimal stub stands in and RECORDS whether `replaceState` was
 * called — which is itself the assertion for the "valid query is untouched"
 * case, and the headless mirror of the live gate's same check.
 */

interface ReplaceCall {
  readonly state: unknown;
  readonly url: string;
}

const replaceCalls: ReplaceCall[] = [];

(globalThis as Record<string, unknown>).window = {
  // A perfectly valid query: the import-time repair must not fire.
  location: { search: "?a=b&c=d", pathname: "/index.html", hash: "" },
  history: {
    state: { marker: 1 },
    replaceState: (state: unknown, _title: string, url: string) =>
      replaceCalls.push({ state, url }),
  },
};

const { sanitizeSearch } = await import("../src/url_guard.ts");

let failures = 0;
function check(name: string, condition: boolean, detail: string): void {
  if (condition) {
    console.log(`  ok   ${name}`);
    return;
  }
  failures += 1;
  console.log(`  FAIL ${name}\n       ${detail}`);
}

// ── 1. Valid queries pass through BYTE-IDENTICAL ─────────────────────────────
// The regression this guards: a "sanitizer" that always rebuilds the string
// would silently normalize `?a=b&&c=` or re-encode a valid `%2F`, quietly
// changing what every downstream URLSearchParams reader sees.
console.log("valid queries are untouched (byte-identical):");
const untouched = [
  "",
  "?",
  "?piece=3",
  "?a=b&c=d",
  "?a=b&&c=d", // empty segment
  "?flag", // bare key, no "="
  "?a=", // empty value
  "?a=b&a=c", // repeated key
  "?path=%2Fsome%2Fwhere", // valid escapes
  "?label=%E2%82%AC", // valid multi-byte escape
  "?q=a+b", // "+" is not percent-encoding
  "?tf_flags=WEBGL_VERSION:2",
  // A real base64url dock blob: the "-"/"_" alphabet, no padding, no "%".
  "?dock=eyJ2IjoyLCJwaWVjZU5hbWUiOiJBZHYtUGFpciJ9",
];
for (const value of untouched) {
  const result = sanitizeSearch(value);
  check(
    JSON.stringify(value),
    result.clean === value && result.dropped.length === 0,
    `clean=${JSON.stringify(result.clean)} dropped=${JSON.stringify(result.dropped)}`
  );
}

// ── 2. Each malformed class is dropped, with a reason ────────────────────────
console.log("\nmalformed parameters are dropped with a reason:");
const malformed: ReadonlyArray<readonly [string, string, string]> = [
  // [input, expected clean, substring the reason must name]
  ["?dock=%%%", "", 'value of "dock"'], // the reported bug
  ["?x=100%", "", 'value of "x"'], // trailing bare "%"
  ["?a=%2", "", 'value of "a"'], // truncated escape
  ["?a=%zz", "", 'value of "a"'], // non-hex digits
  ["?%%%", "", 'parameter name "%%%"'], // malformed KEY, no "="
  ["?%%%=v", "", 'parameter name "%%%"'], // malformed KEY, with value
  ["?=%%%", "", 'value of ""'], // empty key, malformed value
  ["?a=%E2%82", "", 'value of "a"'], // truncated multi-byte sequence
  ["?a=%ED%A0%80", "", 'value of "a"'], // lone surrogate — decodes to nothing valid
];
for (const [input, expectedClean, expectedReason] of malformed) {
  const result = sanitizeSearch(input);
  const reason = result.dropped.join(" | ");
  check(
    JSON.stringify(input),
    result.clean === expectedClean &&
      result.dropped.length === 1 &&
      reason.includes(expectedReason),
    `clean=${JSON.stringify(result.clean)} dropped=${JSON.stringify(result.dropped)}`
  );
}

// ── 3. Mixed valid + invalid keeps the valid ones ────────────────────────────
// The point of the whole design: a mistyped parameter must not cost the user
// the rest of their share link.
console.log("\nmixed queries keep every valid parameter:");
const mixed: ReadonlyArray<readonly [string, string, number]> = [
  ["?dock=%%%&piece=3", "?piece=3", 1],
  ["?piece=3&dock=%%%", "?piece=3", 1],
  ["?a=b&%%%&c=d", "?a=b&c=d", 1],
  ["?a=b&x=%2&y=%3&c=d", "?a=b&c=d", 2],
  ["?good=%2F&bad=%2&also=ok", "?good=%2F&also=ok", 1],
  ["?tf_flags=WEBGL_VERSION:2&bad=%", "?tf_flags=WEBGL_VERSION:2", 1],
];
for (const [input, expectedClean, expectedDrops] of mixed) {
  const result = sanitizeSearch(input);
  check(
    JSON.stringify(input),
    result.clean === expectedClean && result.dropped.length === expectedDrops,
    `clean=${JSON.stringify(result.clean)} (want ${JSON.stringify(expectedClean)}) dropped=${JSON.stringify(result.dropped)}`
  );
}

// ── 4. The repaired output is itself decodable ───────────────────────────────
// The actual contract with tfjs: whatever we hand back must survive the exact
// call that was crashing (`decodeURIComponent` on names and values). A fix that
// merely dropped *some* offenders would pass 1–3 and still blank the page.
console.log("\nrepaired output survives tfjs's own parser:");
for (const [input] of [...malformed, ...mixed]) {
  const { clean } = sanitizeSearch(input);
  let survives = true;
  clean.replace(/[?&]([^=?&]+)(?:=([^&]*))?/g, (_s, name: string, value: string) => {
    try {
      decodeURIComponent(name);
      decodeURIComponent(value || "");
    } catch {
      survives = false;
    }
    return "";
  });
  // The app's own reader must also be happy with it.
  let uspOk = true;
  try {
    new URLSearchParams(clean);
  } catch {
    uspOk = false;
  }
  check(JSON.stringify(input), survives && uspOk, `clean=${JSON.stringify(clean)}`);
}

// ── 5. Idempotence: sanitizing a clean string is a no-op ─────────────────────
console.log("\nsanitize is idempotent:");
for (const [input] of [...malformed, ...mixed]) {
  const once = sanitizeSearch(input).clean;
  const twice = sanitizeSearch(once);
  check(
    JSON.stringify(input),
    twice.clean === once && twice.dropped.length === 0,
    `once=${JSON.stringify(once)} twice=${JSON.stringify(twice.clean)}`
  );
}

// ── 6. The side effect did NOT fire for the valid query it was loaded with ───
console.log("\nimport-time side effect on a VALID query:");
check(
  "replaceState was never called",
  replaceCalls.length === 0,
  `calls=${JSON.stringify(replaceCalls)}`
);

console.log(failures === 0 ? "\nALL PASS" : `\n${failures} FAILURE(S)`);
process.exit(failures === 0 ? 0 : 1);
