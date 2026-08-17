/**
 * Share-link codec for the dock blob — the pure, DOM-free half of the
 * export/share path. `src/index.tsx` owns the v2 schema and its validator;
 * this module owns only the two things that have to be exactly reversible:
 * the base64url transport, and the piece-identity rule.
 *
 * It deliberately imports NOTHING (no React, no tfjs, no GALLERY), so the
 * round-trip can be exercised directly:
 *
 *   bun tools/share_roundtrip_test.ts     (or any scratch script)
 *
 * — which is the only way a "copy link" feature gets a falsifiable test at
 * all: everything downstream of here needs a GPU.
 */

/** `?dock=` decoded back into JSON, or the reason it could not be. */
export type DecodedParam =
  | { readonly tag: "ok"; readonly json: unknown }
  | { readonly tag: "invalid"; readonly reason: string };

/**
 * JSON → base64url. `-`/`_` for `+`/`/` and no `=` padding, so the parameter
 * survives a URL, a chat client and a copy-paste without percent-encoding
 * noise (a plain base64 `+` becomes a SPACE when the link is re-parsed as a
 * query string — that is the bug this alphabet exists to avoid).
 *
 * UTF-8 first, via TextEncoder: `btoa` throws on any code point > 255, and
 * every gallery piece name contains "·" (U+00B7). Latin-1 `btoa` would happen
 * to accept that one and then hand back mojibake on decode.
 */
export function encodeDockParam(value: unknown): string {
  const bytes = new TextEncoder().encode(JSON.stringify(value));
  // Byte-at-a-time: String.fromCharCode(...bytes) blows the argument limit on
  // a spread of a few hundred KB, and a shared dock is unbounded in principle.
  let binary = "";
  for (const byte of bytes) binary += String.fromCharCode(byte);
  return btoa(binary).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

/** base64url → JSON. Typed failure, never a silent empty object. */
export function decodeDockParam(param: string): DecodedParam {
  const base64 = param.replace(/-/g, "+").replace(/_/g, "/");
  // atob requires a multiple-of-4 length; base64url dropped the padding.
  const padded = base64 + "=".repeat((4 - (base64.length % 4)) % 4);
  try {
    const binary = atob(padded);
    const bytes = Uint8Array.from(binary, (ch) => ch.charCodeAt(0));
    // fatal:true so malformed UTF-8 REJECTS instead of decoding to U+FFFD and
    // then failing later as a mystery JSON error.
    const text = new TextDecoder("utf-8", { fatal: true }).decode(bytes);
    return { tag: "ok", json: JSON.parse(text) as unknown };
  } catch (error) {
    return {
      tag: "invalid",
      reason: `not base64url-encoded JSON (${
        error instanceof Error ? error.message : String(error)
      })`,
    };
  }
}

/**
 * How a shared blob's gallery piece was decided.
 *
 * GALLERY is append-only (see the banner above it in src/main.ts) so an index
 * is *usually* stable — but only "usually", and the whole point of a shared
 * link is that it outlives the build that made it. The NAME is the identity
 * convention this codebase already commits to (DEFAULT_PIECE_NAME resolves the
 * default piece by name so a rename fails loudly), so a name that disagrees
 * with the index wins, and says so.
 */
export type SharedPiece =
  /** No name carried, or name and index agree. */
  | { readonly tag: "index"; readonly piece: number }
  /** Name and index disagree — the index is stale, the name won. */
  | {
      readonly tag: "renamed";
      readonly piece: number;
      readonly staleIndex: number;
      readonly name: string;
    }
  /** A name this build has never heard of; only the index is left to try. */
  | { readonly tag: "unknown-name"; readonly piece: number; readonly name: string };

/**
 * κ for piece identity. Classifies once; the caller reports and the schema
 * validator bounds-checks `piece` afterwards (a non-numeric index arrives here
 * as NaN and is rejected there, NOT repaired here).
 */
export function resolveSharedPiece(
  names: readonly string[],
  index: unknown,
  name: unknown
): SharedPiece {
  const piece = Number(index);
  if (typeof name !== "string") return { tag: "index", piece };
  const byName = names.indexOf(name);
  if (byName < 0) return { tag: "unknown-name", piece, name };
  if (byName === piece) return { tag: "index", piece };
  return { tag: "renamed", piece: byName, staleIndex: piece, name };
}
