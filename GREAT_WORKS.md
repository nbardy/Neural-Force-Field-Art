# Great Works

Settings worth never losing. Each entry is a **complete dock recipe** — every
dial, plus the commit it ran on — so a piece that came out right can be brought
back byte-for-byte, on any later build.

The app already serializes the whole dock: **SHARE → COPY LINK** puts a
`?dock=<base64url>` URL on your clipboard, **JSON** gives you the same blob as
text. That blob is what goes here. The piece is resolved **by name**, so a
recipe survives GALLERY being reordered (see `resolveSharedPiece` in
[src/share.ts](src/share.ts)).

## How to use this file

**Save a keeper.** Click **JSON** in the dock, paste it as a new entry below,
click **COPY LINK**, paste that as the entry's link, and record
`git rev-parse HEAD`. Then:

```bash
bun tools/great_works_verify.ts
```

That asserts the link and the JSON describe the same dock, that the link still
decodes, that the piece name still exists, and that the commit is still in the
repo. **Run it before committing a new entry** — an entry that does not verify
is worse than no entry, because it looks like a backup.

**Restore one.** Open the link. That is the whole restore path — it carries
every dial and resolves the piece by name.

**Roll the code back too** (only needed if a later commit changed the kernels or
the piece's gallery defaults, not just dials):

```bash
git switch --detach <commit>
```

---

## Adversary · Pair · HashGrid · Curl — ink swirls

Long laminar filaments with curl strokes over the hashgrid dual field. The
adversary runs the **angle + relative-scale** objective (`A+S ADV`), with the
energy anchor holding absolute RMS so variance cannot be bought by blowing the
field up. Held ~60 FPS at 70k particles.

At capture time (commit `b72aa54`) this objective was a dock override on top of
the piece's shipped `soft-angle` default; it has since been promoted to be the
default. The recipe below is unchanged either way — that is the point of
recording it.

- **Commit:** `b72aa542d29834f573e172d59d0e53362bff55bd`
- **Captured on deploy:** gh-pages `8b9026c`, bundle `index.74c2c06f.js`. That
  is provenance, not where to look now — the link below is piece-resolved by
  name and keeps working across deploys. Live site:
  <https://nbardy.github.io/Neural-Force-Field-Art/>
- **Diverges from the gallery default in:** `border` (`wrap` → `reset`) and
  `discriminatorLearningRate` (3e-3 → 7.2e-4) — and ONLY those, because neither
  has a piece field to be promoted into (the dock hardcodes `{tag:"wrap"}` in
  `defaultsForPiece`, [src/index.tsx](src/index.tsx), and D lr is `?dLR=` only).
  `loss` (`soft-angle` → `angle-relative-scale`), `drive` (0.65 → 0.90) and the
  generator lr (0.001 → 0.0048) WERE divergences and are now this piece's
  shipped GALLERY defaults — see the PROMOTED GREAT WORK note on the entry in
  [src/main.ts](src/main.ts). The recipe above is left untouched on purpose:
  the record has to outlive the default.

**Link**

```
https://nbardy.github.io/Neural-Force-Field-Art/?dock=eyJydW50aW1lIjp7InBpZWNlIjoxNywiYm9yZGVyIjp7InRhZyI6InJlc2V0In0sImVuY29kaW5nIjp7InRhZyI6InBhaXItcm90YXRpb24tc2NhbGUtYWRqdXN0ZWQifSwidGFyZ2V0Ijp7InRhZyI6ImZvcmNlIn0sImxvc3MiOnsidGFnIjoiYW5nbGUtcmVsYXRpdmUtc2NhbGUiLCJ0YXUiOjAuMDUsInNjYWxlV2VpZ2h0IjowLjUsImVuZXJneVdlaWdodCI6MC4xLCJlbmVyZ3lUYXJnZXQiOjAuMzV9LCJhZHZlcnNhcnlLaW5kIjoid3RhIiwiayI6NCwicmVsYXhFcHMiOjAuMDUsImFyY2hQcmVzZXQiOm51bGx9LCJwYXJ0aWNsZXMiOjcwMDAwLCJzYW1wbGVzIjoyNTYsIm1heFZlbG9jaXR5IjoyNCwiZHJpdmUiOjAuOSwiZ2VuZXJhdG9yTGVhcm5pbmdSYXRlIjowLjAwNDgsImRpc2NyaW1pbmF0b3JMZWFybmluZ1JhdGUiOjAuMDAwNzIsInJlc2V0UmF0ZSI6MC4wMDMsImRlY2F5IjowLjk0LCJsb29rIjoiZ2hvc3QiLCJibGVuZCI6MC41NSwic3Ryb2tlU3R5bGUiOiJjdXJsIiwic3Ryb2tlTGVuZ3RoIjozLCJhZHZXZWlnaHQiOjAuMDE1LCJjb2xvck1vZGUiOnsidGFnIjoidmVsb2NpdHkifSwicGllY2VOYW1lIjoiQWR2ZXJzYXJ5IMK3IFBhaXIgwrcgSGFzaEdyaWQgwrcgQ3VybCJ9
```

**Settings**

```json
{
  "runtime": {
    "piece": 17,
    "border": {
      "tag": "reset"
    },
    "encoding": {
      "tag": "pair-rotation-scale-adjusted"
    },
    "target": {
      "tag": "force"
    },
    "loss": {
      "tag": "angle-relative-scale",
      "tau": 0.05,
      "scaleWeight": 0.5,
      "energyWeight": 0.1,
      "energyTarget": 0.35
    },
    "adversaryKind": "wta",
    "k": 4,
    "relaxEps": 0.05,
    "archPreset": null
  },
  "particles": 70000,
  "samples": 256,
  "maxVelocity": 24,
  "drive": 0.9,
  "generatorLearningRate": 0.0048,
  "discriminatorLearningRate": 0.00072,
  "resetRate": 0.003,
  "decay": 0.94,
  "look": "ghost",
  "blend": 0.55,
  "strokeStyle": "curl",
  "strokeLength": 3,
  "advWeight": 0.015,
  "colorMode": {
    "tag": "velocity"
  },
  "pieceName": "Adversary · Pair · HashGrid · Curl"
}
```

> **Provenance.** Reconstructed from the HUD/dock readout rather than exported
> with the JSON button, then **verified end-to-end against the live deploy**: the
> link above was loaded on <https://nbardy.github.io/Neural-Force-Field-Art/> and
> every dial came back matching — particles/train B/max vel/drive/respawn/border,
> trails/stroke/length, blend, objective/target/loss, soft τ/scale w/energy w/
> energy, tuple/observer, reward, G lr/D lr, K/ε, color — plus the telemetry line
> `wta k=4 ε=0.05 · force · angle-relative-scale w 0.015 (fused)`.
>
> One field is derived rather than observed: `look` is `"ghost"` because this
> piece is not `lookEditable`, so the dock hides the control and the value falls
> out of `inkLookFromRenderer("alpha-fade")`. If a future build makes ink look
> editable here, re-export.

---

## Promoting an entry to the piece default

The dock JSON and a `GALLERY` entry are different shapes — the recipe is *live
dials*, the gallery entry is *construction*. The mapping for the fields that
differ:

| dock JSON | GALLERY field |
|---|---|
| `particles` | `particleCount` |
| `samples` | dock-only (train batch; not a piece field) |
| `maxVelocity` | `maxVelocity` |
| `drive` | `drive` — **also recompute** `forceMagnitude: forceMagnitudeForDrive(drive, maxVelocity, friction)` |
| `resetRate` | `resetRate` |
| `decay` / `look` | `renderer` + `alphaBlend` |
| `strokeStyle` / `strokeLength` | `stroke` / `strokeLen` |
| `blend` | `alpha` on `createFieldFromArch({...})` |
| `runtime.border` | **no piece field exists.** `ArtPieceConfig` has no `border`; `startLoop` reads `options.overrides?.border ?? {tag:"wrap"}` and the dock hardcodes `{tag:"wrap"}` in `defaultsForPiece`, passing it down as an override. A piece CANNOT default to bounce/reset today — the `?dock=` link is the only way. |
| `runtime.loss`, `.target`, `.encoding`, `.k`, `.relaxEps` | the `adversary:` block |
| `advWeight` | `adversary.weight` |
| `generatorLearningRate` | `learningRate` |
| `discriminatorLearningRate` | dock-only (no piece field today) |

`forceMagnitude` is the trap: it is stored **derived** from drive, so setting
`drive` alone in a GALLERY entry silently keeps the old force scale.

**Do not** delete the entry from this file when you promote it. The point of the
record is that it outlives the gallery default, which will drift.
