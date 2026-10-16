# Vendored, not authored here

`box_tracker.py` is a verbatim copy of BlazeEar's `utils/box_tracker.py`, and
`../docs/ear_tracker.js` of its `docs/ear_tracker.js`, taken at BlazeEar
**3f76556**. They are pinned to each other by a shared 40-frame behaviour
fixture in BlazeEar, which is where both are maintained.

**Do not edit either file here.** A change belongs in BlazeEar, because the
constants were measured there -- 7,500 frames of 25 conversational clips -- and
a constant only means what it was measured to mean alongside the association
rule it was measured with. Re-copy both together; copying one alone breaks the
cross-language parity the fixture exists to protect.

`ear_tracker.js` is deliberately a **classic script, not an ES module**. From an
ES module: `import './ear_tracker.js'; const { BoxTracker } = globalThis;`.
Adding a bare `export` would break BlazeEar's `require()`-based parity test,
which fails silently rather than loudly.
