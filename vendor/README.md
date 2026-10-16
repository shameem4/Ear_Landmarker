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

## Everything vendored from BlazeEar

Five files, all byte-identical copies, all pinned by
`tests/test_vendored_tracker.py`:

| here | upstream |
|---|---|
| `vendor/box_tracker.py` | `utils/box_tracker.py` |
| `../docs/ear_tracker.js` | `docs/ear_tracker.js` |
| `../docs/blazeear_inference.js` | `docs/blazeear_inference.js` |
| `../docs/BlazeEar_web.onnx` | `docs/BlazeEar_web.onnx` |
| `../docs/BlazeFace_web.onnx` | `docs/BlazeFace_web.onnx` |

`blazeear_inference.js` fell one comment-only commit behind before this list
existed, found by headSize-gnm rather than by either repo. The two files that
were pinned did not drift. **Add any newly vendored file to that test**, or it
is free to do the same.

The pin earned itself within the hour: re-syncing `blazeear_inference.js`
surfaced that `ear_tracker.js` had *also* moved, because BlazeEar had just
fixed the JS/Python smoother divergence this repo reported -- JS accepted only
a class where Python took any zero-arg factory. With both languages now
accepting a factory, `docs/smoothing.js` passes one, the same shape the Python
side uses, instead of the subclass it originally needed.
