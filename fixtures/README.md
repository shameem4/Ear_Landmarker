# Test fixtures

Real photos with a head in frame, used to test the full
detector -> crop -> landmarker path. The repo previously had none, which is why
the web demo's duplicate-detection bug was only caught by eye.

`runs/test_output.jpg` is NOT a usable fixture: it is v1's annotated output, so
BlazeEar finds nothing in it at the default 0.70 confidence threshold.

Drop clean, unannotated frames here as `sample_01.png`, `sample_02.png`, ...
Anything in this folder is treated as personal data: it is for local testing and
should not be published or committed without the subject's consent.
