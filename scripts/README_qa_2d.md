# Independent 2D landmark QA

Two checks that do not involve the landmark model's own geometry. They are
reported SEPARATELY and deliberately not fused: the value is that they fail
for different reasons, and a single blended score would destroy exactly the
property that makes them worth running.

    scripts/qa_silhouette_sam.py   SAM 2.1 mask -> outer silhouette
    scripts/qa_ridge_cv.py         sato dark-ridge -> internal curves
    scripts/qa_silhouette_show.py  best/worst cases, with the mask drawn
    scripts/qa_ridge_show.py       what the filters actually draw

Everything heavy goes through `scripts/guard_run.sh <mem> <secs> <log> <cmd>`.

## Which check covers which group

    group                     SAM silhouette      sato ridges
    helix / descending helix  yes                 yes
    ear lobe                  yes                 yes
    ascending helix           yes                 weak
    ANTIHELIX (desc. inner)   NO -- internal      yes, the strongest of all
    superior crus             no                  yes
    antitragus                no                  yes
    inferior crus, inner hx   no                  weak
    tragus                    no                  no
    concha                    no                  no

SAM only sees the outline, so it can check points 0-19 and nothing else.
The antihelix and crura are internal ridges, which is why the ridge filter
is the only one that reaches them -- and it scores them better than the
helix. The concha is a smooth bowl and the tragus a soft bump; neither
method sees either, and they stay unverifiable from a single photo.

## What each is good for

SAM also yields a closed region, so it gives a signed inside/outside, a
better crop than a detector box, and a silhouette constraint for camera
recovery. It needs a GPU, and it fails outright on about 7% of images --
segmenting hair or background -- so gate on overlap with the landmark
convex hull. Its own IoU score does NOT work as that gate: 0.39 on a
near-perfect mask, 0.72 on one that had taken the hair.

sato needs no model and no GPU, never returned a blank over 200 images,
and is the better default for a quick sweep. Canny is not: it produced a
completely blank edge map on a smooth, evenly-lit ear.

## Reading the numbers

Report the MEDIAN. The mean is dominated by a tail, and the tail is the
segmenter failing rather than the landmarks -- confirmed by looking at the
worst cases, where the points traced the ear correctly every time.

Measure distance to the nearest traced ridge, never the filter response at
the landmark's own pixel: ridges are 1-2 px wide, so a one-pixel
annotation offset reads as no response and understates a filter that the
overlay plainly shows working.
