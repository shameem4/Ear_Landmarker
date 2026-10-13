# Independent 2D landmark QA

Checks that do not involve the landmark model's own geometry. They are
reported SEPARATELY and deliberately not fused: the value is that they fail
for different reasons, and a single blended score would destroy exactly the
property that makes them worth running.

    scripts/qa_silhouette_sam.py   SAM 2.1 mask -> outer silhouette
    scripts/qa_ridge_cv.py         sato -> internal curves, 3 channels
    scripts/qa_depth_dump.py       DA3 depth maps, feeds the depth channel
    scripts/qa_silhouette_show.py  best/worst cases, with the mask drawn
    scripts/qa_ridge_show.py       what the filters actually draw
    scripts/qa_agreement.py        PER-IMAGE agreement with the GT landmarks
    scripts/qa_common.py           padding mask, shared by all of the above

Everything heavy goes through `scripts/guard_run.sh <mem> <secs> <log> <cmd>`.

## The three ridge channels

    base    sato on intensity        the no-model default, needs no GPU
    clahe   sato on CLAHE intensity  the interior: tragus, canal, antitragus
    depth   sato on DA3 depth        the rim and the inner helix

CLAHE is free and strictly better on the interior groups (antitragus
0.74 -> 0.55, canal 0.76 -> 0.57, tragus 0.79 -> 0.63, superior crus
0.55 -> 0.46). It is WORSE on the outer rim, so run both, not one.

Depth is the strongest channel that exists for the rim and inner helix
(descending inner helix 0.55 -> 0.31, inner helix 0.79 -> 0.39, descending
helix 0.62 -> 0.39) and the worst for everything else -- tragus 2.01,
canal 1.75. It traces on 67% of images and is useless on 15%, and it knows
which: ear-vs-surround depth separation is 3.45 sd where it traces and
1.21 sd where it fails, so gate on that, which the script prints.

DA3-SMALL and DA3-BASE are Apache 2.0 for code and weights. The
METRIC-LARGE tier is CC BY-NC and must not be used for anything released.
`.venv_da3` is the only env with depth_anything_3 and it lacks
libnvrtc-builtins.so.13.0, so the second inference in a process dies --
put `exrcise_trainer/.venv_cu130/.../nvidia/cu13/lib` on LD_LIBRARY_PATH
and nothing needs installing. The qa_depth_dump.py docstring has the line.

## Which check covers which group

    group                     SAM silhouette   sato base  clahe  depth
    helix / descending helix  yes              yes        worse  BEST
    ear lobe                  yes              yes        worse  no
    ascending helix           yes              weak       worse  no
    ANTIHELIX (desc. inner)   NO -- internal   yes        no     BEST
    inner helix               no               weak       weak   BEST
    superior crus             no               yes        BEST   weak
    antitragus                no               weak       BEST   no
    tragus                    no               no         yes    no
    canal                     no               weak       BEST   no
    concha                    no               no         no     no

The concha is the one group nothing reaches: 1.05 base, 1.03 the best of
five channels tried. DA3-SMALL at 504 px renders the ear as a smooth blob
with no internal bowl, so depth does not see it either. It needs a shape
prior, not an image filter; do not spend more on image processing there.

## The hull-edge trap -- read this before believing a ratio

Ratios are against random in-hull controls, and the four outer groups
DEFINE the landmark convex hull, so they sit ON its edge: measured at 0.04
of the control distance. Any channel whose hits form a broad region
touching that edge scores ~0 for those groups no matter what the region
contains.

This is not hypothetical. A "shadow" channel -- blurred negative intensity
thresholded at the 85th percentile -- appeared to recover the ascending
helix from 0.62 to 0.34, and three other groups with it. It was an
artefact twice over: unclipped, 83-95% of its hits lay outside the hull
because it had found the HAIR, which is dark and borders exactly those
groups; clipped to the hull, it still won on the boundary alone. Every
number agreed with every other number and all of them were wrong. Only the
render showed it.

So: `qa_ridge_cv.py` prints an `edge` column beside the ratios and flags
any group at edge<0.25, and `qa_ridge_show.py` draws the hit mask and the
fraction of hits outside the hull. Run the show script and LOOK before
reporting a new channel. A thin line on the anatomy is a trace; a blob on
the hair is not.

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

Measure distance to the nearest traced structure, never the filter
response at the landmark's own pixel: ridges are 1-2 px wide, so a
one-pixel annotation offset reads as no response and understates a filter
that the overlay plainly shows working.

## Per-image agreement with GT

`qa_ridge_cv.py` gives a corpus-wide ratio, which is the wrong shape for
auditing: 0.55 overall is equally consistent with every image being
mediocre and with most being perfect and a few being wrong.
`qa_agreement.py` scores each image on its own, using the channel that
covers each group, and writes a CSV so the disagreeing rows can be pulled.

Over 500 images with the depth channel:

    p10 0.25   p25 0.35   p50 0.48   p75 0.66   p90 0.79   p95 0.93
    on structure 81.2%   marginal 15.8%   DISAGREES 3.0%

Over the WHOLE corpus, 10,535 images, pure CV with no GPU (omit
--depth-dir; the rim groups fall back per BEST_CV, which is base for the
helix, descending helix and antihelix and clahe for the inner helix):

    p10 0.35   p25 0.45   p50 0.59   p75 0.74   p90 0.88   p95 0.97
    on structure 69.0%   marginal 27.3%   DISAGREES 3.8%

Dropping depth costs the rim and nothing else -- inner helix 0.15 -> 0.61,
antihelix 0.13 -> 0.47, helix 0.19 -> 0.30 -- and barely moves the
disagreement rate (3.0% -> 3.8%), so the pure-CV run is the right default
for triage and the depth run is for characterising the rim. The whole
corpus takes about 40 s on 24 cores.

The disagreeing 3% are NOT bad annotations. Looked at: ears occluded by
hair, by a finger or by a foreground object; near edge-on poses; severe
motion blur. In several the annotation is the best a human could place and
the image simply has no visible structure to agree with. Agreement tracks
the depth channel's own confidence (rho -0.48 against ear-vs-surround
separation) and NOT ear size in pixels (rho +0.01), so it is measuring
image quality, not annotation scale.

Audited with SAM, the independent check: the 200 worst rows against 200
from the best half. The outer ring sits a MEDIAN 1.15% of ear height INSIDE
the silhouette in the worst rows (control 1.55% inside); no row has it more
than 10% outside, and 1 of 175 is beyond 8% -- those few have SAM masks
covering only 0.63-0.80 of the landmark hull, i.e. SAM under-segmented. So
the annotations hold up even in the tail; what fails there is the image.

Do not trust an eyeball of a crushed-black crop on this point. Stretching
the exposure made the helix ring look inflated onto the background in every
one of the worst rows, and the measurement says the opposite -- what reads
as background in a dark crop is dark ear. The render is the right tool for
"is this channel tracing hair", and the wrong one for "is this landmark a
few pixels out".

Treat the CSV as a triage list, not a verdict. The honest reading is "these
images cannot be verified from pixels", which for a model-seeded corpus
also means these are the rows where a seeded landmark could be a
hallucination and nothing in the image would contradict it.

## Letterbox padding

45.6% of data/manual images carry more than 5% constant-valued padding and
26.8% carry more than 15%. A flat pad meets the photo at a perfectly
straight, perfectly sharp edge -- the strongest ridge in the frame -- so
the filters trace the LETTERBOX and it counts as structure.
`qa_common.padding_mask` removes it and all four scripts apply it.

Correcting it moved the corpus numbers almost not at all (median agreement
0.47 -> 0.48, on-structure 81.6% -> 81.2%), because the pad lies outside
the landmark hull and so rarely decided a landmark's nearest structure. It
is kept because it is correct and because it WILL matter on tight crops,
not because it changed this answer. Pass --keep-padding to reproduce the
old numbers.

## Finding bad annotations (a different defect)

`qa_find_bad_contours.py` ranks rows by how many of the 20 outer landmarks
sit more than --tol% of ear height off the SAM silhouette. This is NOT what
qa_agreement.py measures and the two do not overlap: agreement finds hard
IMAGES, this finds wrong LABELS. Measured on 200 agreement-worst rows
against 200 agreement-best, the misrouted-point rate was the same in both
(4.84% vs 4.70%), so filtering a corpus on agreement drops the wrong rows.

THE HEADLINE RATE IS SET BY THE MASK GATE, so quote it with the gate:

    all trusted masks        11.4% of rows have >=3 misrouted points
    tight masks (bloat<1.4)   2.4% of rows   <- use this one

The count tracks mask bloat at rho +0.474, because a mask carrying hair
pushes the rim outward and makes a correct contour look wrong. Spot-checked
by eye at the tight threshold, about two thirds of flags are genuine, so the
true bad-annotation rate is near 1.5% -- roughly 125 rows of the 8,427-row
train split. Real, worth fixing by hand, far too small to move a model.

SAM 2.1 IS THE BOTTLENECK HERE, not the method: it is rejected on a third of
images and only ~43% of the rest give a tight enough mask to judge a
contour. A dedicated ear segmenter would raise the usable fraction and the
precision together.

## Segmenter options, reviewed Oct 2026

SAM 2.1 is the bottleneck for the contour audit -- rejected on 33% of
images, and only ~43% of the rest give a mask tight enough to judge a
contour. What else there is:

    SAM 3          best model; gated weights, Meta's custom SAM License
    Florence-2     MIT and ungated -- TRIED, REJECTED, see below
    Roboflow 590   CC BY 4.0 DATASET; download it, do not use the API
    umitkacar      MIT U-Net scaffold, no weights, no training data named
    jorge-jrzz     no LICENSE file at all, and YOLOv8 (Ultralytics AGPL)

Florence-2 was the obvious pick on licence -- MIT for code and weights,
ungated, natively supported in transformers 5.x (use
florence-community/Florence-2-large, NOT microsoft/Florence-2-large, whose
remote-code layout has no image_token and raises in the native processor;
and the class is Florence2ForConditionalGeneration, not
AutoModelForCausalLM). It does not work here: coarse low-vertex polygons,
60% dropped by the gates against SAM's 33%, and empty or degenerate masks on
most ears when rendered. Right tool for "where is the object", wrong one for
"where exactly is the rim".

The Roboflow entry's value is the DATASET, not the hosted model: CC BY 4.0
means it can be downloaded and trained on locally, which avoids both the
rate limit and sending images to a third party.

On jorge-jrzz, the two blockers are different in kind. Ultralytics AGPL-3.0
triggers on distribution or network service, so purely internal use does not
trip it. The absent LICENSE file is not a restrictive licence to read
permissively -- it is the absence of any grant, so there is nothing to read.

## Which segmenter -- measured, Oct 2026

Same 40 rows, same gates, head to head:

    model          gates dropped   interior/outer   renders
    SAM 2.1             30.8%          3.05x        noisy, fragments, takes hair
    SAM 3               12.5%          3.52x        TIGHT, accurate rim  <-- use
    U-Net (local)        5.0%          3.62x        blobby, misses the rim
    Florence-2          60.0%          2.30x        empty or degenerate

USE SAM 3. And note what nearly went wrong here: the U-Net wins every gate
statistic and loses badly on the renders. It emits ear-ISH, roughly
ear-SIZED regions, which is precisely what gates on area, coverage and hull
overlap reward -- it swallowed an eye and a slab of background on one row
while scoring the best separation of any model. Gate pass rate measures
"produced a plausible blob", not "found the rim". Do not pick a segmenter on
these numbers without drawing the masks.

SAM 3 is prompted with the CONCEPT "ear" and needs no box or point; the
instance overlapping the landmark hull best is kept, which is the same
coarse localisation SAM 2.1 got from its box, so the comparison is fair.
Licence: Meta's custom SAM License, gated weights, accepted by the user for
this account. The text carries NO Llama-style restriction on using outputs
to train or distil, the grant is royalty-free with no commercial-use bar,
and the live restrictions are acceptable-use and no-reverse-engineering.

The local U-Net stays as the licence-clean fallback -- torchvision (BSD) on
the CC BY 4.0 Roboflow set (590 images, v3), trained locally, val IoU 0.9185
and test IoU 0.9434 IN DOMAIN. In-domain IoU on 256px Roboflow crops says
little about rim precision on wild ears, as the renders show. Its encoder is
ImageNet-pretrained, which is a grey area against the commercially-clean
rule; --scratch trains a fully clean one. CC BY requires crediting SaifulULL
wherever this model or its outputs are published.

## The bad-annotation rate, measured with SAM 3

SAM 3 changes this number, because the old one was limited by SAM 2.1's
masks rather than by the annotations. Over the same 1,500 screened rows:

    segmenter   usable masks   rows with >=3 misrouted points   precision
    SAM 2.1      989 (then 424 after the bloat gate)   2.4%     ~2 in 3
    SAM 3       1390                                   4.9%     8 of 8

SAM 3's masks come out TIGHTER than the landmark hull on every row (bloat
0.61-0.90), so the over-segmentation gate never binds and a flag now means
what it should: the annotation is bigger than the ear. Spot-checking the top
eight by eye, all eight are genuine.

So the honest rate is ~4.9%, about 412 rows of the 8,427-row train split --
roughly three times the earlier estimate, which was measuring SAM 2.1.

ONE FAILURE MODE DOMINATES. The annotated contour over-extends past the ear
into HAIR, and almost always at the top:

    ascending_helix   3.02% of its points >15% off the rim
    ear_lobe          2.04%
    descending_helix  1.37%
    helix             0.14%

The ascending helix is where the ear meets the hairline, and that is where
the annotations run on. The helix proper, with sky or background behind it,
is nearly clean. Worth fixing as a class rather than row by row.

## jorge-jrzz, evaluated

Tried on the same rows, in a throwaway venv so the AGPL package never
entered the mocap env. best_v2 dropped 20.0% of images with interior/outer
2.17x; best_v4 dropped 42.5% at 1.47x -- both well behind SAM 3's 12.5% and
3.52x. The renders show why: YOLOv8-seg crops its prototype masks to the
detection box, so the masks are chopped by hard rectangular edges and carry
slabs of background. Rejected on quality, independently of the missing
LICENCE.

Its real value is a pointer: the dataset bundled in it
(universe.roboflow.com/mutyi/ear-segmentation v3, 434 images) is **MIT**,
which is a better licence than the CC BY set since it carries no attribution
obligation. Combine the two for ~1,024 images if the local U-Net is ever
retrained.

## Prompting and VLM boxes (the lobe shortfall)

`qa_sam3_prompt_sweep.py`  sweep SAM 3 text prompts, scored by the signed
                           distance from each annotated point to the mask
                           edge; watch point 19 minus point 12.
`qa_ear_box_vlm.py`        whole-ear box from a VLM. `ollama` is local
                           qwen2.5vl:7b; `gemini` UPLOADS THE CROPS.
`qa_vlmbox_show.py`        render those boxes over the annotation.
`qa_sam3_boxed.py`         re-score SAM 3 given a VLM box.
`qa_sam3_boxed_show.py`    text vs box masks side by side.

Settled, see RESULTS.md: no prompt and no box moves SAM 3's boundary, so
the lobe shortfall cannot be prompted away. The Gemini box is the useful
output -- median IoU 0.909 with the annotation's own bounding box, no mask
involved -- and is the route to narrowing the 0.86%-5.40% bad-contour
range. Each VLM must be asked in its own coordinate convention, and a
format example must never contain a usable value.
