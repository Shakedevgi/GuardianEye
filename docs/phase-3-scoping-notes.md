# Phase 3 scoping — preliminary evidence from Phase 2 failure frames

**Status: preliminary. Three frames, one room, one session.** This is a head
start on Phase 3's Step 0 measurement, NOT a substitute for it. Run the real
measurement across the actual testing areas before committing to a class list.

Source frames (in `cv/captures/`, gitignored — photos of a real home):
`20260809-082938`, `20260809-083010`, `20260809-083249`, all
`yolo26x_imgsz1600`, conf 0.25.

Note when reading other captures from that session: several are
`yolo26l_imgsz640`, i.e. the *default* settings, not the high-quality ones.
Misses in those frames prove nothing — 640 is the setting Phase 2 showed
starves small objects. Only the `yolo26x_imgsz1600` frames test the ceiling.

## What worked

| Class | Confidence | Verdict |
|---|---|---|
| refrigerator | 0.88 – 0.93 | reliable |
| chair | 0.46 – 0.90 | reliable, confidence varies with occlusion |

Large furniture is not a problem. Stock COCO handles it.

## What failed — and why it matters

### Scissors: complete miss, and not a resolution problem

In `083249` the scissors hang on a white wall, roughly 300px wide,
unoccluded, well lit, viewed straight on. At `yolo26x` / imgsz 1600 they
produced **nothing** at conf 0.25.

This is the decisive observation. Every previous "COCO can't see scissors"
result was explainable by the object being tiny after downscaling. This one
is not: the object is large and clear, and the best model in the family at
near-native resolution still returned nothing.

Likely reason it's hard despite being obvious to a human: silver/white
scissors against a white wall is genuinely low contrast, and "mounted on a
wall" is far outside COCO's distribution for that class (COCO scissors are
on desks, in hands, on contrasting surfaces).

In `083010` scissors and a red-handled implement sit on the black chair,
partially cut off by the frame edge — also missed.

**Implication: CLAUDE.md decision 7 appears wrong.** It assumes stock COCO
covers knife and scissors adequately, with fine-tuning reserved for classes
COCO lacks. On this evidence, scissors (and probably knife) belong in the
fine-tuning set, which makes Phase 3 larger than currently planned. Confirm
with the Step 0 measurement before amending CLAUDE.md.

### Power strips / electrical outlets: no COCO class exists at all

Visible in two of the three frames: a wall-mounted power strip with three
sockets, a plugged-in adapter, and a trailing cable. COCO has no class for
outlets, power strips, extension leads, or cables.

This is a serious omission for this project specifically. The room even has
a "high voltage / danger of death" warning sign in it. Electrical sockets
are a canonical toddler hazard and the system currently cannot see them at
all — not "sees them badly," cannot see them.

**Strong candidate for a custom class**, alongside the choking-hazard
objects and stairs already named in PHASE_PLAN.

### Person detection is weaker than expected — this may be the bigger risk

- `082938`: an arm and hand fill a third of the frame — **not detected at all**
- `083010`: hand and forearm — `person 0.42`
- `083249`: finger and hand — `person 0.38`

All three are partial bodies (arm/hand entering frame, no torso or face).
Stock COCO's person class is trained overwhelmingly on whole or mostly-whole
people, and degrades badly on body parts.

Why this matters more than the hazard classes: **GuardianEye's entire risk
engine depends on detecting the child.** Layer B computes child-to-hazard
proximity, and the single most important moment to get right is a toddler
*reaching toward* a hazard — which is, geometrically, exactly this case: an
arm and hand extending into frame ahead of the body, possibly with the rest
of the child occluded by furniture.

If `person` scores 0.38 or vanishes in precisely that situation, the risk
engine's input is unreliable no matter how good the hazard detection is.

Caveat before over-reacting: these are adult arms at close range. A toddler
at room distance is a small whole body, which may well detect far better.
This needs deliberate testing with a realistic child-sized subject and
reaching poses before drawing conclusions — but it should be tested in
Phase 3, not discovered in Phase 4.

## Suggested Step 0 measurement, given the above

Test these deliberately rather than opportunistically, across the real
testing areas at varied height, angle and lighting:

1. **Scissors and knife** — on contrasting vs. matching backgrounds, on
   counters, on the floor, at 1–4m. Isolate whether contrast or context is
   the dominant factor.
2. **Power strips, sockets, trailing cables** — confirm the total absence
   and estimate how much data a custom class would need.
3. **Person, partial and occluded** — arm reaching into frame, body behind
   furniture, small subject at room distance, crouching/crawling poses.
   This deserves equal weight to the hazard classes.
4. **Everything already working** (refrigerator, chair, oven, sink) — record
   baseline confidences so Phase 3 fine-tuning can be checked for regression
   against them later.
