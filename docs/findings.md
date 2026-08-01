# What the measurements say

Every figure traces to a JSON file in
[`benchmarks/blindtest/results/`](../benchmarks/blindtest/results/) with
per-item answers. Runs noted as such had the response cache disabled, so the
model was genuinely asked rather than replayed.

## A referee closes the gap between models

Same two tasks, one tool given rather than chosen, 150 items each:

| baseline | with the tool | gain |
|---|---|---|
| 38.0% (GPT-4.1, lines) | 99.3% | **+61.3** |
| 65.3% (Qwen-8B, circles) | 98.0% | **+32.7** |
| 78.0% (GPT-4.1, circles) | 98.0% | +20.0 |
| 79.3% (Qwen-30B, lines) | 99.3% | +20.0 |
| 82.7% (Qwen-30B, circles) | 98.0% | +15.3 |
| 86.7% (Qwen-8B, lines) | 99.3% | +12.6 |
| 90.7% (GPT-5.4, circles) | 98.0% | +7.3 |
| 98.7% (GPT-5.4, lines) | 99.3% | +0.6 |

Monotonic, no exceptions. Bare, the four models span 60.7 percentage points;
with a referee they span 1.3.

## The model can pick its own tool

100 items per task, `qwen/qwen3-vl-8b-instruct`, cache disabled. The toolbox
holds the tool that applies, the tool for the other task, and three decoys
that measure something true which does not answer the question.

| | one look | toolbox, model chooses |
|---|---|---|
| Touching Circles | 65.3% | **98.0%** |
| Line Plot Intersections | 86.7% | **100.0%** |

### It does not degrade with catalogue size

Five tools is a pair with distractors. The catalogue was padded with plausible
instruments for questions nobody asked — edge density, stroke width,
convexity — and the choice measured again at each size, 20 items per task:

| toolbox | circles | lines | fallbacks |
|---|---|---|---|
| 5 tools | 18/20 | 20/20 | 0 |
| 10 tools | **20/20** | 20/20 | 0 |
| 20 tools | **20/20** | 20/20 | 0 |

80 choices, no fallbacks, and both lapses are at the *smallest* size — each
took the applicable tool and the sharp decoy together, which never happened at
10 or 20. Why a bigger catalogue reads as easier is not something this
measures.

## Extra fixations do not help

Measured three ways, null in all three.

| task | baseline | more looks | test |
|---|---|---|---|
| Touching Circles | 90.7% | 93.3% | p = 0.39 |
| Touching Circles (GPT-4.1) | 78.0% | 77.3% | p = 1.00 |
| grid_blank | 69.0% | 77.0% | McNemar p = 0.25 (3 fixed, 0 broken, 70 paired) |
| olympic_circles | 20.0% | 20.0% | p = 1.00, 0 changed |

`olympic_circles` spent 3.5× the tokens to change nothing at all.

The distinction that took three failed experiments to see: these measured a
*hardcoded rule* deciding which corner to magnify. The model deciding which
*instrument* to reach for is a different question, and that one it can answer.

## Two defects the benchmark found

Neither was caught by the 577 tests that existed at the time. Both were only
visible against real data, and both are pinned by tests now.

### The referee could object but not overrule

A measurement contradicting the model only lowered confidence; the
contradicted answer was still returned.

| | |
|---|---|
| items where a tool objected | 53 → **1.9%** correct |
| items where none did | 97 → 97.9% correct |

The geometry had the right answer in all 53 and was never asked for it.
Fixing it took the task from 64.0% to 98.0%.

### A bare verdict was judged by every measurement at once

"No" names no subject. Checked against one boolean measurement it is
unambiguous; checked against two that contradict each other, it is being read
as a claim about both — and one of those tools is answering a different
question.

Found by adding decoys. `bounding_box_overlap` reports that two bounding boxes
overlap while the circles do not; both statements are true.

| | before | after |
|---|---|---|
| overall | 93.0% | **98.0%** |
| items where a tool spoke | 86.0% | **100.0%** |

37 conflicts after the fix, 37 correct. The rule is narrow: measurements that
concur stay unambiguous whatever each was measuring, so a single tool behaves
exactly as before.

## Video: what a measurement cannot answer

Two pose detectors disagreeing predicts which frames are wrong — weakly.
AUROC 0.638 over 890 held-out frames, against a 0.70 bar fixed before the run.
Four single-detector geometric signals did worse, three of them worse than
chance.

What disagreement cannot say is *why*, and the causes need different handling:
blur means discard and interpolate, occlusion means the position may still be
inferable, a lost track means reinitialise. The numbers are identical in all
three.

48 frames, GPT-5.4, 12 clips across 8 actions:

| | |
|---|---|
| frames where the detectors agreed | **24/24 called CLEAR** — no invented problems |
| true pose error where it said CLEAR | 0.964 body widths (n=32) |
| true pose error where it named a fault | **2.269** (n=7) |
| self-consistency (asked twice) | 67% |

A 2.35× separation, measured against Penn Action's hand-labelled joints, which
the model never saw.

The limit: Penn Action labels joint positions, not why a frame is hard. This
measures whether the VLM picks out the bad frames — it does — and cannot
measure whether the reason it gives is the right one. Self-consistency at 67%
means one question is not enough.

## Retracted

An earlier AUROC of 0.713 for detector disagreement was measuring a bug.
`shoulder_width` accepted collapsed detections of a couple of pixels, and
dividing an ordinary 60px gap by 2px produced disagreements of 150 body
widths — on exactly the frames that were badly wrong anyway. The bug predicted
the error; the signal did not. The corrected figure is 0.638.

An earlier "2.78× on flagged frames" for joint travel came from tuning and
evaluating on the same clips. On held-out actions it scored 0.64× — below 1.0,
meaning the frames it flagged were the *better* ones.
