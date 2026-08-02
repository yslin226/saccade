# Sandlot Baseball

Movement analysis for baseball, built on [Saccade](../../README.md).

Measures what a pitch or a swing actually did — joint angles, stride length,
the order the kinetic chain fired — and compares it against your own previous
session rather than against a textbook.

## Why consistency first

"You are 4.8° less separated than last time" is arithmetic. "You should be at
45°" needs an authority behind it, and being wrong about it is worse than
saying nothing. The comparison against your own history is the part that
cannot be wrong, so it is the part that ships first.

That only works if the numbers are stable. Ten analyses of one video produce
identical figures — measured before any of this was written, because every
metric inherits the answer. See
[`docs/plans/M3-sandlot-skeleton.md`](../../docs/plans/M3-sandlot-skeleton.md).

## Using it

```bash
uv run sandlot analyze <video>                      # measure and store
uv run sandlot analyze <video> --repeat 10          # check the numbers agree
uv run sandlot analyze <video> --movement swing     # also locate the bat
uv run sandlot compare <session-a> <session-b>      # what changed
```

Every number carries the frame it came from:

```
metric                             value  frames            unit
hip_shoulder_separation          63.9041  frame 21          degrees
stride_length                     1.9418  frame 21          torso lengths
elbow_flexion_L                  29.4803  frame 17          degrees
elbow_flexion_R                  32.9679  frame 19          degrees
kinetic_chain_order               1.0000  frame 0           fraction in order
```

Sessions are stored under `~/.sandlot/sessions/`; `--data-dir` overrides it.
Comparing two sessions built by different detector versions is refused rather
than reported — an upgrade can move a coordinate, and the resulting "change"
would be describing a version bump.

## What it measures, and why those

Driveline's OpenBiomechanics work found pitchers matched on arm speed still
differ by 13 mph, with stride length and kinetic-chain sequencing explaining
much of the rest; and that for hitters, bat speed barely predicts exit
velocity (R² = 0.097) while weight transfer explains a further 37.8%. So
these are the measurements the data says matter rather than the ones that are
easy to see.

| metric | what it is |
|---|---|
| hip-shoulder separation | angle between the two lines, at its maximum |
| elbow flexion | shoulder-elbow-wrist, at its tightest |
| stride length | ankle to ankle, in torso lengths |
| kinetic chain order | which segment peaked first, ground-up or not |
| weight transfer | how far the centre of mass moved (swings) |
| swing plane | the bat's path, when it can be located |

A metric that could not be computed is absent rather than estimated. The bat
in particular is often missing: COCO knows "baseball bat" as a shape, and a
bat at contact speed is a smear.

## Status

M3 complete. The next milestone is the one this project exists for — using
Saccade's active vision to decide which frames the detector got wrong, rather
than trusting every frame equally.

Known limitation, measured: a metric taken as an extremum over a delivery
lets the single worst-detected frame become the answer. An anatomically
impossible value is now refused, but a wrist misplaced by forty pixels still
produces a plausible angle, and catching those needs a second detector to
disagree with the first. That is M4.

## Layout

```
domain/          how a metric is computed, and what counts as a change
application/     the flow, and the ports it needs
infrastructure/  MediaPipe, YOLO, storage, Saccade wiring
interfaces/      the CLI
```

Dependencies point inward. `domain` imports nothing from the other three, so
a metric can be tested without a video, a detector, or a disk.

## License

MIT
