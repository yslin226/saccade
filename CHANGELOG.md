# Changelog

Public API changes, per rule 4. Names in `saccade.__all__` are covered here;
`_`-prefixed modules are internal and change without notice.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project is `0.x`: the public API may still change, but every change is
recorded.

## [Unreleased]

### Fixed

- **A measurement that contradicts the model now decides the answer.**
  Previously a conflict only lowered confidence, and the model's contradicted
  statement was still returned — the tool was an adviser, not a referee, which
  is the opposite of what rule 1 and the project's stated principle require.

  Measured on BlindTest's touching-circles task with
  `qwen/qwen3-vl-8b-instruct` over 150 items. A measurement contradicted the
  model on 53 of them:

  | | before | after |
  |---|---|---|
  | overall | 64.0% | 98.0% |
  | the 53 contradicted items | 1.9% | 98.1% |
  | the 97 uncontested items | 97.9% | 97.9% |

  The conflict count is 53 either way, and the VLM's own replies were
  identical (served from cache) — the entire difference is that the geometry's
  verdict is now used. It had the right answer in all 53 cases and was never
  asked for it.

  Two conditions guard the overruling: the tool must have declared an
  `answer_key`, and the measurement must have been made on the whole image.
  A conflict raised by a cropped view still leaves the model's answer alone,
  for the same reason a cropped statement cannot answer a whole-image
  question.

### Added

- **Three geometric primitives, for applications to build domain tools from.**
  Pure arithmetic on coordinates, with no domain knowledge: an angle is an
  angle whether the vertex is an elbow, a robot joint, or a road junction.

  - `angle_between(a, vertex, b)` — the angle at `vertex`, in degrees, in
    [0, 180]. Unsigned. The vertex is the middle argument, matching how the
    angle is written; passing it first measures something else and returns a
    plausible number.
  - `speed(p1, p2, dt)` — distance over elapsed time, in whatever units the
    caller passed in. Named `speed` rather than `velocity` because it is an
    unsigned scalar, and the other name would imply a direction it does not
    carry.
  - `centroid(points)` — the arithmetic mean. The centre of mass only when
    the points are equally weighted, which body segments are not.

  Each raises rather than returning a plausible number on degenerate input:
  a ray of zero length has no direction, zero elapsed time makes a speed
  infinite rather than large, and the mean of nothing is not a position.
  Returning `0.0`, `inf` or the origin would each survive a comparison
  against a threshold and quietly answer the wrong question.

### Known issues

- `saccade.geometry` refuses to import without OpenCV, but nothing in the
  engine actually uses it — the import exists only as an availability check.
  Callers who want `distance` or `angle_between`, which are `math` and
  nothing else, are made to download 60MB for a dependency that is never
  called. The gate should move to whichever module first needs contour
  finding.

- **`ActiveVisionAgent(..., choose_tools=True)` — the model picks which tools
  to run.** Off by default. The loop otherwise runs every registered tool on
  every step, which is right only while every tool was written for the
  question: a tool that measures the wrong thing does not merely waste a call,
  it produces a confident number the verifier uses to overrule the model.

  Measured on `qwen/qwen3-vl-8b-instruct` with both tasks' tools offered on
  both tasks, cache disabled so the model was genuinely asked:

  | | baseline | one right tool, always run | whole toolbox, model chooses |
  |---|---|---|---|
  | touching circles | 65.3% | 98.0% | **99.0%** |
  | line intersections | 86.7% | 99.3% | **100.0%** |

  Over 40 separately probed items the model picked the applicable tool 40
  times and fell back 0 times. Choosing an instrument and reading it stay
  separate jobs: only the first is delegated, and `is_measurement` still
  governs whether the resulting measurement may overrule anything.

  A failed choice — an unreadable reply, a hallucinated tool name, a network
  error — runs every tool, which is the previous behaviour. `ToolChoice`
  records `fallback` so "chose everything" stays distinguishable from "was
  not understood"; the two run identically and mean opposite things.

- `NullCache` — a `CachePort` that stores nothing, exposed as `--no-cache` on
  the benchmark runner. Rule 6 makes caching the default for good reason, but
  a cache turns a re-run into a replay: an experiment that changes how the
  loop behaves still receives the answers the *old* loop provoked, and looks
  like a fresh measurement while measuring nothing new.

- `Verification.verdict_key` — the entry of `computed` that a tool declared as
  its answer, or `None` when no tool claimed one. Knowing a statement is
  contradicted is not enough to replace it: `computed` also carries diagnostic
  figures, and a line counter reporting 1 crossing across 300 shared columns
  must answer 1.
- `Verification.verdict` — the value at `verdict_key`, or `None`. Read
  `verdict_key` rather than `verdict` to distinguish "no tool answered" from
  "a tool measured `False`".

### Notes

- Results produced before this fix understate `saccade-tools` wherever
  conflicts occurred. The M2 ablation figures (gpt-4.1 +12.0%, gpt-5.4 +0.7%)
  were measured under the old behaviour and need re-running before they can be
  cited.
