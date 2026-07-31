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
