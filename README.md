# Saccade

Ask a frontier vision model whether two circles overlap, or how many times two
lines cross, and it will get it wrong about as often as a coin flip. The
[*Vision Language Models Are Blind*](https://vlmsareblind.github.io/) benchmark
puts seven such tasks — the kind a child solves instantly — to four leading
models. They average **58.07%**. The best of them, Claude 3.5 Sonnet, manages
77.84%. On counting line intersections the field drops to **56.84%**. Humans
score 100%. The paper's own summary: their vision is "at best like that of a
person with myopia seeing fine details as blurry, and at worst like an
intelligent person that is blind making educated guesses."

The interesting question is why. A 2026 position paper,
[*The Systemic Lack of Agency in Visual Reasoning*](https://arxiv.org/pdf/2606.14795),
argues it is not acuity — it is passivity. A VLM gets one forward pass over a
fixed image. It cannot decide to look closer at the part that matters, cannot
go back and check, cannot turn a reasoning need into a visual action.

Human eyes do none of that in one glance either. They saccade — three to four
jumps a second, sampling the scene from different positions while the brain
stitches the samples together. What a VLM does is answer from the first
fixation.

Saccade adds that loop at inference time. No training, no fine-tuning: the
agent decides where to look next, crops and magnifies, asks the model what it
sees, and then — this is the part that matters — checks the answer against
something that can actually count. The model gets to explain. It does not get
to measure.

```
❌  image → VLM → "the circles look like they overlap"
✅  image → geometry → centre distance 47px, radii sum 52px → they overlap
                    → VLM explains the result, with the numbers attached
```

The position paper lists active vision systems and iterative exploration
frameworks as unexplored directions, and the work it points to mostly involves
new architectures and RL budgets. This is the cheap path up the same hill:
find out how far you get by giving an existing model agency over its own
attention.

## Where this is now

**M1 — the loop runs.** `ActiveVisionAgent` plans where to look, crops and
magnifies, asks the model, and records an evidence chain. The BlindTest
runner works end to end against real models, and a rerun reproduces a run
exactly from cache.

What is not done: the verifier has nothing to verify against yet. Confronting
a claim with a measurement needs a tool that can measure — for two circles in
a photograph, that means detecting the circles before the geometry can judge
them. Until that lands, the loop is looking more than once but not checking
itself, which is the half of the idea that matters.

No accuracy figures here. Runs exist under `benchmarks/blindtest/results/`
with per-item detail, but they measure a loop with its verification stage
inert, so quoting them as the effect of active vision would be wrong. A
comparison worth publishing arrives in M2, alongside the ablations that say
which part did the work.

## How it works

```
             ┌──────────────────────────────────────────┐
             │                                          │
             ▼                                          │
    1. Plan  ──▶  2. Act  ──▶  3. Observe  ──▶  4. Verify
   where to      crop/zoom/     ask the VLM     confront the claim
   look next     annotate       what it sees    with measurement
                                                        │
                                    5. Record ◀──────────┘
                            screenshot, numbers, reasoning
```

The loop runs until verified confidence clears the threshold or the step
budget runs out. Running out is a normal result, not an error — you get back
`converged=False` and the full evidence chain that got you there.

Most agents treat tools as an extension of the model, and believe what comes
back. Here a tool is the model's referee: only results flagged
`is_measurement=True` may be used to contradict what the VLM said. A tool that
is itself a VLM does not qualify — one blind witness cannot vouch for another.

## Install

```bash
uv add saccade-vision

# geometric verification (pulls in OpenCV)
uv add "saccade-vision[geometry]"
```

Not on PyPI until M6.

## Development

```bash
uv sync --all-extras --dev
uv run pytest
uv run mypy src
```

Two rules are enforced by `tests/test_architecture.py` rather than by review:
the engine may not import domain packages (MediaPipe, YOLO and friends —
domain capability arrives through `register_tool()`), and the pure logic
modules may not touch the filesystem. Both are AST scans, and both fail the
build. The reasoning is in [CLAUDE.md](CLAUDE.md).

## Roadmap

- [x] **M0** — skeleton, public types, architecture guard
- [x] **M1** — Perceive-Verify loop, three visual actions, response cache, benchmark runner
- [ ] **M2** — measurement tools so the verifier can do its job, full BlindTest across models, ablations for which strategies actually help
- [ ] **M3** — Sandlot Baseball: pose estimation, metrics, run-to-run consistency
- [ ] **M4** — active vision applied to occluded and motion-blurred frames
- [ ] **M5** — retrieval-backed interpretation
- [ ] **M6** — frontend, repo split, PyPI release

## Credits

The BlindTest benchmark is by
[Rahmanzadehgervi et al.](https://github.com/anguyen8/vision-llms-are-blind).
Baseball metric selection draws on findings from Driveline's OpenBiomechanics
Project and [baseball-cv](https://github.com/yasumorishima/baseball-cv)
(CC BY-NC-SA 4.0) — the knowledge of which measurements matter, not the code.

## License

MIT
