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

## First results

On BlindTest's Touching Circles — 150 items, sampled across the full range
of difficulty rather than the easy end:

| | GPT-5.4 | GPT-4.1 |
|---|---|---|
| One look (what a VLM does today) | 90.7% | 78.0% |
| Looking several times | 93.3% | 77.3% |
| Looking several times, checked against geometry | **100.0%** | **89.3%** |

The middle row is the interesting one. Letting the model look again and
again is not what helps: +2.7% on one model and −0.7% on the other, neither
distinguishable from noise (McNemar p = 0.39 and p = 1.00). What helps is
giving it something that can actually measure — +6.7% and +12.0% on top,
p = 0.002 and p = 0.00001, and across both models that step fixed 28 items
while breaking none.

That is worth stating plainly, because it is not the obvious result. The
gain does not come from more attention. It comes from attention plus a
referee: the tool overruled the model 17 times on GPT-5.4 and 51 times on
GPT-4.1, and every one of those was the model being talked out of a wrong
answer by a number.

One task of seven, two models, one sample size. The remaining tasks and a
wider model sweep are still to come, and the figures above are worth exactly
as much as that caveat implies. Every run is in
[`benchmarks/blindtest/results/`](benchmarks/blindtest/results/) with
per-item answers, so none of this has to be taken on trust.

## Where this is now

**M2, partway.** The loop runs, the verifier verifies, and the first
measurement tool is in. Circle detection is validated against the dataset's
own geometry rather than by eye — it agrees with the published answer on
99.2% of 1190 items, which is what makes it fit to referee a model scoring
90%.

Still to do: measurement tools for the other tasks, the full seven-task
sweep, and more models.

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
- [ ] **M2** — measurement tools, full BlindTest across models, ablations
  - [x] circle geometry, verified against the dataset at 99.2%
  - [x] three-way ablation on Touching Circles, two models
  - [ ] line intersections, then the remaining five tasks
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
