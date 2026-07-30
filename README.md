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

**How far, so far: not very.** The measurements below say the extra fixations
buy nothing on BlindTest, while the refereeing buys a great deal on a weak
model and nothing on a strong one. Read that section before the pitch above
convinces you of anything.

## What the measurements say

150 items per run, sampled across the full difficulty range rather than the
easy end. Every figure below traces to a JSON file in
[`benchmarks/blindtest/results/`](benchmarks/blindtest/results/) with
per-item answers.

**Touching Circles**

| | GPT-5.4 | GPT-4.1 |
|---|---|---|
| One look — what a VLM does today | 90.7% | 78.0% |
| Looking several times | 93.3% | 77.3% |
| Looking several times, checked against geometry | 94.0% | 89.3% |
| **Geometry alone, no model at all** | **98.0%** | **98.0%** |

Three things fall out of that table, and only one of them is comfortable.

**Looking again does not help.** +2.7% on one model, −0.7% on the other,
neither distinguishable from noise (p = 0.39, p = 1.00). Same on the other
tasks measured. Whatever this library is worth, it is not worth it for the
extra fixations.

**A referee helps a weak model and not a strong one.** GPT-4.1 gained 12.0%
from being checked against geometry (p = 0.00001, 18 items fixed, none
broken). GPT-5.4 gained 0.7% (p = 1.00). The tool overruled GPT-4.1 fifty-one
times and GPT-5.4 twelve times — a model that is already right leaves a
referee nothing to do.

**On this task the model is the weak link.** Geometry alone beats every
configuration involving a model, and on GPT-4.1 it beats the full loop by
8.7% (p = 0.001). Asked whether two circles touch, the honest answer is to
measure the circles and not ask a language model at all.

### The benchmark has run out of room

| Task | Model | One look | Headroom |
|---|---|---|---|
| Counting Grid — Blank | GPT-5.4 | 100.0% | 0.0% |
| Line Plot Intersections | GPT-5.4 | 98.7% | 1.3% |
| Touching Circles | GPT-5.4 | 90.7% | 9.3% |
| Touching Circles | GPT-4.1 | 78.0% | 22.0% |

BlindTest's headline number is 58.07%, and the task the paper singles out —
counting line intersections — was 56.84%, near chance. GPT-5.4 scores 98.7%
on it cold, and 100% on counting rows and columns in a 2000×2000 grid. The
models have caught up with the benchmark since it was published.

That closes off the experiment from both ends. Where geometry can settle a
task, the tool wins alone and the loop is overhead. Where it cannot, the
current models are already at ceiling and nothing can show up. Choosing this
benchmark as the main evidence was a mistake on my part; it can no longer
test the idea it was chosen to test.

## Where this is now

**M2 done, and it did not go the way the pitch implies.** The loop runs, the
verifier verifies, both measurement tools are validated against the dataset's
own geometry (99.2% and 99.6% agreement), and the ablation is clean enough to
say that the interesting half of the idea — measurement over guessing — holds,
while the half the name refers to does not show up here.

Next is the case this benchmark cannot provide: video, where the detector
itself is unreliable. MediaPipe misplaces joints through occlusion and motion
blur, so "the tool said so" stops being sufficient and something has to decide
which frames to distrust. That is where looking again should earn its keep, if
it earns it anywhere.

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
- [x] **M2** — measurement tools and the ablation that settled what works
  - [x] circle geometry (99.2%) and line crossings (99.6%), both validated against the dataset
  - [x] four-way ablation including a no-model control, two models, three tasks
  - [x] the finding: refereeing helps a weak model, extra looks help nothing,
        and this benchmark is out of headroom
- [ ] **M3** — video, where the detector is unreliable and the tool cannot simply be trusted
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
