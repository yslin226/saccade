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

**Half of that pitch is wrong, and the measurements below say which half.**
The extra fixations buy nothing. What does work is the other half — the model
deciding *which instrument to reach for*, and being overruled by what that
instrument reads.

## What the measurements say

Every figure below traces to a JSON file in
[`benchmarks/blindtest/results/`](benchmarks/blindtest/results/) with per-item
answers. Runs marked as such were made with the response cache disabled, so
the model was genuinely asked rather than replayed.

### A small open model, given tools it must choose between

100 items per task, `qwen/qwen3-vl-8b-instruct` — 8B parameters, Apache 2.0,
runs on a consumer GPU. The toolbox holds five tools: the one that applies,
the one for the other task, and three decoys that measure something true
about the image which does not answer the question.

| | one look | toolbox, model chooses |
|---|---|---|
| Touching Circles | 65.3% | **98.0%** |
| Line Plot Intersections | 86.7% | **100.0%** |

GPT-5.4 scores 90.7% and 98.7% on the same tasks with one look. The 8B model
plus a toolbox it has to navigate beats it on both.

It picked the applicable tool alone on 28 of 30 circle items and 30 of 30
line items, with no fallbacks. Two of the decoys were never chosen at all.

### Choosing does not degrade with the size of the toolbox

Five tools is a pair with distractors, not a toolbox. An application registers
a dozen or more, so the catalogue was padded with plausible instruments for
questions nobody asked — edge density, stroke width, convexity — and the
choice measured again at each size.

| toolbox | Touching Circles | Line Intersections | fallbacks |
|---|---|---|---|
| 5 tools | 18/20 | 20/20 | 0 |
| 10 tools | **20/20** | 20/20 | 0 |
| 20 tools | **20/20** | 20/20 | 0 |

80 choices, no fallbacks, and the two lapses are at the *smallest* size. Both
took the applicable tool *and* the sharp decoy alongside it; at 10 and 20 that
never happened. Why a bigger catalogue reads as easier is not something this
measures — one reading is that a short list makes everything on it look
relevant, while a long one forces the descriptions to actually be read. That
is a hypothesis, not a finding.

### The gain runs inverse to the model

Same tasks, one tool given rather than chosen, across four models:

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
with a referee they span 1.3. **Which model you use stops mattering** — which
is the whole argument for running an open one you can host yourself.

### Looking again still does not help

Measured three separate ways, all null. On the two tasks with a referee, extra
fixations gained +2.7% and −0.7% (p = 0.39, p = 1.00). On tasks without one,
`grid_blank` fixed 3 items and broke 0 over 70 paired (McNemar p = 0.25) and
`olympic_circles` changed nothing whatsoever (p = 1.00) while spending 3.5×
the tokens.

The distinction that took three failed experiments to see: those measured a
*hardcoded rule* deciding which corner to magnify. The model deciding which
*instrument* to reach for is a different question, and that one it can answer.

### What the decoys found

A benchmark where every tool was written for the task it is offered on cannot
tell whether a model is choosing or simply accepting. So the toolbox got three
tools that measure something true and irrelevant — and one of them,
`bounding_box_overlap`, contradicts ground truth on 70% of circle items while
sounding directly applicable.

Adding them cost 6 points, and the cause was a defect in the engine rather
than a failure to choose. A bare "No" names no subject, and the verifier was
checking it against every boolean measurement at once — including one
answering a different question. Both statements were true; the wrong one won.

| | before | after |
|---|---|---|
| overall | 93.0% | **98.0%** |
| items where a tool spoke | 86.0% | **100.0%** |

37 conflicts, 37 correct. That defect is invisible with a single-tool toolbox,
which is the argument for building a benchmark that can embarrass its own
thesis.

## Where this is now

**M2 done, and it took two corrections to get an honest reading.** The first:
a measurement that contradicted the model only lowered confidence, and the
model's contradicted answer was still returned — the referee could object but
not overrule. Items where a tool objected scored 1.9% against 97.9% where none
did, and the geometry had the right answer in every one of them.

The second is the decoy defect above. Neither was caught by the 577 tests that
existed; both were only visible against real data, and both are pinned by
tests now.

What holds: when a question is measurable, an 8B open-weight model navigating
a mixed toolbox matches a frontier model. What is unproven: whether any of
this helps when the question is *not* measurable. Seven of BlindTest's nine
tasks have no computable referee, and on those there is currently no tool to
choose — which is a gap in the tooling, not a finding about the loop.

Next is video, where the detector itself is unreliable. MediaPipe misplaces
joints through occlusion and motion blur, so "the tool said so" stops being
sufficient — and the question of *why* two detectors disagree (blur? occlusion?
lost track?) is one no measurement can answer and a VLM might.

## How it works

```
             ┌────────────────────────────────────────────────────┐
             │                                                    │
             ▼                                                    │
    1. Plan  ─▶  2. Act  ─▶  3. Observe  ─▶  4. Choose  ─▶  5. Verify
   where to     crop/zoom/    ask the VLM     which tools    confront the
   look next    annotate      what it sees    to run         claim with it
                                                                  │
                                        6. Record ◀───────────────┘
                                  viewport, numbers, reasoning
```

The loop runs until verified confidence clears the threshold or the step
budget runs out. Running out is a normal result, not an error — you get back
`converged=False` and the full evidence chain that got you there.

**Step 4 is opt-in** (`choose_tools=True`) and off by default, because running
every tool is right while every tool was written for the question. It stops
being right as the toolbox grows: a tool that measures the wrong thing does
not merely waste a call, it produces a confident number the verifier will use
to overrule the model. A failed choice — unreadable reply, hallucinated tool
name, network error — runs everything, so a bad chooser degrades to the old
loop rather than to no measurement.

**Step 5 is what makes this not a conventional ReAct agent.** Most agents
treat tools as an extension of the model and believe what comes back. Here a
tool is the model's referee: only results flagged `is_measurement=True` may
contradict what the VLM said, and when one does, *the measurement becomes the
answer*. A tool that is itself a VLM does not qualify — one blind witness
cannot vouch for another.

Choosing an instrument and reading it stay separate jobs. Only the first is
delegated to the model.

## Install

```bash
uv add saccade-vision

# geometric verification (pulls in OpenCV)
uv add "saccade-vision[geometry]"
```

Not on PyPI until M6.

## Using it

The engine ships no domain capability and never will — rule 2 forbids it from
importing MediaPipe, YOLO or anything like them. What you get is the loop, the
refereeing, the evidence chain, and closed-form geometric primitives to build
measurements out of. Your detector arrives through `register_tool()`.

```python
from pydantic import BaseModel
from saccade import ActiveVisionAgent, Tool, ToolResult
from saccade.geometry import angle_between


class NoParams(BaseModel):
    """This tool measures the view it is handed."""


def elbow_tool(joints) -> Tool:
    def run(image, viewport) -> ToolResult:
        degrees = angle_between(joints["shoulder"], joints["elbow"], joints["wrist"])
        return ToolResult(
            value={"method": "joint_angle", "extended": degrees > 160, "degrees": degrees},
            is_measurement=True,  # only this may overrule the model
            answer_key="extended",  # which key holds the verdict, vs. context
        )

    return Tool(
        name="elbow_extension",
        description="Measure whether the pitching elbow is extended",
        fn=run,
        params_schema=NoParams,
    )


agent = ActiveVisionAgent("openrouter:qwen/qwen3-vl-8b-instruct", choose_tools=True)
agent.register_tool(elbow_tool(joints))

result = agent.investigate(frame, "Is the elbow extended at release?")
result.answer  # "Yes" — from the measurement, if it contradicted the model
result.evidence_chain  # every look, every number, and why each was taken
```

Three fields carry the contract:

- **`description`** is not a comment. It is the entire basis on which the
  model decides whether to reach for your tool, so write it for the model.
- **`is_measurement`** is the gate. `True` means this result may contradict
  the VLM. A tool that is itself a model must say `False`.
- **`answer_key`** names which entry of `value` is the verdict. Everything
  else is context that reaches the evidence chain without being judged — a
  line counter reporting 1 crossing across 300 shared columns must answer 1.

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
  - [x] five-way ablation including a no-model control, four models, decoy tools
  - [x] the measurement decides the answer, rather than only lowering confidence
  - [x] the model chooses which tools to run, and is measured against decoys
        built to punish choosing badly
  - [x] the finding: extra looks help nothing, but a referee closes a 60-point
        spread between models to 1.3, and an 8B open model navigating a mixed
        toolbox matches a frontier one
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
