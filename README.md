# Saccade

Vision models guess at things a ruler could settle. Saccade wires a measurement
tool into the loop and lets it overrule the model.

```python
agent = ActiveVisionAgent("openrouter:qwen/qwen3-vl-8b-instruct")
agent.register_tool(circle_geometry)

result = agent.investigate(image, "Are the two circles touching?")
result.answer  # the geometry's verdict, if it contradicted the model
result.evidence_chain  # every look, every number, and why
```

An 8B open-weight model goes from 65% to 98% on that question. GPT-5.4 scores
91% without tools.

## Why

Most agent frameworks treat a tool as an extension of the model and believe
what comes back. Here a tool is the model's *referee*: when a computed result
contradicts what the model said, the measurement becomes the answer.

That inverts who is trusted, and it makes the choice of model matter much
less. Across four models on two tasks, bare accuracy spans 60 percentage
points; with a referee, 1.3.

| model | bare | with a referee |
|---|---|---|
| GPT-4.1 (lines) | 38.0% | 99.3% |
| Qwen3-VL-8B (circles) | 65.3% | 98.0% |
| GPT-5.4 (circles) | 90.7% | 98.0% |
| GPT-5.4 (lines) | 98.7% | 99.3% |

The gain runs inverse to the model, monotonically. If a tool can settle the
question, a small open model you host yourself gets there too.

## Install

```bash
uv add saccade-vision                    # not on PyPI until M6
uv add "saccade-vision[geometry]"        # geometric primitives (pulls OpenCV)
```

## Writing a tool

Saccade ships no domain capability and never will — the engine may not import
MediaPipe, YOLO or anything like them, and a test enforces it. You bring the
detector; the engine handles the loop, the refereeing and the evidence chain.

```python
from pydantic import BaseModel

from saccade import ActiveVisionAgent, Tool, ToolResult
from saccade.geometry import angle_between


class NoParams(BaseModel):
    """Measures the view it is handed, so it takes no arguments."""


def elbow_tool(joints) -> Tool:
    def run(image, viewport) -> ToolResult:
        degrees = angle_between(joints["shoulder"], joints["elbow"], joints["wrist"])
        return ToolResult(
            value={"extended": degrees > 160, "degrees": degrees},
            is_measurement=True,  # only this may overrule the model
            answer_key="extended",  # which key is the verdict, vs. context
        )

    return Tool(
        name="elbow_extension",
        description="Measure whether the pitching elbow is extended",
        fn=run,
        params_schema=NoParams,
    )
```

Three fields carry the contract:

- **`description`** — not a comment. With `choose_tools=True` the model reads
  it to decide whether to reach for your tool, so write it for the model.
- **`is_measurement`** — the gate. `True` means this result may contradict the
  VLM. A tool that is itself a model must say `False`; one blind witness
  cannot vouch for another.
- **`answer_key`** — which entry of `value` is the verdict. Everything else
  reaches the evidence chain without being judged, so a line counter
  reporting 1 crossing across 300 shared columns answers 1.

Closed-form primitives to build on: `distance`, `angle_between`, `bearing`,
`speed`, `centroid`, `bbox_iou`, `point_to_segment_distance`, `smooth`,
`circles_overlap`, `segments_intersect`, `count_line_intersections`.

## Letting the model pick the tool

Off by default, because running every tool is fine while every tool applies.
It stops being fine as the toolbox grows: a tool that measures the wrong
thing produces a confident number the referee will act on.

```python
agent = ActiveVisionAgent(model, choose_tools=True)
```

Qwen3-VL-8B picked the applicable tool from a 20-tool catalogue on 40 of 40
items, including three decoys built to be tempting and wrong. A failed choice
runs everything, so a bad chooser degrades to the old behaviour rather than
to no measurement.

## How the loop runs

```
1. Plan  ─▶  2. Act  ─▶  3. Observe  ─▶  4. Choose  ─▶  5. Verify
 where to    crop/zoom     ask the VLM     which tools    confront the
 look next                                 to run         claim with it
                                                                │
                                    6. Record ◀────────────────┘
```

Runs until verified confidence clears the threshold or the step budget runs
out. Running out is a normal result, not an error — you get `converged=False`
and the evidence chain that got you there.

## What does not work

Extra fixations. Measured three ways, null in all three (p = 0.39, 1.00,
0.25). The name refers to a mechanism the data does not support; what works
is the refereeing and the tool choice. Details and per-item results are in
[`docs/findings.md`](docs/findings.md).

## Development

```bash
uv sync --all-packages --all-extras --dev
uv run pytest
uv run mypy src apps/sandlot-baseball/src
```

`--all-packages` is not optional: this is a uv workspace, and without it the
applications under `apps/` are left uninstalled along with their detectors.

Two rules are enforced by `tests/test_architecture.py` rather than by review:
the engine may not import domain packages, and the pure logic modules may not
touch the filesystem. Both are AST scans and both fail the build. Reasoning in
[CLAUDE.md](CLAUDE.md).

## Roadmap

- [x] **M0–M2** — the loop, measurement tools, and the ablation that settled
      what works
- [ ] **M3** — video, where the detector itself is unreliable
- [ ] **M4** — active vision on occluded and motion-blurred frames
- [ ] **M5** — retrieval-backed interpretation
- [ ] **M6** — PyPI release

## Credits

BlindTest benchmark by
[Rahmanzadehgervi et al.](https://github.com/anguyen8/vision-llms-are-blind).
Baseball metric selection draws on Driveline's OpenBiomechanics Project and
[baseball-cv](https://github.com/yasumorishima/baseball-cv) (CC BY-NC-SA 4.0)
— the knowledge of which measurements matter, not the code.

Built with [Claude Code](https://claude.com/claude-code).

## License

MIT
