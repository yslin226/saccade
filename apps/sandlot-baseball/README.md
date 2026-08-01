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

## Status

M3, in progress. The skeleton is here; the metrics are not yet.

```bash
uv run sandlot analyze <video>
uv run sandlot compare <session-a> <session-b>
```

Sessions are stored under `~/.sandlot/sessions/` by default; `--data-dir`
overrides it.

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
