"""Internal: letting the model decide which tools to run.

The default loop runs every registered tool on every step. That is defensible
when the tools were written for the question — on BlindTest each task has one
tool that solves exactly it, and running it unconditionally reaches 98%. It
stops being defensible the moment an application registers a dozen: most will
not apply, several will return confident numbers about the wrong thing, and a
measurement that answers a different question is worse than no measurement,
because the verifier will treat it as grounds to overrule the model.

So this module asks. The model is shown the tools by name and description and
picks the ones it wants; whatever it names is run, and nothing else. What it
cannot do is decide what the answer is — the chosen tool still measures, and
``is_measurement`` still governs whether that measurement may overrule
anything. Choosing an instrument and reading it are different jobs, and only
the first is delegated here.

Failure is expected and is not an error. A smaller model may name a tool that
does not exist, or reply with prose instead of a list. Both fall back to
running everything, which is the previous behaviour — a bad chooser degrades
to the old loop rather than to no measurement at all.

Rule 3: no I/O here. The VLM is injected, and images arrive in memory.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from PIL.Image import Image

from saccade.ports import VLMPort
from saccade.tools import Tool

__all__ = ["CHOOSE_PROMPT", "ToolChoice", "choose_tools"]

logger = logging.getLogger("saccade")

# Asking for names on one line is deliberate. Structured output would be
# stricter, but it is not available on every provider this has to run against,
# and a model that cannot follow "one line, comma-separated" is unlikely to
# have chosen well anyway — the fallback catches it either way.
CHOOSE_PROMPT = """You can run measurement tools on this image before answering.

Question: {question}

Available tools:
{catalogue}

Which tools would measure something that helps answer the question? A tool
that measures the wrong thing is worse than no tool, so name only those that
apply. Reply with tool names separated by commas, or NONE.
"""

# "NONE" has to survive a model that writes "none." or "None of them".
_NONE = re.compile(r"^\W*none\b", re.IGNORECASE)


@dataclass(frozen=True)
class ToolChoice:
    """Which tools to run this step, and how that was decided.

    ``fallback`` records that the reply could not be read, so a benchmark can
    separate "the model chose everything" from "the model was ignored". The
    two produce identical tool runs and mean opposite things.
    """

    tools: list[Tool]
    reason: str
    fallback: bool = False


def catalogue(tools: list[Tool]) -> str:
    """The tool list as the model sees it."""
    return "\n".join(f"- {tool.name}: {tool.description}" for tool in tools)


async def choose_tools(
    vlm: VLMPort,
    images: list[Image],
    question: str,
    tools: list[Tool],
) -> ToolChoice:
    """Ask the model which of ``tools`` to run.

    Args:
        vlm: The model doing the choosing.
        images: The current view — the choice depends on what is visible, not
            only on the question.
        question: What the investigation is trying to answer.
        tools: Everything registered. An empty list short-circuits.

    Returns:
        The tools to run. On any failure — an unreadable reply, a named tool
        that does not exist, a network error — every tool is returned with
        ``fallback=True``, which is the behaviour the loop had before this
        module existed.
    """
    if not tools:
        return ToolChoice(tools=[], reason="no tools registered")

    prompt = CHOOSE_PROMPT.format(question=question, catalogue=catalogue(tools))

    try:
        response = await vlm.ask(images, prompt)
    except Exception as exc:  # a failed choice must not end the investigation
        logger.debug("tool choice failed (%s); running everything", exc)
        return ToolChoice(
            tools=list(tools),
            reason=f"could not ask which tools to run ({type(exc).__name__})",
            fallback=True,
        )

    return _read_choice(response.text, tools)


def _read_choice(reply: str, tools: list[Tool]) -> ToolChoice:
    """Turn a reply into a tool list.

    Matching is on names appearing anywhere in the reply rather than on exact
    parsing, because models wrap the answer in prose however they like. The
    risk that inverts this — a tool named as an example of what *not* to run —
    needs a model articulate enough to say so, and those follow the format.
    """
    if _NONE.match(reply.strip()):
        return ToolChoice(tools=[], reason="model saw nothing worth measuring")

    lowered = reply.lower()
    chosen = [tool for tool in tools if tool.name.lower() in lowered]

    if not chosen:
        # The model said something, but named nothing that exists. Treat it as
        # a failed choice rather than as "no tools": a hallucinated name is
        # evidence the model wanted to measure, not that it did not.
        return ToolChoice(
            tools=list(tools),
            reason=f"named no known tool ({reply.strip()[:60]!r}); running everything",
            fallback=True,
        )

    names = ", ".join(tool.name for tool in chosen)
    return ToolChoice(tools=chosen, reason=f"model chose {names}")
