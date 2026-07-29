"""Building VLMs for benchmark runs.

Gemini's free tier allows 20 requests per minute, which is not enough to
finish a saccade run over a real sample — each item costs several calls, and
retries stack on top. Supporting a second provider is what makes the
comparison runnable at all, and M2 needs more than one model anyway.
"""

from __future__ import annotations

import os

from saccade.ports import VLMPort
from saccade.vlm.pydantic_ai import PydanticAIVLM

__all__ = ["PROVIDER_RPM", "build_vlm", "default_rpm"]

# Requests per minute to assume when the caller does not say. Gemini's free
# tier is the measured 20/min; Azure depends on the deployment's quota, so
# the default is conservative rather than optimistic.
PROVIDER_RPM = {
    "google": 20,
    "azure": 60,
    "openai": 60,
}


def build_vlm(model: str) -> VLMPort:
    """Build a VLM from a model string.

    Args:
        model: Either a Pydantic AI model string with a provider prefix
            (``"google:gemini-flash-latest"``), or ``"azure:<deployment>"``
            for an Azure OpenAI deployment.

            Azure names a *deployment*, not a model — the name is whatever
            was chosen when the deployment was created, which is why it
            cannot be inferred.

    Raises:
        RuntimeError: If Azure is requested without the required environment
            variables. Failing here names the missing variable, rather than
            surfacing as an opaque 401 partway through a run.
    """
    if not model.startswith("azure:"):
        return PydanticAIVLM(model)

    # "azure:" with nothing after it falls back to the configured default,
    # so a single-deployment setup need not repeat the name every run.
    deployment = model.split(":", 1)[1] or os.environ.get("AZURE_OPENAI_DEPLOYMENT", "")
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    api_key = os.environ.get("AZURE_OPENAI_API_KEY")

    # No default for the API version. A guessed one is worse than none: a
    # version the resource does not serve returns "404 Resource not found",
    # which reads as a missing deployment and sends you hunting for the
    # wrong bug. Demanding the value names the real problem immediately.
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION")

    missing = [
        name
        for name, value in (
            ("AZURE_OPENAI_ENDPOINT", endpoint),
            ("AZURE_OPENAI_API_KEY", api_key),
            ("AZURE_OPENAI_DEPLOYMENT", deployment),
            ("AZURE_OPENAI_API_VERSION", api_version),
        )
        if not value
    ]
    if missing:
        raise RuntimeError(
            f"Azure model {model!r} needs {', '.join(missing)}. Put them in .env — "
            f"see .env.example."
        )

    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.azure import AzureProvider

    return PydanticAIVLM(
        OpenAIChatModel(
            deployment,
            provider=AzureProvider(
                azure_endpoint=endpoint,
                api_version=api_version,
                api_key=api_key,
            ),
        )
    )


def default_rpm(model: str) -> int:
    """The rate limit to assume for a model string."""
    provider = model.split(":", 1)[0] if ":" in model else ""
    return PROVIDER_RPM.get(provider, 20)
