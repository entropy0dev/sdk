"""
Entropy0 LangChain tool — trust-gate any domain or URL before your agent uses it.
"""
from __future__ import annotations

from typing import Any, Optional, Type

import httpx
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


class Entropy0Input(BaseModel):
    target: str = Field(
        description=(
            "The domain name or full URL to evaluate. "
            "Examples: 'example.com', 'https://example.com/page'. "
            "Always check before fetching content from an external source."
        )
    )


class Entropy0Tool(BaseTool):
    """
    Trust-gate for LangChain agents.

    Calls the Entropy0 Trust Control Plane (POST /v1/decide) to evaluate a
    domain or URL before your agent fetches content from it. Returns a
    recommended action the agent can reason about directly.

    Usage::

        from entropy0_langchain import Entropy0Tool

        tools = [Entropy0Tool(api_key="sk_ent0_xxxx")]
        agent = initialize_agent(tools, llm, agent=AgentType.OPENAI_FUNCTIONS)

    The agent will receive responses like::

        "Trust evaluation for evil-phish.com: recommended_action=deny
        (confidence=91%), uncertainty=low, signals=NEWLY_REGISTERED_DOMAIN,
        BRAND_MISMATCH, CERTIFICATE_ANOMALY."
    """

    name: str = "entropy0_trust_check"
    description: str = (
        "Check whether a domain or URL is safe to interact with before fetching "
        "content from it. Use this before visiting any external URL in a RAG "
        "pipeline, web browsing task, or data retrieval step. "
        "Returns one of: proceed, proceed_with_caution, sandbox, "
        "escalate_to_human, or deny — with a confidence score and reason codes."
    )
    args_schema: Type[BaseModel] = Entropy0Input
    return_direct: bool = False

    # ── Config fields (set at construction time) ───────────────────────────
    api_key: str = Field(..., description="Your Entropy0 API key (sk_ent0_xxxx)")
    base_url: str = Field(
        default="https://entropy0.ai/api",
        description="Override the Entropy0 API base URL.",
    )
    policy: str = Field(
        default="balanced",
        description="Policy profile: open | balanced | strict | critical",
    )
    interaction_kind: str = Field(
        default="fetch",
        description="How the agent will interact: navigate | fetch | enrich | download_file | submit_credentials | initiate_payment",
    )
    interaction_mode: str = Field(
        default="read_only",
        description="Privilege level: read_only | transactional | privileged",
    )
    interaction_sensitivity: str = Field(
        default="medium",
        description="Data sensitivity: low | medium | high | critical",
    )
    timeout: float = Field(default=10.0, description="HTTP timeout in seconds.")

    # ── Internal helper ────────────────────────────────────────────────────

    def _build_payload(self, target: str) -> dict[str, Any]:
        target_type = "url" if target.startswith(("http://", "https://")) else "domain"
        return {
            "target": {"type": target_type, "value": target},
            "interaction": {
                "kind":        self.interaction_kind,
                "mode":        self.interaction_mode,
                "sensitivity": self.interaction_sensitivity,
            },
            "policy": {"profile": self.policy},
        }

    @staticmethod
    def _format_result(target: str, data: dict[str, Any]) -> str:
        action     = data["decision"]["recommended_action"]
        confidence = data["decision"]["action_confidence"]
        reasons    = data["decision"].get("reason_codes", [])
        uncertainty = data["uncertainty"]["state"]
        valid_until = data["validity"]["valid_until"]

        reason_str = ", ".join(reasons[:4]) if reasons else "no specific flags raised"

        return (
            f"Trust evaluation for {target}: "
            f"recommended_action={action} (confidence={confidence:.0%}), "
            f"uncertainty={uncertainty}, "
            f"signals=[{reason_str}]. "
            f"Decision valid until {valid_until}."
        )

    # ── Sync execution ────────────────────────────────────────────────────

    def _run(self, target: str, **kwargs: Any) -> str:
        payload = self._build_payload(target)
        response = httpx.post(
            f"{self.base_url}/v1/decide",
            headers={"X-API-Key": self.api_key, "Content-Type": "application/json"},
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return self._format_result(target, response.json())

    # ── Async execution ───────────────────────────────────────────────────

    async def _arun(self, target: str, **kwargs: Any) -> str:
        payload = self._build_payload(target)
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/v1/decide",
                headers={"X-API-Key": self.api_key, "Content-Type": "application/json"},
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
        return self._format_result(target, response.json())
