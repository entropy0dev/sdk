"""
Entropy0 trust gate for LangGraph

Pipeline: Search -> Trust gate -> Fetch approved -> Synthesize

Every URL returned by web search passes through the Entropy0 /v1/decide
endpoint before content enters the model context. Trust metadata is attached
to each step so it appears in LangSmith traces.

Agents that learn from traces need trustworthy trace data.
Entropy0 gates what enters the trace before the model ever sees it.

Setup:
    pip install -r requirements.txt

    export ENTROPY0_API_KEY=sk_ent0_...
    export ANTHROPIC_API_KEY=sk-ant-...
    export TAVILY_API_KEY=tvly-...
    export LANGCHAIN_API_KEY=ls__...       # optional — enables LangSmith tracing
    export LANGCHAIN_TRACING_V2=true       # optional

Run:
    python agent.py "what are the risks of prompt injection in LLM agents?"
"""

import os
import sys
import httpx
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langsmith import traceable

ENTROPY0_BASE = os.getenv("ENTROPY0_API_URL", "https://entropy0.ai/api")
ENTROPY0_KEY  = os.environ["ENTROPY0_API_KEY"]


# ── State ─────────────────────────────────────────────────────────────────────

class ResearchState(TypedDict):
    query:     str
    urls:      list[str]
    approved:  list[dict]   # {url, action, trust, threat, deviation, reason_codes}
    sandboxed: list[dict]
    denied:    list[dict]
    content:   list[str]
    answer:    str


# ── Entropy0 ──────────────────────────────────────────────────────────────────

def check_url(url: str) -> dict:
    """Call Entropy0 /v1/decide for a single URL. Never throws."""
    try:
        r = httpx.post(
            f"{ENTROPY0_BASE}/v1/decide",
            headers={"X-API-Key": ENTROPY0_KEY},
            json={
                "target":      {"type": "url", "value": url},
                "interaction": {"kind": "fetch", "mode": "read_only", "sensitivity": "medium"},
                "policy":      {"profile": "balanced"},
                "options":     {"include_evidence": False},
            },
            timeout=10,
        )
        if not r.is_success:
            return {"url": url, "action": "sandbox", "error": f"HTTP {r.status_code}"}
        d = r.json()
        return {
            "url":          url,
            "action":       d["decision"]["recommended_action"],
            "trust":        d["scores"]["trust"]["score"],
            "threat":       d["scores"]["threat"]["score"],
            "deviation":    d["scores"]["deviation"]["score"],
            "reason_codes": d["decision"]["reason_codes"],
        }
    except Exception as e:
        return {"url": url, "action": "sandbox", "error": str(e)[:120]}


# ── Nodes ─────────────────────────────────────────────────────────────────────

@traceable(name="search")
def search_node(state: ResearchState) -> dict:
    tool   = TavilySearchResults(max_results=6)
    result = tool.invoke(state["query"])
    urls   = [r["url"] for r in result if "url" in r]
    return {"urls": urls}


@traceable(name="entropy0_trust_gate")
def trust_gate_node(state: ResearchState) -> dict:
    """
    Evaluate every candidate URL through Entropy0.
    Approved sources proceed to fetch. Sandboxed and denied are logged and skipped.
    Trust metadata is returned on state so it appears in the LangSmith trace.
    """
    approved  = []
    sandboxed = []
    denied    = []

    for url in state["urls"]:
        ev     = check_url(url)
        action = ev.get("action", "sandbox")
        if action in ("proceed", "proceed_with_caution"):
            approved.append(ev)
        elif action == "deny":
            denied.append(ev)
        else:
            sandboxed.append(ev)

    print(f"\n[entropy0] {len(approved)} approved / {len(sandboxed)} sandboxed / {len(denied)} denied")
    for ev in denied + sandboxed:
        label = "DENY" if ev["action"] == "deny" else "SANDBOX"
        print(f"  {label}  {ev['url']}")
        print(f"         reason_codes={ev.get('reason_codes', [])} error={ev.get('error', '')}")

    return {"approved": approved, "sandboxed": sandboxed, "denied": denied}


@traceable(name="fetch_content")
def fetch_node(state: ResearchState) -> dict:
    content = []
    for ev in state["approved"]:
        try:
            r = httpx.get(
                ev["url"], timeout=8, follow_redirects=True,
                headers={"User-Agent": "entropy0-research-agent/1.0"},
            )
            if r.is_success:
                text = r.text[:2500].strip()
                content.append(
                    f"[Source: {ev['url']}]\n"
                    f"[Entropy0: trust={ev['trust']} threat={ev['threat']} deviation={ev['deviation']} action={ev['action']}]\n"
                    f"{text}"
                )
        except Exception:
            pass
    return {"content": content}


@traceable(name="synthesize")
def synthesize_node(state: ResearchState) -> dict:
    llm = ChatAnthropic(model="claude-sonnet-4-6", max_tokens=1024)

    gate_summary = (
        f"{len(state['approved'])} sources cleared the Entropy0 trust gate. "
        f"{len(state['sandboxed'])} were sandboxed and {len(state['denied'])} were denied."
    )
    context = "\n\n---\n\n".join(state["content"]) or "No approved content was retrieved."

    response = llm.invoke([{
        "role": "user",
        "content": (
            f"Research question: {state['query']}\n\n"
            f"Trust gate summary: {gate_summary}\n\n"
            f"Content from approved sources only:\n\n{context}\n\n"
            "Answer the research question based solely on the approved sources above. "
            "If a source was sandboxed or denied, do not speculate about its content."
        ),
    }])
    return {"answer": response.content}


# ── Graph ─────────────────────────────────────────────────────────────────────

def build_graph():
    g = StateGraph(ResearchState)
    g.add_node("search",     search_node)
    g.add_node("trust_gate", trust_gate_node)
    g.add_node("fetch",      fetch_node)
    g.add_node("synthesize", synthesize_node)

    g.set_entry_point("search")
    g.add_edge("search",     "trust_gate")
    g.add_edge("trust_gate", "fetch")
    g.add_edge("fetch",      "synthesize")
    g.add_edge("synthesize", END)

    return g.compile()


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) or "what are the risks of prompt injection in LLM agents?"
    print(f"Query: {query}\n")

    graph  = build_graph()
    result = graph.invoke({
        "query":     query,
        "urls":      [],
        "approved":  [],
        "sandboxed": [],
        "denied":    [],
        "content":   [],
        "answer":    "",
    })

    print("\n" + "=" * 60)
    print("ANSWER")
    print("=" * 60)
    print(result["answer"])

    if result["denied"] or result["sandboxed"]:
        print("\n" + "=" * 60)
        print("BLOCKED / SANDBOXED SOURCES")
        print("=" * 60)
        for ev in result["denied"] + result["sandboxed"]:
            print(f"  [{ev['action'].upper()}] {ev['url']}")
            print(f"    reason_codes: {ev.get('reason_codes', [])}")
