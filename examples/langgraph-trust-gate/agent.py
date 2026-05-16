"""
Entropy0 trust gate for LangGraph

Pipeline: Search -> Trust gate -> Extract approved -> Synthesize

Every URL returned by web search is evaluated through the Entropy0 /v1/decide
endpoint before content enters the model context. Approved URLs are then
extracted via Tavily Extract (clean markdown, not raw HTML). Sources that
return sandbox or deny are logged and skipped. Trust metadata is attached
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
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic
from langchain_tavily import TavilySearch, TavilyExtract
from langsmith import traceable

load_dotenv()

ENTROPY0_BASE = os.getenv("ENTROPY0_API_URL", "https://entropy0.ai/api")
ENTROPY0_KEY  = os.environ["ENTROPY0_API_KEY"]

# Known user-generated content platforms — platform trust != content trust
UGC_DOMAINS = {
    "reddit.com", "twitter.com", "x.com", "facebook.com", "youtube.com",
    "github.com", "stackoverflow.com", "quora.com", "medium.com",
    "linkedin.com", "news.ycombinator.com", "lobste.rs",
}

# Reason codes that are trust SIGNALS (positive indicators of domain health).
# When a sandboxed domain's reason_codes are all positive, the sandbox decision
# came from score thresholds (threat/deviation band), not from these signals.
# Showing them as the "sandbox reason" is misleading — derive reason from scores instead.
POSITIVE_TRUST_CODES = {
    "LONG_OPERATIONAL_HISTORY", "STRONG_BRAND_ALIGNMENT", "TRUST_SIGNAL_CONSISTENCY",
    "KNOWN_BENIGN_PATTERN", "APEX_DOMAIN_TRUST_INHERITANCE",
}

def is_ugc(url: str) -> bool:
    return any(d in url for d in UGC_DOMAINS)

def infer_sandbox_reason(trust: float, threat: float, deviation: float) -> list[str]:
    """Derive sandbox reason from scores when API codes are all positive signals."""
    codes = []
    if threat >= 40:
        codes.append("ELEVATED_THREAT_SCORE")
    if deviation >= 40:
        codes.append("ELEVATED_DEVIATION")
    if trust < 70:
        codes.append("INSUFFICIENT_TRUST_BAND")
    return codes or ["SCORE_BASED_GATE"]


# ── State ─────────────────────────────────────────────────────────────────────

class ResearchState(TypedDict):
    query:      str
    urls:       list[str]
    approved:   list[dict]   # passed trust gate
    sandboxed:  list[dict]   # gated but not denied
    denied:     list[dict]   # hard block
    unverified: list[dict]   # could not gather evidence (timeout / extraction fail)
    content:    list[str]
    answer:     str


# ── Entropy0 ──────────────────────────────────────────────────────────────────

def check_url(url: str) -> dict:
    """
    Call Entropy0 /v1/decide for a single URL.

    Returns a dict with:
      action              — proceed / proceed_with_caution / sandbox / deny / unverified
      trust_reason_codes  — positive trust signals from the API
      decision_reason_codes — why the gate decision was made (may differ for UGC)
    Never throws.
    """
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
            return {
                "url": url, "action": "unverified",
                "trust_reason_codes": [], "decision_reason_codes": ["API_ERROR"],
                "error": f"HTTP {r.status_code}",
            }

        d      = r.json()
        action = d["decision"]["recommended_action"]
        trust_codes = d["decision"]["reason_codes"]

        trust  = d["scores"]["trust"]["score"]
        threat = d["scores"]["threat"]["score"]
        dev    = d["scores"]["deviation"]["score"]

        # Separate trust signal from decision rationale.
        # Case 1 — UGC: platform trust ≠ content trust.
        # Case 2 — Sandboxed with only positive codes: the sandbox decision came
        #          from score thresholds, not from these signals — derive from scores.
        if is_ugc(url):
            decision_codes = ["USER_GENERATED_CONTENT", "REQUIRES_CORROBORATION"]
        elif (action in ("sandbox", "escalate_to_human", "deny") and
              trust_codes and
              all(c in POSITIVE_TRUST_CODES for c in trust_codes)):
            decision_codes = infer_sandbox_reason(trust, threat, dev)
        else:
            decision_codes = trust_codes

        return {
            "url":                   url,
            "action":                action,
            "trust":                 trust,
            "threat":                threat,
            "deviation":             dev,
            "trust_reason_codes":    trust_codes,
            "decision_reason_codes": decision_codes,
        }

    except httpx.TimeoutException:
        return {
            "url": url, "action": "unverified",
            "trust_reason_codes": [], "decision_reason_codes": ["FETCH_TIMEOUT"],
            "error": "entropy0 check timed out — defaulting to unverified",
        }
    except Exception as e:
        return {
            "url": url, "action": "unverified",
            "trust_reason_codes": [], "decision_reason_codes": ["CHECK_FAILED"],
            "error": str(e)[:120],
        }


# ── Nodes ─────────────────────────────────────────────────────────────────────

@traceable(name="search")
def search_node(state: ResearchState) -> dict:
    tool   = TavilySearch(max_results=6, search_depth="advanced")
    result = tool.invoke(state["query"])
    items = result.get("results", []) if isinstance(result, dict) else result
    urls  = [r["url"] for r in items if isinstance(r, dict) and "url" in r]
    return {"urls": urls}


@traceable(name="entropy0_trust_gate")
def trust_gate_node(state: ResearchState) -> dict:
    """
    Evaluate every candidate URL through Entropy0.
    Splits URLs into approved / sandboxed / denied / unverified.
    Trust metadata is returned on state so it appears in the LangSmith trace.
    """
    approved   = []
    sandboxed  = []
    denied     = []
    unverified = []

    for url in state["urls"]:
        ev     = check_url(url)
        action = ev.get("action", "unverified")
        if action in ("proceed", "proceed_with_caution"):
            approved.append(ev)
        elif action == "deny":
            denied.append(ev)
        elif action == "unverified":
            unverified.append(ev)
        else:
            sandboxed.append(ev)

    print(
        f"\n[entropy0] {len(approved)} approved / "
        f"{len(sandboxed)} sandboxed / "
        f"{len(denied)} denied / "
        f"{len(unverified)} unverified"
    )
    for ev in sandboxed:
        print(f"  SANDBOX    {ev['url']}")
        trust_codes    = ev.get("trust_reason_codes", [])
        decision_codes = ev.get("decision_reason_codes", [])
        if trust_codes and trust_codes != decision_codes:
            print(f"             trust signals: {trust_codes}")
        print(f"             sandbox reason: {decision_codes}")
    for ev in denied:
        print(f"  DENY       {ev['url']}")
        print(f"             decision: {ev.get('decision_reason_codes', [])}")
    for ev in unverified:
        print(f"  UNVERIFIED {ev['url']}")
        print(f"             reason: {ev.get('error', '')}")

    return {"approved": approved, "sandboxed": sandboxed, "denied": denied, "unverified": unverified}


@traceable(name="extract_content")
def fetch_node(state: ResearchState) -> dict:
    """
    Use Tavily Extract to pull clean markdown from approved URLs.
    Raw httpx.get returns boilerplate HTML — TavilyExtract returns article content.
    """
    if not state["approved"]:
        return {"content": []}

    approved_urls = [ev["url"] for ev in state["approved"]]
    ev_by_url     = {ev["url"]: ev for ev in state["approved"]}

    try:
        extractor = TavilyExtract(extract_depth="advanced", format="markdown", include_images=False)
        result    = extractor.invoke({"urls": approved_urls})
    except Exception as e:
        print(f"[extract] TavilyExtract failed: {e}")
        return {"content": []}

    content = []
    for item in result.get("results", []):
        url  = item.get("url", "")
        text = (item.get("raw_content") or "").strip()
        if not text:
            continue
        ev = ev_by_url.get(url, {})
        content.append(
            f"[Source: {url}]\n"
            f"[Entropy0: trust={ev.get('trust','?')} threat={ev.get('threat','?')} "
            f"deviation={ev.get('deviation','?')} action={ev.get('action','?')}]\n\n"
            f"{text[:3000]}"
        )

    for item in result.get("failed_results", []):
        url = item.get("url", "")
        ev  = ev_by_url.get(url, {})
        ev["action"]               = "unverified"
        ev["decision_reason_codes"] = ["CONTENT_UNAVAILABLE"]
        state["unverified"].append(ev)
        print(f"  UNVERIFIED {url} — extraction failed")

    return {"content": content}


@traceable(name="synthesize")
def synthesize_node(state: ResearchState) -> dict:
    llm = ChatAnthropic(model="claude-sonnet-4-6", max_tokens=1024)

    gate_summary = (
        f"{len(state['approved'])} sources cleared the Entropy0 trust gate "
        f"({len(state['sandboxed'])} sandboxed, {len(state['denied'])} denied, "
        f"{len(state['unverified'])} unverified — could not gather evidence)."
    )

    if state["content"]:
        context = "\n\n---\n\n".join(state["content"])
        evidence_note = ""
    else:
        context      = "No usable content was extracted from approved sources."
        evidence_note = (
            "\n\nEvidence quality note: the trust gate approved sources but the "
            "extraction layer returned insufficient article text. Limit your answer "
            "to what can be verified from source titles and metadata only. "
            "Do not fabricate or infer details not present in the retrieved content."
        )

    response = llm.invoke([{
        "role": "user",
        "content": (
            f"Research question: {state['query']}\n\n"
            f"Trust gate summary: {gate_summary}\n\n"
            f"Content from approved sources only:\n\n{context}"
            f"{evidence_note}\n\n"
            "Answer the research question based solely on the approved sources above. "
            "If extraction returned boilerplate rather than article text, say so clearly "
            "as an evidence quality warning — do not fabricate details."
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
        "query":      query,
        "urls":       [],
        "approved":   [],
        "sandboxed":  [],
        "denied":     [],
        "unverified": [],
        "content":    [],
        "answer":     "",
    })

    print("\n" + "=" * 60)
    print("ANSWER")
    print("=" * 60)
    print(result["answer"])

    blocked = result["sandboxed"] + result["denied"] + result["unverified"]
    if blocked:
        print("\n" + "=" * 60)
        print("GATED / UNVERIFIED SOURCES")
        print("=" * 60)
        for ev in result["sandboxed"]:
            print(f"  [SANDBOX]    {ev['url']}")
            t_codes = ev.get("trust_reason_codes", [])
            d_codes = ev.get("decision_reason_codes", [])
            if t_codes and t_codes != d_codes:
                print(f"               trust signals:  {t_codes}")
            print(f"               sandbox reason: {d_codes}")
        for ev in result["denied"]:
            print(f"  [DENY]       {ev['url']}")
            print(f"               {ev.get('decision_reason_codes', [])}")
        for ev in result["unverified"]:
            print(f"  [UNVERIFIED] {ev['url']}")
            print(f"               {ev.get('error') or ev.get('decision_reason_codes', [])}")
