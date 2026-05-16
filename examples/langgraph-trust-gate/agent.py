"""
Entropy0 trust gate for LangGraph

Pipeline: Search → Trust gate → Extract → Evidence scoring → Synthesize

Every URL returned by web search passes through two quality gates before the
model generates an answer:

  1. Entropy0 trust gate   — evaluates source trustworthiness via /v1/decide
  2. Evidence usability    — classifies whether extracted text is usable

No Tavily API key required. Search uses DuckDuckGo (free, no key).
Extraction uses trafilatura (free, no key).

Setup:
    pip install -r requirements.txt

    export ENTROPY0_API_KEY=sk_ent0_...
    export ANTHROPIC_API_KEY=sk-ant-...
    export LANGCHAIN_API_KEY=ls__...       # optional — enables LangSmith tracing
    export LANGCHAIN_TRACING_V2=true       # optional

Run (web search mode):
    python agent.py "what are the risks of prompt injection in LLM agents?"

Run (bring your own URLs — skip search):
    python agent.py --urls https://owasp.org https://example.com
    python agent.py "my query" --urls https://owasp.org https://example.com
"""

import argparse
import os
import re
import sys
import httpx
import trafilatura
from typing import TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
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

# Terms that, when present in high density, indicate boilerplate rather than article text
BOILERPLATE_SIGNALS = [
    "cookie", "accept all", "privacy policy", "skip to content",
    "sign in", "sign up", "log in", "register", "subscribe",
    "toggle navigation", "breadcrumb", "all rights reserved",
    "copyright ©", "consent", "gdpr", "terms of service",
]


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


def score_evidence_usability(text: str) -> dict:
    """
    Classify extracted content quality into one of five tiers.

    Returns:
      content_status     — body_text_captured / partial / title_only /
                           boilerplate_dominant / unusable
      evidence_usability — high / medium / low / none
      content_reason_codes — machine-readable reason list
    """
    cleaned = (text or "").strip()

    if len(cleaned) < 100:
        return {
            "content_status":       "unusable",
            "evidence_usability":   "none",
            "content_reason_codes": ["NO_CONTENT_EXTRACTED"],
        }

    text_lower = cleaned.lower()
    lines      = [l.strip() for l in cleaned.split("\n") if l.strip()]
    n_lines    = max(len(lines), 1)

    boilerplate_hits = sum(1 for sig in BOILERPLATE_SIGNALS if sig in text_lower)
    short_line_ratio = sum(1 for l in lines if len(l) < 50) / n_lines
    link_lines       = sum(1 for l in lines if re.search(r'\[.+?\]\(.+?\)', l))
    link_density     = link_lines / n_lines

    if boilerplate_hits >= 3 or (short_line_ratio > 0.65 and link_density > 0.30):
        return {
            "content_status":       "boilerplate_dominant",
            "evidence_usability":   "low",
            "content_reason_codes": ["BOILERPLATE_DOMINANT"],
        }

    if len(cleaned) < 400:
        return {
            "content_status":       "title_only",
            "evidence_usability":   "low",
            "content_reason_codes": ["TITLE_ONLY_EVIDENCE"],
        }

    if len(cleaned) < 1000 or boilerplate_hits >= 2:
        return {
            "content_status":       "partial",
            "evidence_usability":   "medium",
            "content_reason_codes": ["ARTICLE_BODY_PARTIAL"],
        }

    return {
        "content_status":       "body_text_captured",
        "evidence_usability":   "high",
        "content_reason_codes": ["FULL_ARTICLE_BODY"],
    }


def extract_text(url: str) -> str:
    """Extract article text from a URL using trafilatura. Never throws."""
    try:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return ""
        text = trafilatura.extract(
            downloaded,
            include_comments=False,
            include_tables=True,
            no_fallback=False,
        )
        return (text or "").strip()
    except Exception:
        return ""


# ── State ─────────────────────────────────────────────────────────────────────

class ResearchState(TypedDict):
    query:            str
    urls:             list[str]
    approved:         list[dict]   # passed trust gate
    sandboxed:        list[dict]   # gated but not denied
    denied:           list[dict]   # hard block
    unverified:       list[dict]   # could not gather evidence (timeout / extraction fail)
    content:          list[str]
    evidence_quality: list[dict]   # per-source evidence usability scores
    answer:           str


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

        d           = r.json()
        action      = d["decision"]["recommended_action"]
        trust_codes = d["decision"]["reason_codes"]
        trust       = d["scores"]["trust"]["score"]
        threat      = d["scores"]["threat"]["score"]
        dev         = d["scores"]["deviation"]["score"]

        # Separate trust signal from decision rationale.
        # Case 1 — UGC: platform trust ≠ content trust.
        # Case 2 — Sandboxed with only positive codes: sandbox came from score
        #          thresholds, not from these signals — derive reason from scores.
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
    """
    If URLs were pre-populated (--urls flag), skip search.
    Otherwise run DuckDuckGo — free, no API key required.
    """
    if state["urls"]:
        print(f"[search] using {len(state['urls'])} provided URLs — skipping search")
        return {}

    wrapper = DuckDuckGoSearchAPIWrapper(max_results=6)
    results = wrapper.results(state["query"], max_results=6)
    urls    = [r["link"] for r in results if "link" in r]
    print(f"[search] {len(urls)} URLs from DuckDuckGo")
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
    Extract article text from approved URLs using trafilatura, then score
    each result for evidence usability. Boilerplate-dominant or unusable
    extraction is flagged so the synthesis step can calibrate confidence.
    """
    if not state["approved"]:
        return {"content": [], "evidence_quality": []}

    content          = []
    evidence_quality = []

    for ev in state["approved"]:
        url  = ev["url"]
        text = extract_text(url)
        eq   = score_evidence_usability(text)

        evidence_quality.append({"url": url, **eq})

        if not text:
            unverified_ev = {**ev, "action": "unverified", "decision_reason_codes": ["CONTENT_UNAVAILABLE"]}
            state["unverified"].append(unverified_ev)
            print(f"  UNVERIFIED {url} — extraction failed")
            continue

        content.append(
            f"[Source: {url}]\n"
            f"[Entropy0: trust={ev.get('trust','?')} threat={ev.get('threat','?')} "
            f"deviation={ev.get('deviation','?')} action={ev.get('action','?')}]\n"
            f"[Evidence: {eq['content_status']} — usability={eq['evidence_usability']}]\n\n"
            f"{text[:3000]}"
        )

    _ICON = {
        "body_text_captured":   "[+]",
        "partial":              "[~]",
        "title_only":           "[!]",
        "boilerplate_dominant": "[!]",
        "unusable":             "[x]",
    }
    print("\n[evidence layer]")
    for eq in evidence_quality:
        icon = _ICON.get(eq["content_status"], "?")
        print(f"  {icon} {eq['url']}")
        print(f"    {eq['content_status']} — usability={eq['evidence_usability']} {eq['content_reason_codes']}")

    return {"content": content, "evidence_quality": evidence_quality}


@traceable(name="synthesize")
def synthesize_node(state: ResearchState) -> dict:
    llm = ChatAnthropic(model="claude-sonnet-4-6", max_tokens=1024)

    gate_summary = (
        f"{len(state['approved'])} sources cleared the Entropy0 trust gate "
        f"({len(state['sandboxed'])} sandboxed, {len(state['denied'])} denied, "
        f"{len(state['unverified'])} unverified — could not gather evidence)."
    )

    eq_lines = [
        f"  - {eq['url']}: {eq['content_status']} (usability={eq['evidence_usability']})"
        for eq in state.get("evidence_quality", [])
    ]
    eq_summary = "\n".join(eq_lines) if eq_lines else "  (no evidence quality data)"

    context = "\n\n---\n\n".join(state["content"]) if state["content"] else \
              "No usable content was extracted from approved sources."

    response = llm.invoke([{
        "role": "user",
        "content": (
            f"Research question: {state['query']}\n\n"
            f"Trust gate summary: {gate_summary}\n\n"
            f"Evidence quality per approved source:\n{eq_summary}\n\n"
            f"Content from approved sources only:\n\n{context}\n\n"
            "Answer the research question based solely on the approved sources above.\n"
            "Calibrate confidence strictly by evidence usability:\n"
            "- usability=high:   cite directly with high confidence\n"
            "- usability=medium: cite with a caveat (partial extraction)\n"
            "- usability=low:    note the source exists but make no specific claims\n"
            "- usability=none:   mark as unusable, do not cite\n"
            "Do not fabricate or infer details not present in the retrieved content."
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
    parser = argparse.ArgumentParser(description="Entropy0 trust gate for LangGraph")
    parser.add_argument(
        "query",
        nargs="?",
        default="what are the risks of prompt injection in LLM agents?",
        help="Research question (used for search and synthesis)",
    )
    parser.add_argument(
        "--urls",
        nargs="+",
        metavar="URL",
        help="Skip search and evaluate these URLs directly through the trust gate",
    )
    args = parser.parse_args()

    query        = args.query
    initial_urls = args.urls or []

    if initial_urls:
        print(f"Query:  {query}")
        print(f"URLs:   {initial_urls}\n")
    else:
        print(f"Query: {query}\n")

    graph  = build_graph()
    result = graph.invoke({
        "query":            query,
        "urls":             initial_urls,
        "approved":         [],
        "sandboxed":        [],
        "denied":           [],
        "unverified":       [],
        "content":          [],
        "evidence_quality": [],
        "answer":           "",
    })

    print("\n" + "=" * 60)
    print("ANSWER")
    print("=" * 60)
    print(result["answer"])

    if result.get("evidence_quality"):
        print("\n" + "=" * 60)
        print("EVIDENCE LAYER")
        print("=" * 60)
        _LABEL = {
            "body_text_captured":   "FULL",
            "partial":              "PARTIAL",
            "title_only":           "TITLE",
            "boilerplate_dominant": "BOILERPLATE",
            "unusable":             "UNUSABLE",
        }
        for eq in result["evidence_quality"]:
            label = _LABEL.get(eq["content_status"], eq["content_status"].upper())
            print(f"  [{label}]  {eq['url']}")
            print(f"             usability={eq['evidence_usability']}  {eq.get('content_reason_codes', [])}")

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
