# Entropy0 trust gate for LangGraph

A trusted domain is not the same as usable evidence.  
Entropy0 makes that distinction visible before the agent answers.

## What this does

```
Search → Entropy0 trust gate → Extract → Evidence scoring → Synthesize
```

Every URL returned by web search passes through two quality gates before the model generates an answer:

1. **Entropy0 trust gate** — evaluates source trustworthiness via `/v1/decide`. Sources that return `sandbox` or `deny` are logged and skipped. Trust metadata (`trust`, `threat`, `deviation`, `reason_codes`, `action`) is attached to each step and visible in LangSmith traces.

2. **Evidence usability scoring** — classifies extracted content quality into one of five tiers: `body_text_captured`, `partial`, `title_only`, `boilerplate_dominant`, `unusable`. The synthesis step uses these tiers to calibrate confidence: `usability=high` sources are cited directly; `usability=low` sources are noted but no specific claims are attributed to them.

## Why this matters

Most AI search agents pass every retrieved source directly to the model. This creates two problems:

**Problem 1 — source trust**: A newly registered domain, a typosquat, or a parked page should not be treated the same as an established authoritative source. Entropy0 provides source-trust telemetry at the pre-fetch boundary.

**Problem 2 — evidence quality**: Even a trusted source can return nav menus, cookie banners, and UI chrome instead of article text. An agent that doesn't detect this will either fabricate content or hallucinate confidence it doesn't have.

Entropy0 solves Problem 1. The evidence usability layer solves Problem 2. Together they let the agent answer:

> "This source was trusted and the content was usable" — or be honest that it wasn't.

### LangChain continual learning context

LangChain Labs and similar efforts are building continual learning for agents — training on traces, feedback, and production behavior. If agents learn from their traces, what enters those traces determines what they learn. A poisoned, deceptive, or structurally weak domain should not be treated the same as a stable trusted source. Entropy0 provides source-trust telemetry at the pre-fetch boundary — before content reaches the model.

## Setup

```bash
pip install -r requirements.txt

export ENTROPY0_API_KEY=sk_ent0_...
export ANTHROPIC_API_KEY=sk-ant-...

# Optional: LangSmith tracing
export LANGCHAIN_API_KEY=ls__...
export LANGCHAIN_TRACING_V2=true
```

No Tavily key required. Search uses DuckDuckGo (free). Extraction uses trafilatura (free).

Get an Entropy0 API key at [entropy0.ai](https://entropy0.ai).

## Run

```bash
# Web search mode — DuckDuckGo finds candidate URLs automatically
python agent.py "what are the risks of prompt injection in LLM agents?"

# Bring your own URLs — skip search, evaluate specific sources directly
python agent.py --urls https://owasp.org/www-project-top-ten/ https://example.com

# Both — use your URLs but provide a query for synthesis context
python agent.py "prompt injection risks" --urls https://owasp.org https://arxiv.org/abs/2302.12173
```

## Example output

```
Query: what are the risks of prompt injection in LLM agents?

[entropy0] 5 approved / 1 sandboxed / 0 denied / 0 unverified
  SANDBOX    https://outpost24.com/blog/explaining-prompt-injection-attacks/
             trust signals: ['LONG_OPERATIONAL_HISTORY', 'STRONG_BRAND_ALIGNMENT']
             sandbox reason: ['ELEVATED_DEVIATION']

[evidence layer]
  ! https://www.microsoft.com/en-us/security/blog/...
    boilerplate_dominant — usability=low ['BOILERPLATE_DOMINANT']
  ! https://genai.owasp.org/llmrisk/llm01-prompt-injection/
    boilerplate_dominant — usability=low ['BOILERPLATE_DOMINANT']
  ✓ https://pmc.ncbi.nlm.nih.gov/articles/PMC12717619/
    body_text_captured — usability=high ['FULL_ARTICLE_BODY']
  ✓ https://www.securecodewarrior.com/article/prompt-injection-...
    body_text_captured — usability=high ['FULL_ARTICLE_BODY']

============================================================
ANSWER
============================================================
[answer citing only the two high-usability sources, explicitly noting that
Microsoft and OWASP returned boilerplate and cannot be cited]

============================================================
EVIDENCE LAYER
============================================================
  [BOILERPLATE]  https://www.microsoft.com/en-us/security/blog/...
             usability=low  ['BOILERPLATE_DOMINANT']
  [FULL]  https://pmc.ncbi.nlm.nih.gov/articles/PMC12717619/
             usability=high  ['FULL_ARTICLE_BODY']

============================================================
GATED / UNVERIFIED SOURCES
============================================================
  [SANDBOX]    https://outpost24.com/blog/explaining-prompt-injection-attacks/
               trust signals:  ['LONG_OPERATIONAL_HISTORY', 'STRONG_BRAND_ALIGNMENT']
               sandbox reason: ['ELEVATED_DEVIATION']
```

## The two-gate model

| Gate | Question answered | Output |
|---|---|---|
| Entropy0 trust gate | Should this source enter the workflow? | approved / sandboxed / denied / unverified |
| Evidence usability | Did we retrieve usable article text? | body_text_captured / partial / title_only / boilerplate_dominant / unusable |

An **approved source with boilerplate-dominant extraction** is explicitly labelled — the agent knows the domain is trustworthy but the content wasn't usable. This distinction matters: the failure is in the extraction layer, not the source.

> **Entropy0 does not merely block bad sources. It helps agents know when not enough trustworthy evidence was actually retrieved.**

## Trust gate node

The core trust gate logic in `trust_gate_node`:

```python
for url in state["urls"]:
    ev = check_url(url)           # calls Entropy0 /v1/decide
    if ev["action"] in ("proceed", "proceed_with_caution"):
        approved.append(ev)
    elif ev["action"] == "deny":
        denied.append(ev)
    elif ev["action"] == "unverified":
        unverified.append(ev)     # timeout or API error
    else:
        sandboxed.append(ev)      # sandbox + escalate_to_human
```

`check_url` never throws — it returns `action: unverified` on any error, so the agent degrades gracefully when the trust API is unavailable.

### Trust signals vs sandbox reason

For sandboxed domains, reason codes are split to avoid confusion:

- **UGC platforms** (Reddit, GitHub, etc.): decision codes are `["USER_GENERATED_CONTENT", "REQUIRES_CORROBORATION"]` — platform trust does not equal content trust
- **Sandboxed with only positive trust signals**: when `LONG_OPERATIONAL_HISTORY` and `STRONG_BRAND_ALIGNMENT` are present but the domain is still sandboxed, the sandbox decision came from score thresholds (threat/deviation band), not from those signals. The display shows them separately so the reason is not misread.

## Evidence usability node

`score_evidence_usability()` classifies each extracted text block using signal density:

```python
# boilerplate_dominant: high density of nav/consent terms + short lines
# partial:              some article text but under 1000 chars or mixed
# body_text_captured:   substantial article text, low boilerplate density
```

The synthesis prompt instructs Claude to calibrate confidence by tier:

```
- usability=high:   cite directly with high confidence
- usability=medium: cite with a caveat (partial extraction)
- usability=low:    note the source exists, make no specific claims
- usability=none:   mark as unusable, do not cite
```

## How it connects to LangSmith

Each node is wrapped with `@traceable`, so both the trust gate decision and the evidence usability score appear as named steps in the LangSmith trace. The approved sources carry inline trust metadata (`trust`, `threat`, `deviation`) and evidence quality metadata (`content_status`, `evidence_usability`) as context passed to the synthesis step.

Every trace produced by this agent carries:
- A trust provenance record for every source evaluated
- An evidence quality record for every source that passed the trust gate
