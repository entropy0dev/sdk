# Entropy0 trust gate for LangGraph

Agents that learn from traces need trustworthy trace data.  
Entropy0 gates what enters the trace before the model ever sees it.

## What this does

```
Search → Trust gate → Fetch approved → Synthesize
```

Every URL returned by web search is evaluated through the Entropy0 `/v1/decide` endpoint before content enters the model context. Sources that return `sandbox` or `deny` are logged and skipped. Trust metadata (`trust`, `threat`, `deviation`, `reason_codes`, `action`) is attached to each step and visible in LangSmith traces.

## Why this matters

LangChain Labs and similar efforts are building continual learning for agents — training on traces, feedback, and production behavior. If agents learn from their traces, what enters those traces determines what they learn.

A poisoned, deceptive, parked, or structurally weak domain should not be treated the same as a stable trusted source. Entropy0 provides source-trust telemetry at the pre-fetch boundary — before content reaches the model.

## Setup

```bash
pip install -r requirements.txt

export ENTROPY0_API_KEY=sk_ent0_...
export ANTHROPIC_API_KEY=sk-ant-...
export TAVILY_API_KEY=tvly-...

# Optional: LangSmith tracing
export LANGCHAIN_API_KEY=ls__...
export LANGCHAIN_TRACING_V2=true
```

Get an Entropy0 API key at [entropy0.ai](https://entropy0.ai).

## Run

```bash
python agent.py "what are the risks of prompt injection in LLM agents?"
```

## Example output

```
Query: what are the risks of prompt injection in LLM agents?

[entropy0] 5 approved / 1 sandboxed / 0 denied
  SANDBOX  http://sketchy-ai-tips.io
           reason_codes=['NEWLY_REGISTERED_DOMAIN', 'LOW_REPUTATION_EVIDENCE']

===========================================================
ANSWER
===========================================================
Prompt injection in LLM agents poses several risks...
[answer from approved sources only]

===========================================================
BLOCKED / SANDBOXED SOURCES
===========================================================
  [SANDBOX] http://sketchy-ai-tips.io
    reason_codes: ['NEWLY_REGISTERED_DOMAIN', 'LOW_REPUTATION_EVIDENCE']
```

## How it connects to LangSmith

Each node is wrapped with `@traceable`, so the trust gate decision — including which URLs were approved, sandboxed, or denied, and why — appears as a named step in the LangSmith trace. The approved sources carry inline trust metadata (`trust`, `threat`, `deviation`) as context passed to the synthesis step.

This means every trace produced by this agent carries a trust provenance record for every source that contributed to the answer.

## The trust gate node

The core logic is in `trust_gate_node`:

```python
for url in state["urls"]:
    ev = check_url(url)           # calls Entropy0 /v1/decide
    if ev["action"] in ("proceed", "proceed_with_caution"):
        approved.append(ev)
    elif ev["action"] == "deny":
        denied.append(ev)
    else:
        sandboxed.append(ev)      # sandbox + escalate_to_human
```

`check_url` never throws — it returns `action: sandbox` on any error, so the agent degrades gracefully when the trust API is unavailable.
