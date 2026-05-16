# Entropy0 trust gate for LangGraph

A trusted domain is not the same as usable evidence.  
Entropy0 makes that distinction visible before the agent answers.

## Try it on your own URLs right now

Two API keys. No other dependencies.

```bash
pip install -r requirements.txt

export ENTROPY0_API_KEY=sk_ent0_...
export ANTHROPIC_API_KEY=sk-ant-...

python agent.py --urls https://genai.owasp.org/llmrisk/llm01-prompt-injection/ https://example.com
```

Pass any URLs you want to evaluate. The agent runs each through the Entropy0 trust gate, scores the extracted content for usability, then answers only from what was actually retrieved.

No Tavily key. No search API signup. Search uses DuckDuckGo (free). Extraction uses trafilatura (free).

Get an Entropy0 API key at [entropy0.ai](https://entropy0.ai).

---

## What this does

```
Search → Entropy0 trust gate → Extract → Evidence scoring → Synthesize
```

Two quality gates before the model sees anything:

**Gate 1 — Entropy0 trust gate**: calls `/v1/decide` on every candidate URL. Sources that return `sandbox` or `deny` are logged and skipped. Trust metadata (`trust`, `threat`, `deviation`, `reason_codes`, `action`) flows through the graph and appears in LangSmith traces.

**Gate 2 — Evidence usability scoring**: classifies extracted content into five tiers: `body_text_captured`, `partial`, `title_only`, `boilerplate_dominant`, `unusable`. The synthesis step calibrates confidence by tier — `usability=high` sources are cited directly; `usability=low` sources are noted but no specific claims are attributed to them.

The key distinction:

> An approved source can still return boilerplate. Gate 1 tells you the domain is trustworthy. Gate 2 tells you whether the content was actually usable.

---

## Run modes

```bash
# Bring your own URLs — skip search, run the trust gate on sources you choose
python agent.py --urls https://genai.owasp.org/llmrisk/llm01-prompt-injection/ https://arxiv.org/abs/2302.12173

# With a custom question for synthesis context
python agent.py "what are the risks of prompt injection?" --urls https://genai.owasp.org/llmrisk/llm01-prompt-injection/

# Web search mode — DuckDuckGo finds candidate URLs automatically
python agent.py "what are the risks of prompt injection in LLM agents?"
```

---

## Example output (`--urls` mode)

```
Query:  what are the risks of prompt injection in LLM agents?
URLs:   ['https://genai.owasp.org/llmrisk/llm01-prompt-injection/', 'https://owasp.org']

[search] using 2 provided URLs — skipping search

[entropy0] 2 approved / 0 sandboxed / 0 denied / 0 unverified

[evidence layer]
  [+] https://genai.owasp.org/llmrisk/llm01-prompt-injection/
    body_text_captured — usability=high ['FULL_ARTICLE_BODY']
  [+] https://owasp.org
    body_text_captured — usability=high ['FULL_ARTICLE_BODY']

============================================================
ANSWER
============================================================
[High-confidence answer sourced directly from the OWASP GenAI page.
Cites specific risk types: sensitive information disclosure, content
manipulation, unauthorized access, arbitrary command execution.
Distinguishes direct vs indirect injection. Notes that RAG and
fine-tuning do not fully mitigate the vulnerability.]

============================================================
EVIDENCE LAYER
============================================================
  [FULL]  https://genai.owasp.org/llmrisk/llm01-prompt-injection/
             usability=high  ['FULL_ARTICLE_BODY']
  [FULL]  https://owasp.org
             usability=high  ['FULL_ARTICLE_BODY']
```

## Example output (search mode, showing trust gate in action)

```
[entropy0] 5 approved / 1 sandboxed / 0 denied / 0 unverified
  SANDBOX    https://outpost24.com/blog/explaining-prompt-injection-attacks/
             trust signals:  ['LONG_OPERATIONAL_HISTORY', 'STRONG_BRAND_ALIGNMENT']
             sandbox reason: ['ELEVATED_DEVIATION']

[evidence layer]
  [!] https://www.microsoft.com/en-us/security/blog/...
    boilerplate_dominant — usability=low ['BOILERPLATE_DOMINANT']
  [+] https://pmc.ncbi.nlm.nih.gov/articles/PMC12717619/
    body_text_captured — usability=high ['FULL_ARTICLE_BODY']
  [+] https://www.securecodewarrior.com/article/prompt-injection-...
    body_text_captured — usability=high ['FULL_ARTICLE_BODY']

============================================================
GATED / UNVERIFIED SOURCES
============================================================
  [SANDBOX]    https://outpost24.com/blog/explaining-prompt-injection-attacks/
               trust signals:  ['LONG_OPERATIONAL_HISTORY', 'STRONG_BRAND_ALIGNMENT']
               sandbox reason: ['ELEVATED_DEVIATION']
```

Note: outpost24.com has genuine positive trust signals. It was sandboxed because its deviation score crossed the threshold — not because it is malicious. The display separates the two so the reason is not misread.

---

## Why this matters

Most AI search agents pass every retrieved source directly to the model. Two problems follow:

**Problem 1 — source trust**: A newly registered domain, a typosquat, or a parked page should not be treated the same as an established authoritative source. Entropy0 provides source-trust telemetry at the pre-fetch boundary — before content reaches the model.

**Problem 2 — evidence quality**: Even a trusted source can return nav menus, cookie banners, and UI chrome instead of article text. An agent that doesn't detect this will fabricate content or hallucinate confidence it doesn't have.

Entropy0 solves Problem 1. The evidence usability layer solves Problem 2. Together:

> **Entropy0 does not merely block bad sources. It helps agents know when not enough trustworthy evidence was actually retrieved.**

### LangChain continual learning context

If agents learn from their traces, what enters those traces determines what they learn. A poisoned, deceptive, or structurally weak domain should not be treated the same as a stable trusted source. Entropy0 provides source-trust telemetry at the pre-fetch boundary — before content reaches the model or the trace.

---

## The two-gate model

| Gate | Question answered | Output |
|---|---|---|
| Entropy0 trust gate | Should this source enter the workflow? | approved / sandboxed / denied / unverified |
| Evidence usability | Did we retrieve usable article text? | body_text_captured / partial / title_only / boilerplate_dominant / unusable |

---

## Trust gate node

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

`check_url` never throws — it returns `action: unverified` on any error so the agent degrades gracefully when the trust API is unavailable.

### Trust signals vs sandbox reason

For sandboxed domains, reason codes are split:

- **UGC platforms** (Reddit, GitHub, etc.): decision codes are `["USER_GENERATED_CONTENT", "REQUIRES_CORROBORATION"]` — platform trust does not equal content trust
- **Sandboxed with only positive trust signals**: when `LONG_OPERATIONAL_HISTORY` and `STRONG_BRAND_ALIGNMENT` are present but the domain is still sandboxed, the sandbox decision came from score thresholds (threat/deviation band). The display shows them separately so the reason is not misread as "trusted signals caused the sandbox."

---

## Evidence usability node

`score_evidence_usability()` classifies each extracted text block using boilerplate signal density, short-line ratio, and markdown link density:

```
body_text_captured   — substantial article text, low boilerplate
partial              — some article text, under 1000 chars or mixed
title_only           — under 400 chars, likely just title/metadata
boilerplate_dominant — high density of nav/consent/cookie terms
unusable             — under 100 chars or extraction failed entirely
```

The synthesis prompt instructs the model to calibrate confidence by tier:

```
usability=high:   cite directly with high confidence
usability=medium: cite with a caveat (partial extraction)
usability=low:    note the source exists, make no specific claims
usability=none:   mark as unusable, do not cite
```

---

## How it connects to LangSmith

Each node is wrapped with `@traceable`. Both the trust gate decision and the evidence usability score appear as named steps in the LangSmith trace. Every trace carries:

- A trust provenance record for every source evaluated
- An evidence quality record for every source that passed the trust gate
