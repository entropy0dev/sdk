# Trust-gated RAG agent

A LangChain agent that evaluates every external URL through the Entropy0
Trust Control Plane before fetching content from it.

Without a trust gate, RAG pipelines fetch from any source the LLM decides
to visit — including newly registered domains, phishing infrastructure, and
brand impersonation sites. This example adds one tool call between retrieval
and synthesis that makes the decision explicit and auditable.

## How it works

```
User query
    ↓
Agent decides to fetch a URL
    ↓
entropy0_trust_check(url)           ← Entropy0 /v1/decide
    ↓
recommended_action = "proceed"      → fetch + add to context
recommended_action = "proceed_with_caution" → fetch with reduced trust
recommended_action = "sandbox"      → block, inform user
recommended_action = "deny"         → block, inform user
```

The agent receives a plain-English result it can reason about:

```
Trust evaluation for secure-login-verify-account.xyz:
recommended_action=deny (confidence=91%), uncertainty=low,
signals=[NEWLY_REGISTERED_DOMAIN, BRAND_MISMATCH, CERTIFICATE_ANOMALY].
```

## Setup

```bash
pip install -r requirements.txt

export ENTROPY0_API_KEY=sk_ent0_xxxx   # get from entropy0.ai/signup
export OPENAI_API_KEY=sk-...
```

## Run

```bash
python agent.py
```

## Expected output

```
Query: Summarise the content at http://secure-login-verify-account.xyz
...
> Entering new AgentExecutor chain...
> Invoking: entropy0_trust_check with {'target': 'http://secure-login-verify-account.xyz'}
  Trust evaluation: recommended_action=deny (confidence=91%), uncertainty=low,
  signals=[NEWLY_REGISTERED_DOMAIN, BRAND_MISMATCH, CERTIFICATE_ANOMALY].
...
Final answer: I was unable to fetch this URL. Entropy0 flagged it as a
high-confidence threat: newly registered domain with brand impersonation
signals and certificate anomalies. Recommended action: deny.
```

## Customise

Change the policy to make the agent stricter or more permissive:

```python
Entropy0Tool(
    api_key=os.environ["ENTROPY0_API_KEY"],
    policy="strict",          # stricter: sandboxes faster under ambiguity
    interaction_sensitivity="high",  # treats all fetches as high-sensitivity
)
```

## Links

- [entropy0-langchain on PyPI](https://pypi.org/project/entropy0-langchain/)
- [Entropy0 API docs](https://entropy0.ai/docs)
- [Decision model](https://entropy0.ai/docs/decision-model)
- [Get an API key](https://entropy0.ai/signup)
