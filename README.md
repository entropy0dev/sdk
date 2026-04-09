# Entropy0 SDK

AI agents fetch URLs, follow links, and act on domains they've never seen before. Most of the time that's fine. Sometimes it isn't — phishing infrastructure, newly registered lookalike domains, brand impersonation sites, or plain malware hosting.

Entropy0 adds a trust gate between your agent and the external web. One API call returns a machine-readable recommended action your agent can reason about before it fetches, navigates, or transacts.

**[Try it without signing up →](https://entropy0.ai/playground)**

```
POST /v1/decide
{ "domain": "secure-login-verify-account.xyz" }

→ recommended_action: "deny"
  confidence: 91%
  signals: [NEWLY_REGISTERED_DOMAIN, BRAND_MISMATCH, CERTIFICATE_ANOMALY]
```

---

## Packages

| Package | Registry | Description |
|---|---|---|
| [`entropy0-langchain`](packages/langchain) | PyPI | LangChain tool — trust-gate URLs before agents fetch from them |
| [`@entropy0/express`](packages/express) | npm | Express middleware — evaluate request targets through `/v1/decide` |

---

## Quick start

**LangChain (Python)**
```bash
pip install entropy0-langchain
```
```python
from entropy0_langchain import Entropy0Tool

tools = [Entropy0Tool(api_key="sk_ent0_xxxx")]
# Agent will call entropy0_trust_check before fetching any external URL
```

**Express (Node.js)**
```bash
npm install @entropy0/express
```
```typescript
import { entropy0Guard } from "@entropy0/express";

app.use(entropy0Guard({ apiKey: process.env.ENTROPY0_API_KEY! }));
// Requests to flagged domains are blocked before your handlers run
```

**Direct API**
```bash
curl -X POST https://entropy0.ai/v1/decide \
  -H "X-API-Key: sk_ent0_xxxx" \
  -H "Content-Type: application/json" \
  -d '{"domain": "example.com"}'
```

---

## How it works

Each decision runs a deterministic pipeline — same inputs always produce the same output:

1. Classifies the domain (Clear Threat → Safe Known) using WHOIS, DNS, SSL, and threat intel feeds
2. Maps classification to a base action under your chosen policy
3. Shifts strictness based on interaction risk (fetch vs transactional vs privileged)
4. Applies confidence clamps — low-confidence negatives never hard-deny
5. Returns `recommended_action` + reason codes + uncertainty + bounded validity window

No probabilistic black boxes. Auditable, explainable, overridable.

---

## Examples

| Example | Description |
|---|---|
| [`examples/rag-agent`](examples/rag-agent) | LangChain agent that trust-gates every URL before fetching content |

---

## Links

- [Live playground](https://entropy0.ai/playground) — no sign-up required
- [API reference](https://entropy0.ai/docs)
- [Get a free API key](https://entropy0.ai/signup) — 150 scans/month, no credit card
