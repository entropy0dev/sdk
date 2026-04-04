# Entropy0 SDK

First-party integrations for the [Entropy0 Trust Control Plane](https://entropy0.ai).

## Packages

| Package | Registry | Description |
|---|---|---|
| [`entropy0-langchain`](packages/langchain) | PyPI | LangChain tool — trust-gate URLs before agents fetch from them |
| [`@entropy0/express`](packages/express) | npm | Express middleware — evaluate request targets through `/v1/decide` |

## Quick start

**LangChain (Python)**
```bash
pip install entropy0-langchain
```
```python
from entropy0_langchain import Entropy0Tool
tools = [Entropy0Tool(api_key="sk_ent0_xxxx")]
```

**Express (Node.js)**
```bash
npm install @entropy0/express
```
```typescript
import { entropy0Guard } from "@entropy0/express";
app.use(entropy0Guard({ apiKey: process.env.ENTROPY0_API_KEY! }));
```

## Examples

| Example | Description |
|---|---|
| [`examples/rag-agent`](examples/rag-agent) | LangChain agent that trust-gates every URL before fetching content |

## Links

- [API reference](https://entropy0.ai/docs)
- [Decision model](https://entropy0.ai/docs/decision-model)
- [Get an API key](https://entropy0.ai/signup)
