# entropy0-langchain

Entropy0 Trust Control Plane tool for LangChain agents.

Evaluates any domain or URL through the Entropy0 `/v1/decide` endpoint before your agent fetches content from it. Returns a machine-readable recommended action the agent can reason about directly.

**Get a free API key at [entropy0.ai/signup](https://entropy0.ai/signup)** — no credit card required.

## Install

```bash
pip install entropy0-langchain
```

## Usage

```python
from langchain.agents import initialize_agent, AgentType
from langchain_openai import ChatOpenAI
from entropy0_langchain import Entropy0Tool

llm = ChatOpenAI(model="gpt-4o")

tools = [
    Entropy0Tool(api_key="sk_ent0_xxxx")
]

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
)

# The agent will call entropy0_trust_check before fetching any external URL
agent.invoke("Summarize the content at https://suspicious-domain.xyz/report")
```

The agent receives structured trust output:

```
Trust evaluation for suspicious-domain.xyz: recommended_action=sandbox
(confidence=81%), uncertainty=medium, signals=[NEWLY_REGISTERED_DOMAIN,
HOSTING_PATTERN_SUSPICIOUS, LOW_REPUTATION_EVIDENCE].
```

It can then decide to proceed, apply caution, or refuse — without you writing any decision logic.

## Configuration

```python
Entropy0Tool(
    api_key="sk_ent0_xxxx",          # required
    policy="strict",                  # open | balanced | strict | critical
    interaction_kind="fetch",         # navigate | fetch | enrich | download_file | ...
    interaction_mode="read_only",     # read_only | transactional | privileged
    interaction_sensitivity="high",   # low | medium | high | critical
    timeout=10.0,                     # seconds
)
```

## How it works

Each call to `entropy0_trust_check` sends a `POST /v1/decide` request with the target and your interaction context. The Entropy0 engine runs a deterministic 7-step pipeline:

1. Classifies the target into one of six states (Clear Threat → Safe Known)
2. Looks up the base action from a policy routing table
3. Shifts strictness based on interaction risk tier
4. Applies confidence clamps (low-confidence negatives never hard-deny)
5. Returns `recommended_action` + reason codes + uncertainty + bounded validity

The same inputs always produce the same output. No probabilistic scoring, no black-box models.

## Requirements

- Python 3.9+
- `langchain-core >= 0.1.0`
- `httpx >= 0.24.0`

## Links

- [API docs](https://entropy0.ai/docs)
- [Decision model](https://entropy0.ai/docs/decision-model)
- [Get an API key](https://entropy0.ai/signup)
