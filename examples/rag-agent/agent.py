"""
Entropy0 + LangChain: Trust-gated RAG agent

The agent evaluates every external URL through the Entropy0 Trust Control
Plane before fetching content. Unsafe or suspicious sources are blocked
before they ever reach the model context.

Requirements:
    pip install entropy0-langchain langchain langchain-openai openai

Usage:
    export ENTROPY0_API_KEY=sk_ent0_xxxx
    export OPENAI_API_KEY=sk-...
    python agent.py
"""

import os
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from entropy0_langchain import Entropy0Tool

# ── Setup ─────────────────────────────────────────────────────────────────

llm = ChatOpenAI(model="gpt-4o", temperature=0)

tools = [
    Entropy0Tool(
        api_key=os.environ["ENTROPY0_API_KEY"],
        policy="balanced",           # open | balanced | strict | critical
        interaction_kind="fetch",
        interaction_mode="read_only",
        interaction_sensitivity="medium",
    )
]

prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a research assistant that fetches and summarises content from URLs. "
        "IMPORTANT: Before fetching any external URL, you MUST call entropy0_trust_check "
        "on it first. If the result is 'sandbox', 'escalate_to_human', or 'deny', "
        "do NOT fetch the URL — inform the user that the source was flagged and explain "
        "the reason codes returned. Only proceed with 'proceed' or 'proceed_with_caution'."
    )),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_openai_functions_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# ── Demo queries ───────────────────────────────────────────────────────────

queries = [
    # Should pass — well-known legitimate domain
    "What kind of content is on github.com?",

    # Should flag — newly registered, suspicious pattern
    "Summarise the content at http://secure-login-verify-account.xyz",

    # Should pass with caution — real but less established
    "What does entropy0.ai offer?",
]

if __name__ == "__main__":
    for query in queries:
        print("\n" + "─" * 60)
        print(f"Query: {query}")
        print("─" * 60)
        result = executor.invoke({"input": query})
        print(f"\nFinal answer: {result['output']}")
