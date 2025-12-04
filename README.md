# 4-Day LangChain/LangGraph Series

Welcome to the 4-day series on building Agentic AI applications with LangChain and LangGraph in Node.js.

## Prerequisites

1.  **Node.js**: Ensure you have Node.js installed (v18+ recommended).
2.  **API Keys**:
    *   Create a `.env` file in the root directory (copy from `.env.example`).
    *   Add your `OPENAI_API_KEY`.
    *   (Optional) Add `TAVILY_API_KEY` if you want to use real search in Day 3 (code defaults to mock).

## Installation

```bash
npm install
```

## Agenda & Curriculum

This series is designed to take you from basic LLM interactions to building a production-ready, human-in-the-loop agentic workflow.

### [Day 1: Foundations (RAG)](https://github.com/RaviDasari/learn-ai-langgraph/blob/main/day1-foundations/README.md)
*   **Concepts**: Embeddings, Vector Stores, Retrieval Augmented Generation.
*   **Goal**: Build a "Smart Reader" that can answer questions about your private data.
*   **Run**:
    ```bash
    node day1-foundations/1-simple-chat.js
    node day1-foundations/2-rag-chain.js
    ```

### [Day 2: Intro to LangGraph (Agents)](https://github.com/RaviDasari/learn-ai-langgraph/blob/main/day2-langgraph/README.md)
*   **Concepts**: StateGraphs, Nodes, Edges, Conditional Logic, Tools.
*   **Goal**: Refactor the linear RAG chain into an autonomous Agent that *decides* when to search.
*   **Run**:
    ```bash
    node day2-langgraph/agent.js
    ```

### [Day 3: Multi-Agent Systems](https://github.com/RaviDasari/learn-ai-langgraph/blob/main/day3-multi-agent/README.md)
*   **Concepts**: Supervisor Pattern, Specialized Agents, Shared State.
*   **Goal**: Orchestrate a team of agents (Researcher + Writer) to collaborate on a task.
*   **Run**:
    ```bash
    node day3-multi-agent/team.js
    ```

### [Day 4: Advanced Patterns (Persistence & Control)](https://github.com/RaviDasari/learn-ai-langgraph/blob/main/day4-advanced/README.md)
*   **Concepts**: Checkpointers (Memory), Interrupts, Human-in-the-loop.
*   **Goal**: Add "Time Travel" and Human Approval steps to make the agent safe for production.
*   **Run**:
    ```bash
    node day4-advanced/human-loop.js
    ```
