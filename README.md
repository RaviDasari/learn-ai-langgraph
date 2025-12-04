# 4-Day LangChain/LangGraph Series

Welcome to the 4-day series on building Agentic AI applications with LangChain and LangGraph in Node.js and in Python.

## Why this Series?

Large Language Models (LLMs) are powerful, but building reliable applications requires more than just a prompt. This series focuses on **Agentic AI**—systems that can reason, use tools, and make decisions to solve complex problems.

By the end of this course, you will understand how to:
*   Move beyond simple linear chains to cyclic graphs.
*   Build agents that can browse the web and interact with external APIs.
*   Orchestrate multiple agents working together.
*   Implement "human-in-the-loop" workflows for safety and control.

This is essential for developers looking to build production-grade AI applications that are robust, stateful, and capable of autonomous action.

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

### Python

```bash
pip install -r requirements.txt
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
    # Python
    python day1-foundations/1-simple-chat.py
    python day1-foundations/2-rag-chain.py
    ```

### [Day 2: Intro to LangGraph (Agents)](https://github.com/RaviDasari/learn-ai-langgraph/blob/main/day2-langgraph/README.md)
*   **Concepts**: StateGraphs, Nodes, Edges, Conditional Logic, Tools.
*   **Goal**: Refactor the linear RAG chain into an autonomous Agent that *decides* when to search.
*   **Run**:
    ```bash
    node day2-langgraph/agent.js
    # Python
    python day2-langgraph/agent.py
    ```

### [Day 3: Multi-Agent Systems](https://github.com/RaviDasari/learn-ai-langgraph/blob/main/day3-multi-agent/README.md)
*   **Concepts**: Supervisor Pattern, Specialized Agents, Shared State.
*   **Goal**: Orchestrate a team of agents (Researcher + Writer) to collaborate on a task.
*   **Run**:
    ```bash
    node day3-multi-agent/team.js
    # Python
    python day3-multi-agent/team.py
    ```

### [Day 4: Advanced Patterns (Persistence & Control)](https://github.com/RaviDasari/learn-ai-langgraph/blob/main/day4-advanced/README.md)
*   **Concepts**: Checkpointers (Memory), Interrupts, Human-in-the-loop.
*   **Goal**: Add "Time Travel" and Human Approval steps to make the agent safe for production.
*   **Run**:
    ```bash
    node day4-advanced/human-loop.js
    # Python
    python day4-advanced/human-loop.py
    ```
