import sys
import os
import asyncio
from typing import TypedDict, Annotated, List

# Add the parent directory to sys.path to allow imports from common
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.model import get_model
from langchain_core.messages import HumanMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

# Define State
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

async def run():
    model = get_model()

    # --- Define Nodes ---
    async def agent(state: AgentState):
        print("--- Agent Node ---")
        response = await model.ainvoke(state["messages"])
        return {"messages": [response]}

    async def human_review(state: AgentState):
        print("--- Human Review Node ---")
        # This node doesn't do much, it's just a placeholder for the interrupt.
        return {}

    # --- Define Graph ---
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent)
    workflow.add_node("human_review", human_review)

    workflow.add_edge(START, "agent")
    workflow.add_edge("agent", "human_review")
    workflow.add_edge("human_review", END)

    # --- Persistence ---
    checkpointer = MemorySaver()

    app = workflow.compile(
        checkpointer=checkpointer,
        interrupt_before=["human_review"], # Pause before entering this node
    )

    # --- Run 1: Initial Execution ---
    thread_id = "thread-1"
    config = {"configurable": {"thread_id": thread_id}}

    print("1. Starting execution...")
    # Using stream to handle the interrupt gracefully or just invoke
    # invoke will stop at the interrupt
    result1 = await app.ainvoke(
        {"messages": [HumanMessage(content="Draft a tweet about LangGraph.")]},
        config
    )

    print("\n[Paused] Current State:")
    state1 = await app.aget_state(config)
    last_message = state1.values["messages"][-1]
    print(f'Agent wrote: "{last_message.content}"')
    print(f"Next step: {state1.next}")

    # --- Run 2: Resume (Human Approval) ---
    print("\n2. Resuming execution (Simulating Human Approval)...")

    # To resume, we can pass a Command with resume value
    result2 = await app.ainvoke(
        Command(resume="Approved"),
        config
    )

    print("\nFinal Result:")
    print(result2["messages"][-1].content)

if __name__ == "__main__":
    asyncio.run(run())
