import sys
import os
import asyncio
from typing import TypedDict, Annotated, List, Union, Literal

# Add the parent directory to sys.path to allow imports from common
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.model import get_model
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages

# --- Tools ---
@tool
def search_tool(query: str) -> str:
    """Search the web for information."""
    print(f"[Search Tool] Searching for: {query}")
    return "LangGraph is a library for building stateful, multi-actor applications with LLMs. It is built on top of LangChain."

tools = [search_tool]
tool_node = ToolNode(tools)

# Define State
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

# --- Agent Factory ---
def create_agent(model, system_prompt, tools=None):
    if tools is None:
        tools = []
    
    model_with_tools = model.bind_tools(tools)
    
    async def agent_node(state: AgentState):
        messages = [SystemMessage(content=system_prompt)] + state["messages"]
        response = await model_with_tools.ainvoke(messages)
        return {"messages": [response]}
        
    return agent_node

async def run():
    model = get_model()

    # --- Define Agents ---
    researcher_node = create_agent(
        model,
        "You are a researcher. You have access to a search tool. Find information about the user's topic.",
        tools
    )

    writer_node = create_agent(
        model,
        "You are a writer. Write a short blog post based on the research provided in the conversation history. Do not use tools."
    )

    # --- Supervisor (Router) ---
    def should_continue_researcher(state: AgentState) -> Literal["tools", "writer"]:
        last_message = state["messages"][-1]
        if last_message.tool_calls:
            return "tools"
        return "writer"

    workflow = StateGraph(AgentState)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("tools", tool_node)
    workflow.add_node("writer", writer_node)

    workflow.add_edge(START, "researcher")
    workflow.add_conditional_edges("researcher", should_continue_researcher)
    workflow.add_edge("tools", "researcher")
    workflow.add_edge("writer", END)

    app = workflow.compile()

    print("User: Research LangGraph and write a blog post.")
    result = await app.ainvoke({
        "messages": [HumanMessage(content="Research LangGraph and write a blog post.")]
    })

    print("\nFinal Result (Last Message):")
    print(result["messages"][-1].content)

if __name__ == "__main__":
    asyncio.run(run())
