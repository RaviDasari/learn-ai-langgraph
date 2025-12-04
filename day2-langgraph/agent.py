import sys
import os
import asyncio
from typing import TypedDict, Annotated, List, Union

# Add the parent directory to sys.path to allow imports from common
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.model import get_model
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_core.tools import tool
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages

# Define State
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

# --- Setup Vector Store (Same as Day 1) ---
async def setup_vector_store():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "../day1-foundations/info.txt")
    loader = TextLoader(file_path)
    docs = await asyncio.to_thread(loader.load)
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    split_docs = splitter.split_documents(docs)
    
    embeddings = OpenAIEmbeddings()
    return await asyncio.to_thread(FAISS.from_documents, split_docs, embeddings)

async def run():
    vector_store = await setup_vector_store()
    retriever = vector_store.as_retriever()

    # --- Define Tools ---
    @tool
    async def lookup_policy(query: str) -> str:
        """Ask this tool questions about LangGraph."""
        docs = await retriever.ainvoke(query)
        return "\n\n".join([d.page_content for d in docs])

    tools = [lookup_policy]
    tool_node = ToolNode(tools)

    # --- Define Graph ---
    model = get_model().bind_tools(tools)

    # Node: Agent (Calls Model)
    async def call_model(state: AgentState):
        response = await model.ainvoke(state["messages"])
        return {"messages": [response]}

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)
    
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", tools_condition)
    workflow.add_edge("tools", "agent")

    app = workflow.compile()

    # --- Run ---
    print("User: What is LangGraph inspired by?")
    result = await app.ainvoke({
        "messages": [HumanMessage(content="What is LangGraph inspired by?")]
    })

    print("\nFinal Response:")
    print(result["messages"][-1].content)

if __name__ == "__main__":
    asyncio.run(run())
