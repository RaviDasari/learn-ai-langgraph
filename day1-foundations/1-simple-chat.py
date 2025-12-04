import sys
import os

# Add the parent directory to sys.path to allow imports from common
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.model import get_model
from langchain_core.messages import HumanMessage, SystemMessage
import asyncio

async def run():
    model = get_model()

    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="What is LangGraph?"),
    ]

    print("Sending request to LLM...")
    response = await model.ainvoke(messages)

    print("\nResponse:")
    print(response.content)

if __name__ == "__main__":
    asyncio.run(run())
