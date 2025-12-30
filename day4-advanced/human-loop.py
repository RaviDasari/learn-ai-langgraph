import sys
import os
import asyncio
import json
from typing import TypedDict, Annotated, List

# Add the parent directory to sys.path to allow imports from common
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.model import get_model
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

# Define State
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

def get_user_feedback() -> str:
    """Get feedback from the user via terminal input."""
    return input("\nYour feedback: ").strip()

async def run():
    model = get_model()

    # --- Define Nodes ---
    async def agent(state: AgentState):
        print("--- Agent Node ---")
        messages = [
            SystemMessage(content="You are a helpful assistant that drafts tweets. When revising, only output the revised tweet, nothing else. Do not include any explanations or the user's feedback in your response."),
            *state["messages"]
        ]
        response = await model.ainvoke(messages)
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
    result1 = await app.ainvoke(
        {"messages": [HumanMessage(content="Draft a tweet about LangGraph.")]},
        config
    )

    print("\n[Paused] Current State:")
    state1 = await app.aget_state(config)
    last_message = state1.values["messages"][-1]
    print(f'Agent wrote: "{last_message.content}"')
    print(f"Next step: {state1.next}")

    # --- Run 2: Human Approval Loop ---
    approved = False
    result2 = None
    
    while not approved:
        print("\n2. Awaiting human review...")
        
        # Get user feedback
        user_feedback = get_user_feedback()
        
        # Get current state
        current_state = await app.aget_state(config)
        current_tweet = current_state.values["messages"][-1].content
        
        # Ask LLM to interpret user's intent
        print("\nAnalyzing your feedback...")
        analysis_prompt = f'''You are analyzing user feedback about generated content.

Generated tweet: "{current_tweet}"

User feedback: "{user_feedback}"

Determine if the user is happy with the output or wants to regenerate it.
Respond with ONLY a JSON object in this exact format:
{{
  "satisfied": true/false,
  "reason": "brief explanation"
}}

If the user expresses approval, satisfaction, or says it's good/ok/fine, set satisfied to true.
If the user requests changes, expresses dissatisfaction, or wants improvements, set satisfied to false.'''

        analysis_response = await model.ainvoke([HumanMessage(content=analysis_prompt)])
        
        # Parse LLM response
        try:
            import re
            json_match = re.search(r'\{[\s\S]*\}', analysis_response.content)
            analysis = json.loads(json_match.group(0))
        except Exception as e:
            print("Could not parse LLM response, asking for clarification...")
            analysis = {"satisfied": False, "reason": "unclear feedback"}
        
        satisfied = analysis.get("satisfied", False)
        reason = analysis.get("reason", "")
        print(f"Decision: {'✓ User is satisfied' if satisfied else '↻ User wants changes'} - {reason}")
        
        if satisfied:
            print("\n✓ Approved - Continuing execution...")
            result2 = await app.ainvoke(
                Command(resume="Approved"),
                config
            )
            print("\nFinal Result:")
            print(result2["messages"][-1].content)
            approved = True
        else:
            print(f'\n↻ Regenerating with feedback: "{user_feedback}"')
            
            # Directly ask the model to revise the tweet with full context
            print("\nAsking LLM to regenerate...")
            revision_messages = [
                SystemMessage(content="You are a helpful assistant that drafts tweets. When revising, only output the revised tweet text, nothing else. Do not include any explanations, prefixes like 'Revised tweet:', or quotes."),
                HumanMessage(content=f'Here is the original tweet:\n\n"{current_tweet}"\n\nPlease revise it based on this feedback: {user_feedback}')
            ]
            revised_response = await model.ainvoke(revision_messages)
            revised_tweet = revised_response.content
            
            # Update the state with the feedback and the new AI response
            await app.aupdate_state(config, {
                "messages": [
                    HumanMessage(content=f"Please revise the tweet with this feedback: {user_feedback}"),
                    AIMessage(content=revised_tweet)
                ]
            })
            
            print(f'\n[Paused] Agent revised: "{revised_tweet}"')
            
            # Loop continues to ask for approval again

if __name__ == "__main__":
    asyncio.run(run())
