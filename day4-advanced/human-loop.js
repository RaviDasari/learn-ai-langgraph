import { getModel } from "../common/model.js";
import { StateGraph, MessagesAnnotation, MemorySaver } from "@langchain/langgraph";
import { HumanMessage, AIMessage } from "@langchain/core/messages";
import { Command } from "@langchain/langgraph";

const run = async () => {
    const model = getModel();

    // --- Define Nodes ---
    const agent = async (state) => {
        console.log("--- Agent Node ---");
        const response = await model.invoke(state.messages);
        return { messages: [response] };
    };

    const humanReview = async (state) => {
        console.log("--- Human Review Node ---");
        // This node doesn't do much, it's just a placeholder for the interrupt.
        // In a real app, you might validate the input here.
        return {};
    };

    // --- Define Graph ---
    const workflow = new StateGraph(MessagesAnnotation)
        .addNode("agent", agent)
        .addNode("human_review", humanReview)
        .addEdge("__start__", "agent")
        .addEdge("agent", "human_review")
        .addEdge("human_review", "__end__");

    // --- Persistence ---
    const checkpointer = new MemorySaver();

    const app = workflow.compile({
        checkpointer,
        interruptBefore: ["human_review"], // Pause before entering this node
    });

    // --- Run 1: Initial Execution ---
    const threadId = "thread-1";
    const config = { configurable: { thread_id: threadId } };

    console.log("1. Starting execution...");
    const result1 = await app.invoke(
        { messages: [new HumanMessage("Draft a tweet about LangGraph.")] },
        config
    );

    console.log("\n[Paused] Current State:");
    const state1 = await app.getState(config);
    const lastMessage = state1.values.messages[state1.values.messages.length - 1];
    console.log(`Agent wrote: "${lastMessage.content}"`);
    console.log(`Next step: ${state1.next}`);

    // --- Run 2: Resume (Human Approval) ---
    console.log("\n2. Resuming execution (Simulating Human Approval)...");

    // We can update the state if we want (e.g., editing the tweet), 
    // or just resume. Let's just resume.
    // To resume, we can pass null or a Command.

    const result2 = await app.invoke(
        new Command({ resume: "Approved" }),
        config
    );

    console.log("\nFinal Result:");
    console.log(result2.messages[result2.messages.length - 1].content);
};

run().catch(console.error);
