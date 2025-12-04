import { getModel } from "../common/model.js";
import { StateGraph, MessagesAnnotation } from "@langchain/langgraph";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { tool } from "@langchain/core/tools";
import { z } from "zod";
import { ToolNode } from "@langchain/langgraph/prebuilt";

// --- Tools ---
const searchTool = tool(
    async ({ query }) => {
        console.log(`[Search Tool] Searching for: ${query}`);
        return "LangGraph is a library for building stateful, multi-actor applications with LLMs. It is built on top of LangChain.";
    },
    {
        name: "search",
        description: "Search the web for information.",
        schema: z.object({ query: z.string() }),
    }
);

const tools = [searchTool];
const toolNode = new ToolNode(tools);

// --- Agent Factory ---
const createAgent = (model, systemPrompt, tools = []) => {
    const modelWithTools = model.bindTools(tools);
    return async (state) => {
        const messages = [
            new SystemMessage(systemPrompt),
            ...state.messages,
        ];
        const response = await modelWithTools.invoke(messages);
        return { messages: [response] };
    };
};

const run = async () => {
    const model = getModel();

    // --- Define Agents ---
    const researcherNode = createAgent(
        model,
        "You are a researcher. You have access to a search tool. Find information about the user's topic.",
        tools
    );

    const writerNode = createAgent(
        model,
        "You are a writer. Write a short blog post based on the research provided in the conversation history. Do not use tools."
    );

    // --- Supervisor (Router) ---
    // In a real app, this would be an LLM deciding. For simplicity, we'll hardcode a flow:
    // Researcher -> Tools -> Researcher -> Writer -> End

    // Actually, let's make it slightly smarter using a conditional edge from Researcher.
    // If Researcher calls a tool -> Tools.
    // If Researcher returns a final answer (text) -> Writer.

    const shouldContinueResearcher = (state) => {
        const lastMessage = state.messages[state.messages.length - 1];
        if (lastMessage.tool_calls?.length) {
            return "tools";
        }
        return "writer";
    };

    const workflow = new StateGraph(MessagesAnnotation)
        .addNode("researcher", researcherNode)
        .addNode("tools", toolNode)
        .addNode("writer", writerNode)
        .addEdge("__start__", "researcher")
        .addConditionalEdges("researcher", shouldContinueResearcher)
        .addEdge("tools", "researcher")
        .addEdge("writer", "__end__");

    const app = workflow.compile();

    console.log("User: Research LangGraph and write a blog post.");
    const result = await app.invoke({
        messages: [new HumanMessage("Research LangGraph and write a blog post.")],
    });

    console.log("\nFinal Result (Last Message):");
    console.log(result.messages[result.messages.length - 1].content);
};

run().catch(console.error);
