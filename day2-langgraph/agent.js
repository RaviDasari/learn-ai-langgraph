import { getModel } from "../common/model.js";
import { StateGraph, MessagesAnnotation } from "@langchain/langgraph";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import { tool } from "@langchain/core/tools";
import { z } from "zod";
import { HumanMessage } from "@langchain/core/messages";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { OpenAIEmbeddings } from "@langchain/openai";
import { TextLoader } from "langchain/document_loaders/fs/text";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// --- Setup Vector Store (Same as Day 1) ---
const setupVectorStore = async () => {
    const loader = new TextLoader(path.join(__dirname, "../day1-foundations/info.txt"));
    const docs = await loader.load();
    const splitter = new RecursiveCharacterTextSplitter({ chunkSize: 200, chunkOverlap: 20 });
    const splitDocs = await splitter.splitDocuments(docs);
    return await MemoryVectorStore.fromDocuments(splitDocs, new OpenAIEmbeddings());
};

const run = async () => {
    const vectorStore = await setupVectorStore();
    const retriever = vectorStore.asRetriever();

    // --- Define Tools ---
    const lookupPolicy = tool(
        async ({ query }) => {
            const docs = await retriever.invoke(query);
            const content = docs.map((d) => d.pageContent).join("\n\n");
            // console.log("\n[Tool] Retrieved Documents:");
            // console.log(content);
            return content;
        },
        {
            name: "lookup_policy",
            description: "Ask this tool questions about LangGraph.",
            schema: z.object({
                query: z.string().describe("The search query"),
            }),
        }
    );

    const tools = [lookupPolicy];
    const toolNode = new ToolNode(tools);

    // --- Define Graph ---
    const model = getModel().bindTools(tools);

    // Node: Agent (Calls Model)
    const callModel = async (state) => {
        const response = await model.invoke(state.messages);
        return { messages: [response] };
    };

    // Edge: Should Continue?
    const shouldContinue = (state) => {
        const lastMessage = state.messages[state.messages.length - 1];
        if (lastMessage.tool_calls?.length) {
            return "tools";
        }
        return "__end__";
    };

    const workflow = new StateGraph(MessagesAnnotation)
        .addNode("agent", callModel)
        .addNode("tools", toolNode)
        .addEdge("__start__", "agent")
        .addConditionalEdges("agent", shouldContinue)
        .addEdge("tools", "agent");

    const app = workflow.compile();

    // --- Run ---
    console.log("User: What is LangGraph inspired by? What is the latest version available?");
    const result = await app.invoke({
        messages: [new HumanMessage("What is LangGraph inspired by? What is the latest version available?")],
    });

    console.log("\nFinal Response:");
    console.log(result.messages[result.messages.length - 1].content);
};

run().catch(console.error);
