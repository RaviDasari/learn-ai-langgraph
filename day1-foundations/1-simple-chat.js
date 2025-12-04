import { getModel } from "../common/model.js";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";

const run = async () => {
    const model = getModel();

    const messages = [
        new SystemMessage("You are a helpful assistant."),
        new HumanMessage("What is LangGraph?"),
    ];

    console.log("Sending request to LLM...");
    const response = await model.invoke(messages);

    console.log("\nResponse:");
    console.log(response.content);
};

run().catch(console.error);
