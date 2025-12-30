import { getModel } from "../common/model.js";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";

const run = async () => {
    await askQuestion("What is LangGraph?");
    await askQuestion("What is the latest version of LangGraph?");
    await askQuestion("What is LangGraph inspired by?");
};

async function askQuestion(question) {
    const model = getModel();
    let messages = [
        new SystemMessage("You are a helpful assistant."),
        new HumanMessage(question),
    ];

    console.log('\n\n----------------\n' +question);
    let response = await model.invoke(messages);

    console.log("\nResponse:");
    console.log(response.content);
}

run().catch(console.error);
