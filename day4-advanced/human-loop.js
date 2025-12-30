import { getModel } from "../common/model.js";
import { StateGraph, MessagesAnnotation, MemorySaver } from "@langchain/langgraph";
import { HumanMessage, AIMessage, SystemMessage } from "@langchain/core/messages";
import { Command } from "@langchain/langgraph";
import * as readline from "readline";

// Helper function to get user feedback
const getUserFeedback = () => {
    return new Promise((resolve) => {
        const rl = readline.createInterface({
            input: process.stdin,
            output: process.stdout
        });
        
        rl.question("\nYour feedback: ", (answer) => {
            rl.close();
            resolve(answer.trim());
        });
    });
};

const run = async () => {
    const model = getModel();

    // --- Define Nodes ---
    const agent = async (state) => {
        console.log("--- Agent Node ---");
        const messages = [
            new SystemMessage("You are a helpful assistant that drafts tweets. When revising, only output the revised tweet, nothing else. Do not include any explanations or the user's feedback in your response."),
            ...state.messages
        ];
        const response = await model.invoke(messages);
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

    // --- Run 2: Human Approval Loop ---
    let approved = false;
    let result2;
    
    while (!approved) {
        console.log("\n2. Awaiting human review...");
        
        // Get user feedback
        const userFeedback = await getUserFeedback();
        
        // Get current state
        const currentState = await app.getState(config);
        const currentTweet = currentState.values.messages[currentState.values.messages.length - 1].content;
        
        // Ask LLM to interpret user's intent
        console.log("\nAnalyzing your feedback...");
        const analysisPrompt = `You are analyzing user feedback about generated content.

Generated tweet: "${currentTweet}"

User feedback: "${userFeedback}"

Determine if the user is happy with the output or wants to regenerate it.
Respond with ONLY a JSON object in this exact format:
{
  "satisfied": true/false,
  "reason": "brief explanation"
}

If the user expresses approval, satisfaction, or says it's good/ok/fine, set satisfied to true.
If the user requests changes, expresses dissatisfaction, or wants improvements, set satisfied to false.`;

        const analysisResponse = await model.invoke([new HumanMessage(analysisPrompt)]);
        
        // Parse LLM response
        let analysis;
        try {
            const jsonMatch = analysisResponse.content.match(/\{[\s\S]*\}/);
            analysis = JSON.parse(jsonMatch[0]);
        } catch (e) {
            console.log("Could not parse LLM response, asking for clarification...");
            analysis = { satisfied: false, reason: "unclear feedback" };
        }
        
        console.log(`Decision: ${analysis.satisfied ? '✓ User is satisfied' : '↻ User wants changes'} - ${analysis.reason}`);
        
        if (analysis.satisfied) {
            console.log("\n✓ Approved - Continuing execution...");
            result2 = await app.invoke(
                new Command({ resume: "Approved" }),
                config
            );
            console.log("\nFinal Result:");
            console.log(result2.messages[result2.messages.length - 1].content);
            approved = true;
        } else {
            console.log(`\n↻ Regenerating with feedback: "${userFeedback}"`);
            
            // Directly ask the model to revise the tweet with full context
            console.log("\nAsking LLM to regenerate...");
            const revisionMessages = [
                new SystemMessage("You are a helpful assistant that drafts tweets. When revising, only output the revised tweet text, nothing else. Do not include any explanations, prefixes like 'Revised tweet:', or quotes."),
                new HumanMessage(`Here is the original tweet:\n\n"${currentTweet}"\n\nPlease revise it based on this feedback: ${userFeedback}`)
            ];
            const revisedResponse = await model.invoke(revisionMessages);
            const revisedTweet = revisedResponse.content;
            
            // Update the state with the feedback and the new AI response
            await app.updateState(config, {
                messages: [
                    new HumanMessage(`Please revise the tweet with this feedback: ${userFeedback}`),
                    new AIMessage(revisedTweet)
                ]
            });
            
            console.log(`\n[Paused] Agent revised: "${revisedTweet}"`);
            
            // Loop continues to ask for approval again
        }
    }
};

run().catch(console.error);