import { ChatOpenAI } from "@langchain/openai";
import { config } from "dotenv";

config();

/**
 * Returns a configured ChatOpenAI instance.
 * @param {Object} options - Optional overrides for the model.
 * @returns {ChatOpenAI}
 */
export const getModel = (options = {}) => {
  return new ChatOpenAI({
    modelName: process.env.MODEL_NAME || "gpt-4o",
    temperature: 0,
    ...options,
  });
};
