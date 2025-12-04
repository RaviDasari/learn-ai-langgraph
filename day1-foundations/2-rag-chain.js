import { getModel } from "../common/model.js";
import { TextLoader } from "langchain/document_loaders/fs/text";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { OpenAIEmbeddings } from "@langchain/openai";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { createRetrievalChain } from "langchain/chains/retrieval";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const run = async () => {
    // 1. Load Document
    const loader = new TextLoader(path.join(__dirname, "info.txt"));
    const docs = await loader.load();

    // 2. Split Document
    const splitter = new RecursiveCharacterTextSplitter({
        chunkSize: 200,
        chunkOverlap: 20,
    });
    const splitDocs = await splitter.splitDocuments(docs);

    // 3. Create Vector Store
    const embeddings = new OpenAIEmbeddings();
    const vectorStore = await MemoryVectorStore.fromDocuments(
        splitDocs,
        embeddings
    );

    // 4. Create Retriever
    const retriever = vectorStore.asRetriever();

    // 5. Create Chain
    const model = getModel();

    const prompt = ChatPromptTemplate.fromTemplate(`
    Answer the user's question based only on the following context:
    
    <context>
    {context}
    </context>
    
    Question: {input}
  `);

    const combineDocsChain = await createStuffDocumentsChain({
        llm: model,
        prompt,
    });

    const retrievalChain = await createRetrievalChain({
        retriever,
        combineDocsChain,
    });

    // 6. Invoke
    console.log("Asking: What is LangGraph inspired by?");
    const response = await retrievalChain.invoke({
        input: "What is LangGraph inspired by?",
    });

    console.log("\nAnswer:");
    console.log(response.answer);
};

run().catch(console.error);
