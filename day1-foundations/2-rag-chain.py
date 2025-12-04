import sys
import os
import asyncio

# Add the parent directory to sys.path to allow imports from common
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.model import get_model
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

async def run():
    # 1. Load Document
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "info.txt")
    loader = TextLoader(file_path)
    docs = await asyncio.to_thread(loader.load)

    # 2. Split Document
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20,
    )
    split_docs = splitter.split_documents(docs)

    # 3. Create Vector Store
    embeddings = OpenAIEmbeddings()
    vector_store = await asyncio.to_thread(FAISS.from_documents, split_docs, embeddings)

    # 4. Create Retriever
    retriever = vector_store.as_retriever()

    # 5. Create Chain
    model = get_model()

    prompt = ChatPromptTemplate.from_template("""
    Answer the user's question based only on the following context:
    
    <context>
    {context}
    </context>
    
    Question: {input}
    """)

    combine_docs_chain = create_stuff_documents_chain(
        llm=model,
        prompt=prompt,
    )

    retrieval_chain = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=combine_docs_chain,
    )

    # 6. Invoke
    print("Asking: What is LangGraph inspired by?")
    response = await retrieval_chain.ainvoke({
        "input": "What is LangGraph inspired by?",
    })

    print("\nAnswer:")
    print(response["answer"])

if __name__ == "__main__":
    asyncio.run(run())
