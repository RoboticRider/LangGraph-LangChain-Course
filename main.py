import os
from operator import itemgetter

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()


def main():

    print("Initializing the components...")

    embeddings = OllamaEmbeddings(model="qwen3-embedding")

    llm = ChatOllama(model="qwen3:0.6b", temperature=0)

    vectorStore = PineconeVectorStore(
        embedding=embeddings, index_name=os.environ.get("INDEX_NAME")
    )

    retriever = vectorStore.as_retriever(search_kwargs={"k": 3})

    promptTemplate = ChatPromptTemplate.from_template(
        """Answer the Question based only on the folllowing context:-
        {context}

        Question:- {question}

        Provide a detailed answer:-"""
    )

    def format_docs(docs):
        """Formats the retrieved documents into a Single string."""
        return "\n\n".join([doc.page_content for doc in docs])


if __name__ == "__main__":
    main()
