import os

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import CharacterTextSplitter

load_dotenv()


def main():
    print("Hello from RAG Pipelines demo!")

    print("Loading document...")
    loader = TextLoader(
        "C:\\Users\\ppjai\\Desktop\\LangGraph-LangChain-Course\\MediumBlog.txt"
    )
    document = loader.load()

    print("Splitting document into chunks...")

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(f"Number of chunks: {len(texts)}")

    print("Creating embeddings and storing in Pinecone...")
    embeddings = OllamaEmbeddings(model="qwen3-embedding")
    PineconeVectorStore.from_documents(
        texts, embeddings, index_name=os.environ.get("INDEX_NAME")
    )

    print("Embeddings created....")


if __name__ == "__main__":
    main()
