import asyncio
import os
import ssl
from typing import Any, Dict, List

import certifi
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import tavily_crawl, tavily_map, tavily_search
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Configure SSL context to use certifi certificates
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

embeddings = OllamaEmbeddings(model="qwen3-embedding")


load_dotenv()


def main():
    print("In Ingestion Pipeline File")


if __name__ == "__main__":
    main()
