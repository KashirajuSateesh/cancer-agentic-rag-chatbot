import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

PDF_DIR = "data/pdfs"
VECTORSTORE_DIR = "vectorstore/faiss_index"

def load_pdfs():
    print("Loading PDFs...")
    loader = PyPDFDirectoryLoader(PDF_DIR)
    documents = loader.load()
    print(f"Total pages loaded: {len(documents)}")
    return documents

def chunk_documents(documents):
    print("Chunking documents...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = splitter.split_documents(documents)

    for chunk in chunks:
        if "page" in chunk.metadata:
            chunk.metadata["page_display"] = chunk.metadata["page"] + 1
        else:
            chunk.metadata["page_display"] = "N/A"

    print(f"Total chunks created: {len(chunks)}")
    return chunks

def create_vectorstore(chunks):
    print("Creating embeddings and FAISS index...")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    vectorstore.save_local(VECTORSTORE_DIR)
    print(f"Vectorstore saved at: {VECTORSTORE_DIR}")

def main():
    documents = load_pdfs()
    chunks = chunk_documents(documents)
    create_vectorstore(chunks)
    print("Ingestion completed successfully.")

if __name__ == "__main__":
    main()