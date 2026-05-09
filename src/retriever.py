from dotenv import load_dotenv
import os

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

VECTORSTORE_DIR = "vectorstore/faiss_index"

def load_vectorstore():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vectorstore = FAISS.load_local(
        VECTORSTORE_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vectorstore

def get_retriever():
    vectorstore = load_vectorstore()

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 4}
    )

    return retriever

def test_retriever():
    retriever = get_retriever()

    query = "What is cancer informatics?"

    results = retriever.invoke(query)

    for i, doc in enumerate(results):
        print(f"\n--- Result {i+1} ---")
        print(doc.page_content[:300])
        print("Source:", doc.metadata.get("source"))

if __name__ == "__main__":
    test_retriever()