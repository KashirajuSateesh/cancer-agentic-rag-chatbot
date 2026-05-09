import copy
from rank_bm25 import BM25Okapi

from src.retriever import load_vectorstore


def tokenize(text):
    return text.lower().split()


def get_all_documents_from_faiss(vectorstore):
    documents = []

    for doc_id in vectorstore.docstore._dict:
        documents.append(vectorstore.docstore._dict[doc_id])

    return documents


def rerank_docs(query, docs, top_k=6):
    query_terms = set(tokenize(query))
    scored_docs = []

    for doc in docs:
        doc_terms = set(tokenize(doc.page_content))
        overlap_score = len(query_terms.intersection(doc_terms))

        doc.metadata["rerank_score"] = overlap_score
        scored_docs.append((overlap_score, doc))

    scored_docs = sorted(scored_docs, key=lambda x: x[0], reverse=True)

    return [doc for score, doc in scored_docs[:top_k]]


def hybrid_search(query, faiss_k=12, bm25_k=12, final_k=6):
    vectorstore = load_vectorstore()

    # =========================
    # FAISS dense retrieval
    # =========================
    faiss_results = vectorstore.similarity_search_with_score(query, k=faiss_k)

    faiss_docs = []

    for doc, score in faiss_results:
        doc_copy = copy.deepcopy(doc)
        doc_copy.metadata["faiss_score"] = float(score)
        doc_copy.metadata["retrieval_source"] = "faiss"
        faiss_docs.append(doc_copy)

    # =========================
    # BM25 keyword retrieval
    # =========================
    all_docs = get_all_documents_from_faiss(vectorstore)

    tokenized_docs = [tokenize(doc.page_content) for doc in all_docs]
    bm25 = BM25Okapi(tokenized_docs)

    tokenized_query = tokenize(query)
    bm25_scores = bm25.get_scores(tokenized_query)

    bm25_ranked = sorted(
        zip(all_docs, bm25_scores),
        key=lambda x: x[1],
        reverse=True
    )[:bm25_k]

    bm25_docs = []

    for doc, score in bm25_ranked:
        doc_copy = copy.deepcopy(doc)
        doc_copy.metadata["bm25_score"] = float(score)
        doc_copy.metadata["retrieval_source"] = "bm25"
        bm25_docs.append(doc_copy)

    # =========================
    # Merge duplicates
    # =========================
    merged_docs = []
    seen = set()

    for doc in faiss_docs + bm25_docs:
        source = doc.metadata.get("source", "")
        page = doc.metadata.get("page", "")
        content_preview = doc.page_content[:100]

        key = (source, page, content_preview)

        if key not in seen:
            # ensure missing scores are available
            doc.metadata.setdefault("faiss_score", None)
            doc.metadata.setdefault("bm25_score", None)
            doc.metadata.setdefault("rerank_score", 0)

            merged_docs.append(doc)
            seen.add(key)

    # =========================
    # Rerank final chunks
    # =========================
    return rerank_docs(query, merged_docs, top_k=final_k)


if __name__ == "__main__":
    query = "Medicare out-of-pocket costs for cancer patients"

    results = hybrid_search(query)

    for i, doc in enumerate(results):
        print(f"\n--- Result {i + 1} ---")
        print(doc.page_content[:500])
        print(doc.metadata)