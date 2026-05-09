import json
import os

from src.hybrid_retriever import hybrid_search
from src.retrieval_metrics import evaluate_retrieval
from src.rag_chain import rewrite_query_with_llm


EVAL_FILE = "data/eval_questions.json"


def make_doc_id(doc):
    source = doc.metadata.get("source", "")
    file_name = os.path.basename(source)
    page = doc.metadata.get("page", "N/A")

    return f"{file_name}::{page}"


def run_evaluation(k=5):
    with open(EVAL_FILE, "r", encoding="utf-8") as f:
        eval_questions = json.load(f)

    all_scores = []

    for item in eval_questions:
        question = item["question"]
        relevant_ids = item["relevant_pages"]

        rewritten_query = rewrite_query_with_llm(question)

        print("REWRITTEN QUERY:", rewritten_query)

        docs = hybrid_search(
            question,
            faiss_k=12,
            bm25_k=12,
            final_k=k
        )

        retrieved_ids = [make_doc_id(doc) for doc in docs]
        retrieved_ids = list(dict.fromkeys(retrieved_ids))

        scores = evaluate_retrieval(
            retrieved_ids=retrieved_ids,
            relevant_ids=relevant_ids,
            k=k
        )

        all_scores.append(scores)

        print("\nQUESTION:", question)
        print("RELEVANT:", relevant_ids)
        print("RETRIEVED:", retrieved_ids)
        print("SCORES:", scores)

    avg_scores = {}

    for key in all_scores[0].keys():
        avg_scores[key] = sum(score[key] for score in all_scores) / len(all_scores)

    print("\n==============================")
    print("AVERAGE RETRIEVAL METRICS")
    print("==============================")

    for key, value in avg_scores.items():
        print(f"{key}: {value:.3f}")


if __name__ == "__main__":
    run_evaluation(k=10)