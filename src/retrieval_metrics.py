import math


def precision_at_k(retrieved_ids, relevant_ids, k):
    retrieved_k = retrieved_ids[:k]

    if k == 0:
        return 0.0

    hits = len(set(retrieved_k).intersection(set(relevant_ids)))
    return hits / k


def recall_at_k(retrieved_ids, relevant_ids, k):
    retrieved_k = retrieved_ids[:k]

    if not relevant_ids:
        return 0.0

    hits = len(set(retrieved_k).intersection(set(relevant_ids)))
    return hits / len(set(relevant_ids))


def hit_rate_at_k(retrieved_ids, relevant_ids, k):
    retrieved_k = retrieved_ids[:k]

    return 1.0 if set(retrieved_k).intersection(set(relevant_ids)) else 0.0


def mrr_at_k(retrieved_ids, relevant_ids, k):
    retrieved_k = retrieved_ids[:k]
    relevant_set = set(relevant_ids)

    for rank, doc_id in enumerate(retrieved_k, start=1):
        if doc_id in relevant_set:
            return 1.0 / rank

    return 0.0


def dcg_at_k(retrieved_ids, relevant_ids, k):
    retrieved_k = retrieved_ids[:k]
    relevant_set = set(relevant_ids)

    dcg = 0.0

    for i, doc_id in enumerate(retrieved_k):
        relevance = 1 if doc_id in relevant_set else 0
        dcg += relevance / math.log2(i + 2)

    return dcg


def ndcg_at_k(retrieved_ids, relevant_ids, k):
    dcg = dcg_at_k(retrieved_ids, relevant_ids, k)

    ideal_hits = min(len(set(relevant_ids)), k)
    ideal_relevances = [1] * ideal_hits

    idcg = 0.0
    for i, relevance in enumerate(ideal_relevances):
        idcg += relevance / math.log2(i + 2)

    if idcg == 0:
        return 0.0

    return dcg / idcg


def evaluate_retrieval(retrieved_ids, relevant_ids, k=10):
    return {
        f"precision@{k}": precision_at_k(retrieved_ids, relevant_ids, k),
        f"recall@{k}": recall_at_k(retrieved_ids, relevant_ids, k),
        f"hit_rate@{k}": hit_rate_at_k(retrieved_ids, relevant_ids, k),
        f"mrr@{k}": mrr_at_k(retrieved_ids, relevant_ids, k),
        f"ndcg@{k}": ndcg_at_k(retrieved_ids, relevant_ids, k),
    }