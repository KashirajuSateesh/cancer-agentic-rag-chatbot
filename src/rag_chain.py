import time

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import tiktoken

from src.hybrid_retriever import hybrid_search
from src.logger import log_rag_event
from src.pii import mask_pii
from src.cache import (
    get_cached_rewrite,
    set_cached_rewrite,
    get_cached_answer,
    set_cached_answer
)
from src.judge import evaluate_answer
from src.prompts import (
    INTENT_CLASSIFIER_PROMPT,
    CONTEXTUALIZE_QUESTION_PROMPT,
    QUERY_REWRITE_PROMPT,
    ANSWER_GENERATION_PROMPT
)


load_dotenv()


def get_llm():
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )


def get_slm():
    return ChatOpenAI(
        model="gpt-4.1-nano",
        temperature=0
    )


def count_tokens(text, model="gpt-4o-mini"):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


def classify_intent_with_slm(question):
    slm = get_slm()

    prompt = INTENT_CLASSIFIER_PROMPT.format(
        question=question
    )

    response = slm.invoke(prompt)
    return response.content.strip().lower()


def contextualize_question_with_slm(question, conversation_context):
    slm = get_slm()

    prompt = CONTEXTUALIZE_QUESTION_PROMPT.format(
        previous_user_question=conversation_context.get(
            "previous_user_question",
            ""
        ),
        previous_rewritten_query=conversation_context.get(
            "previous_rewritten_query",
            ""
        ),
        question=question
    )

    response = slm.invoke(prompt)
    return response.content.strip()


def rewrite_query_with_llm(question):
    slm = get_slm()

    rewrite_prompt = QUERY_REWRITE_PROMPT.format(
        question=question
    )

    response = slm.invoke(rewrite_prompt)
    return response.content.strip()


def get_prompt():
    return ChatPromptTemplate.from_template(
        ANSWER_GENERATION_PROMPT
    )


def get_general_response():
    return (
        "Hi! I am a Cancer Research RAG chatbot.\n\n"
        "I can answer questions based on your uploaded cancer-related PDFs.\n\n"
        "Try asking:\n"
        "- What is cancer informatics?\n"
        "- How does Medicare cover cancer screening?\n"
        "- What are cancer risk factors?"
    )


def ask_question(user_question, conversation_context=None):
    start_time = time.time()

    original_question = user_question
    masked_question = mask_pii(user_question)
    conversation_context = conversation_context or {}

    # =========================
    # SLM INTENT CLASSIFICATION
    # =========================
    intent = classify_intent_with_slm(masked_question)

    if intent in ["greeting", "meta_question"]:
        eval_info = {
            "cache_hit": False,
            "rewrite_cache_hit": False,
            "rewritten_query_source": "not_applicable",
            "contextualized_question": masked_question,
            "context_used": False,
            "context_source": "not_applicable",
            "context_model": None,
            "intent": intent,
            "intent_model": "gpt-4.1-nano",
            "masked_question": masked_question,
            "judge_result": None,
            "latency_seconds": round(time.time() - start_time, 2),
            "pipeline": [
                "pii_masking",
                "slm_intent_classification",
                "general_response"
            ]
        }

        return get_general_response(), [], original_question, eval_info

    previous_user_question = mask_pii(
        conversation_context.get("previous_user_question", "")
    )
    previous_rewritten_query = conversation_context.get(
        "previous_rewritten_query",
        ""
    )
    context_used = bool(
        previous_user_question.strip() or previous_rewritten_query.strip()
    )

    if context_used:
        contextualized_question = contextualize_question_with_slm(
            masked_question,
            {
                "previous_user_question": previous_user_question,
                "previous_rewritten_query": previous_rewritten_query
            }
        )
        context_source = "last_turn"
        context_model = "gpt-4.1-nano"
    else:
        contextualized_question = masked_question
        context_source = "none"
        context_model = None

    # =========================
    # SLM QUERY REWRITE + CACHE
    # =========================
    cached_rewrite = get_cached_rewrite(contextualized_question)

    if cached_rewrite:
        rewritten_query = cached_rewrite
        rewrite_cache_hit = True
        rewritten_query_source = "rewrite_cache"
    else:
        rewritten_query = rewrite_query_with_llm(contextualized_question)
        set_cached_rewrite(contextualized_question, rewritten_query)
        rewrite_cache_hit = False
        rewritten_query_source = "slm_generated"

    # =========================
    # ANSWER CACHE CHECK
    # =========================
    cached_answer = get_cached_answer(rewritten_query)

    if cached_answer:
        latency = round(time.time() - start_time, 2)

        answer, docs, rewritten_query, old_eval_info = cached_answer
        pipeline_skipped = [
            "hybrid_retrieval",
            "llm_generation",
            "llm_as_judge"
        ]

        if rewrite_cache_hit:
            pipeline_skipped.insert(0, "slm_query_rewrite")

        eval_info = {
            **old_eval_info,
            "cache_hit": True,
            "cache_type": "answer_cache",
            "rewrite_cache_hit": rewrite_cache_hit,
            "rewritten_query_source": rewritten_query_source,
            "contextualized_question": contextualized_question,
            "context_used": context_used,
            "context_source": context_source,
            "context_model": context_model,
            "latency_seconds": latency,
            "pipeline_skipped": pipeline_skipped
        }

        log_rag_event({
            "original_question": original_question,
            "masked_question": masked_question,
            "contextualized_question": contextualized_question,
            "context_used": context_used,
            "context_source": context_source,
            "context_model": context_model,
            "intent": intent,
            "rewritten_query": rewritten_query,
            "retrieved_chunks": len(docs),
            "sources": [
                {
                    "source": doc.metadata.get("source"),
                    "page": doc.metadata.get("page"),
                    "page_display": doc.metadata.get("page_display"),
                    "faiss_score": doc.metadata.get("faiss_score"),
                    "bm25_score": doc.metadata.get("bm25_score"),
                    "rerank_score": doc.metadata.get("rerank_score")
                }
                for doc in docs
            ],
            "model": "cache",
            "latency_seconds": latency,
            "cache_hit": True,
            "cache_type": "answer_cache",
            "rewrite_cache_hit": rewrite_cache_hit,
            "rewritten_query_source": rewritten_query_source
        })

        return answer, docs, rewritten_query, eval_info

    # =========================
    # HYBRID RETRIEVAL
    # =========================
    docs = hybrid_search(
        rewritten_query,
        faiss_k=12,
        bm25_k=12,
        final_k=6
    )

    context = "\n\n".join([doc.page_content for doc in docs])
    prompt_tokens = count_tokens(context + rewritten_query)

    if not context.strip():
        eval_info = {
            "cache_hit": False,
            "rewrite_cache_hit": rewrite_cache_hit,
            "rewritten_query_source": rewritten_query_source,
            "contextualized_question": contextualized_question,
            "context_used": context_used,
            "context_source": context_source,
            "context_model": context_model,
            "intent": intent,
            "intent_model": "gpt-4.1-nano",
            "rewrite_model": "gpt-4.1-nano",
            "masked_question": masked_question,
            "judge_result": None,
            "latency_seconds": round(time.time() - start_time, 2),
            "pipeline": [
                "pii_masking",
                "slm_intent_classification",
                "slm_contextualization",
                "slm_query_rewrite",
                "hybrid_retrieval",
                "empty_context"
            ]
        }

        return (
            "I could not find relevant information in the documents.",
            [],
            rewritten_query,
            eval_info
        )

    # =========================
    # LLM ANSWER GENERATION
    # =========================
    llm = get_llm()
    prompt = get_prompt()
    chain = prompt | llm

    response = chain.invoke({
        "context": context,
        "question": rewritten_query
    })

    answer = response.content

    completion_tokens = count_tokens(answer)
    total_tokens = prompt_tokens + completion_tokens
    estimated_cost = round((total_tokens / 1_000_000) * 0.60, 6)

    # =========================
    # LLM-AS-JUDGE
    # =========================
    judge_result = evaluate_answer(
        question=rewritten_query,
        context=context,
        answer=answer
    )

    latency = round(time.time() - start_time, 2)

    eval_info = {
        "cache_hit": False,
        "rewrite_cache_hit": rewrite_cache_hit,
        "rewritten_query_source": rewritten_query_source,
        "contextualized_question": contextualized_question,
        "context_used": context_used,
        "context_source": context_source,
        "context_model": context_model,
        "intent": intent,
        "intent_model": "gpt-4.1-nano",
        "rewrite_model": "gpt-4.1-nano",
        "masked_question": masked_question,
        "answer_model": "gpt-4o-mini",
        "judge_result": judge_result,
        "latency_seconds": latency,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "estimated_cost": estimated_cost,
        "context_characters": len(context),
        "retrieved_chunks": len(docs),
        "pipeline": [
            "pii_masking",
            "slm_intent_classification",
            "slm_contextualization",
            "slm_query_rewrite",
            "hybrid_retrieval",
            "reranking",
            "context_building",
            "llm_generation",
            "llm_as_judge"
        ]
    }

    log_rag_event({
        "original_question": original_question,
        "masked_question": masked_question,
        "contextualized_question": contextualized_question,
        "context_used": context_used,
        "context_source": context_source,
        "context_model": context_model,
        "intent": intent,
        "intent_model": "gpt-4.1-nano",
        "rewrite_model": "gpt-4.1-nano",
        "answer_model": "gpt-4o-mini",
        "rewritten_query": rewritten_query,
        "retrieved_chunks": len(docs),
        "sources": [
            {
                "source": doc.metadata.get("source"),
                "page": doc.metadata.get("page"),
                "page_display": doc.metadata.get("page_display"),
                "faiss_score": doc.metadata.get("faiss_score"),
                "bm25_score": doc.metadata.get("bm25_score"),
                "rerank_score": doc.metadata.get("rerank_score")
            }
            for doc in docs
        ],
        "model": "gpt-4o-mini",
        "latency_seconds": latency,
        "cache_hit": False,
        "rewrite_cache_hit": rewrite_cache_hit,
        "rewritten_query_source": rewritten_query_source,
        "judge_result": judge_result
    })

    result = (answer, docs, rewritten_query, eval_info)

    set_cached_answer(rewritten_query, result)

    return result


if __name__ == "__main__":
    question = "What is cancer informatics?"

    answer, docs, rewritten_query, eval_info = ask_question(question)

    print("\nREWRITTEN QUERY:\n")
    print(rewritten_query)

    print("\nANSWER:\n")
    print(answer)

    print("\nEVAL INFO:\n")
    print(eval_info)

    print("\nSOURCES:\n")
    for doc in docs:
        print(doc.metadata.get("source"), doc.metadata.get("page"))
