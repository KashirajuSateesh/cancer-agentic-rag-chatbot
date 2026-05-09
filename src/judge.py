import json
from langchain_openai import ChatOpenAI


def evaluate_answer(question, context, answer):
    judge_llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    judge_prompt = f"""
You are evaluating a RAG system answer.

Your goal is NOT to be overly strict.

Guidelines:
- If the answer is MOSTLY supported by the context → give high score (>= 0.7)
- Minor rewording or summarization is OK
- Only mark hallucination_detected = true if the answer contains facts NOT present in context

Return ONLY valid JSON:

{{
  "faithfulness_score": float (0 to 1),
  "answer_relevance_score": float (0 to 1),
  "hallucination_detected": boolean,
  "feedback": string
}}

Question:
{question}

Context:
{context}

Answer:
{answer}
"""

    response = judge_llm.invoke(judge_prompt)

    try:
        return json.loads(response.content)
    except json.JSONDecodeError:
        return {
            "faithfulness_score": 0.0,
            "answer_relevance_score": 0.0,
            "hallucination_detected": True,
            "feedback": "Judge response was not valid JSON."
        }