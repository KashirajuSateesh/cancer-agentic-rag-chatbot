import os
import time
import streamlit as st
import streamlit as st
from dotenv import load_dotenv

from src.rag_chain import ask_question


load_dotenv()


st.set_page_config(
    page_title="Cancer Research RAG Chatbot",
    layout="wide"
)


def render_pipeline(active_step):
    steps = [
        "User Query",
        "PII Masking",
        "SLM Intent",
        "Context Resolve",
        "Query Rewrite",
        "Cache Check",
        "Hybrid Search",
        "Reranking",
        "Context Build",
        "LLM Answer",
        "LLM Judge",
        "Final Response"
    ]

    html = """
    <div style="padding:16px;border-radius:12px;background-color:#111827;margin-bottom:18px;font-family:Arial;">
        <h3 style="color:white;margin-bottom:14px;">Live RAG Pipeline</h3>
        <div style="display:flex;flex-wrap:wrap;align-items:center;gap:8px;">
    """

    for i, step in enumerate(steps):
        if i < active_step:
            color = "#22c55e"
            text_color = "white"
        elif i == active_step:
            color = "#f59e0b"
            text_color = "black"
        else:
            color = "#374151"
            text_color = "white"

        html += f"""
        <div style="
            padding:8px 12px;
            border-radius:999px;
            background-color:{color};
            color:{text_color};
            font-size:13px;
            font-weight:600;
            white-space:nowrap;
        ">
            {step}
        </div>
        """

        if i < len(steps) - 1:
            html += """
            <span style="color:#9ca3af;font-weight:bold;">→</span>
            """

    html += """
        </div>
    </div>
    """

    return html


st.title("🧬 Cancer Research RAG Chatbot")
st.write("Ask questions from your uploaded PDFs")

debug_mode = st.sidebar.checkbox("Enable Debug Mode")
show_pipeline = st.sidebar.checkbox("Show Live Pipeline", value=True)


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "last_docs" not in st.session_state:
    st.session_state.last_docs = []

if "last_rewritten_query" not in st.session_state:
    st.session_state.last_rewritten_query = ""

if "last_contextualized_question" not in st.session_state:
    st.session_state.last_contextualized_question = ""

if "last_user_question" not in st.session_state:
    st.session_state.last_user_question = ""

if "last_eval_info" not in st.session_state:
    st.session_state.last_eval_info = {}


user_input = st.chat_input("Ask a question...")


if user_input:
    conversation_context = {
        "previous_user_question": st.session_state.last_user_question,
        "previous_rewritten_query": st.session_state.last_rewritten_query
    }

    if show_pipeline:

        sidebar_pipeline = st.sidebar.empty()

        for step in range(0, 11):
            sidebar_pipeline.html(render_pipeline(step))
            time.sleep(0.15)

        answer, docs, rewritten_query, eval_info = ask_question(
            user_input,
            conversation_context=conversation_context
        )

        sidebar_pipeline.html(render_pipeline(11))

    else:
        with st.spinner("Thinking..."):
            answer, docs, rewritten_query, eval_info = ask_question(
                user_input,
                conversation_context=conversation_context
            )

    st.session_state.chat_history.append(("user", user_input))
    st.session_state.chat_history.append(("assistant", answer))

    st.session_state.last_docs = docs
    st.session_state.last_user_question = user_input
    st.session_state.last_contextualized_question = eval_info.get(
        "contextualized_question",
        user_input
    )
    st.session_state.last_rewritten_query = rewritten_query
    st.session_state.last_eval_info = eval_info


for role, message in st.session_state.chat_history:
    if role == "user":
        st.chat_message("user").write(message)
    else:
        st.chat_message("assistant").write(message)


latest_answer = ""

if st.session_state.chat_history:
    for role, message in reversed(st.session_state.chat_history):
        if role == "assistant":
            latest_answer = message
            break


if (
    st.session_state.last_docs
    and "I could not find this clearly" not in latest_answer
    and "I could not find relevant information" not in latest_answer
):
    st.subheader("Sources")

    unique_sources = set()

    for doc in st.session_state.last_docs:
        source = doc.metadata.get("source", "Unknown")
        file_name = os.path.basename(source)

        page = doc.metadata.get(
            "page_display",
            doc.metadata.get("page", "N/A")
        )

        unique_sources.add(f"{file_name} (Page {page})")

    for src in sorted(unique_sources):
        st.write(f"- {src}")


if debug_mode:
    st.markdown("## 🛠️ Debug Dashboard")

    eval_info = st.session_state.last_eval_info
    judge_result = eval_info.get("judge_result")

    col1, col2, col3 = st.columns(3)

    col1.metric("Answer Cache", str(eval_info.get("cache_hit")))
    col2.metric("Rewrite Cache", str(eval_info.get("rewrite_cache_hit")))
    col3.metric("Latency", f"{eval_info.get('latency_seconds')}s")

    intent = eval_info.get("intent", "N/A")

    st.markdown(
        f"""
        <div style="
            padding: 10px 14px;
            border-radius: 10px;
            background-color: #eef2ff;
            color: #3730a3;
            font-weight: 700;
            margin-top: 10px;
            margin-bottom: 10px;
            width: fit-content;
        ">
            Intent: {intent}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### 🔁 Query Processing")
    st.info(f"**Original Query:** {st.session_state.last_user_question}")

    st.warning(
        f"**Masked Query:** "
        f"{eval_info.get('masked_question', 'N/A')}"
    )

    st.info(
        f"**Contextualized Query:** "
        f"{st.session_state.last_contextualized_question}"
    )

    st.info(f"**Rewritten Query:** {st.session_state.last_rewritten_query}")

    with st.expander("📌 Pipeline Steps", expanded=True):
        pipeline = eval_info.get("pipeline", [])
        for i, step in enumerate(pipeline, start=1):
            st.write(f"✅ **Step {i}:** {step}")

    st.markdown("### 🧪 LLM-as-Judge Metrics")

    if judge_result:
        j1, j2, j3 = st.columns(3)

        j1.metric(
            "Faithfulness",
            judge_result.get("faithfulness_score")
        )

        j2.metric(
            "Answer Relevance",
            judge_result.get("answer_relevance_score")
        )

        j3.metric(
            "Hallucination",
            str(judge_result.get("hallucination_detected"))
        )

        st.success(f"**Judge Feedback:** {judge_result.get('feedback')}")
    else:
        st.warning("No judge result available.")

    st.markdown("### 🔢 Token Usage")

    t1, t2, t3, t4 = st.columns(4)

    t1.metric(
        "Prompt Tokens",
        eval_info.get("prompt_tokens")
    )

    t2.metric(
        "Completion Tokens",
        eval_info.get("completion_tokens")
    )

    t3.metric(
        "Total Tokens",
        eval_info.get("total_tokens")
    )

    t4.metric(
        "Estimated Cost",
        f"${eval_info.get('estimated_cost')}"
    )

    st.markdown("### 📚 Retrieved Chunks")

    if st.session_state.last_docs:
        for i, doc in enumerate(st.session_state.last_docs):
            source = doc.metadata.get("source", "Unknown")
            file_name = os.path.basename(source)

            page = doc.metadata.get(
                "page_display",
                doc.metadata.get("page", "N/A")
            )

            faiss_score = doc.metadata.get("faiss_score")

            if faiss_score is not None:
                faiss_similarity = round(1 / (1 + faiss_score), 3)
            else:
                faiss_similarity = "N/A"

            with st.expander(
                f"Chunk {i + 1} — {file_name}, Page {page}",
                expanded=False
            ):
                c1, c2, c3, c4 = st.columns(4)

                c1.metric("FAISS Similarity", faiss_similarity)
                c2.metric("BM25 Score", doc.metadata.get("bm25_score", "N/A"))
                c3.metric("Rerank Score", doc.metadata.get("rerank_score", "N/A"))
                c4.metric("Source", doc.metadata.get("retrieval_source", "N/A"))

                st.markdown("**Chunk Preview:**")
                st.write(doc.page_content[:700])
    else:
        st.warning("No retrieved chunks yet.")
