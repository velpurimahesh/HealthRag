import os
import streamlit as st

from utils.vector_db import get_collection
from utils.retriever import semantic_search
from utils.prompt_builder import build_prompt
from utils.openrouter_llm import generate_answer
from build_knowledge import build_knowledge_base


st.set_page_config(
    page_title="HealthRAG | Healthcare Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# PROFESSIONAL LIGHT THEME
# ============================================================

st.markdown(
    """
<style>
:root {
    --primary: #2563eb;
    --text: #172033;
    --muted: #64748b;
    --border: #e2e8f0;
}

.stApp {
    background: #f6f8fc !important;
    color: var(--text) !important;
}

[data-testid="stHeader"] {
    background: #f6f8fc !important;
}

[data-testid="stToolbar"] {
    background: transparent !important;
}

.main {
    background: #f6f8fc !important;
}

.main .block-container {
    max-width: 1180px;
    padding-top: 1.2rem !important;
    padding-bottom: 7rem !important;
}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid var(--border) !important;
}

section[data-testid="stSidebar"] > div {
    background: #ffffff !important;
}

section[data-testid="stSidebar"] * {
    color: var(--text);
}

.sidebar-brand {
    text-align: center;
    padding: 12px 8px 22px 8px;
}

.sidebar-brand-icon {
    width: 54px;
    height: 54px;
    margin: 0 auto 10px auto;
    border-radius: 16px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: linear-gradient(135deg, #2563eb, #4f8df7);
    color: white !important;
    font-size: 28px;
    box-shadow: 0 8px 20px rgba(37, 99, 235, 0.18);
}

.sidebar-brand-title {
    font-size: 23px;
    font-weight: 800;
    color: #172033 !important;
}

.sidebar-brand-subtitle {
    margin-top: 4px;
    font-size: 12px;
    color: #64748b !important;
}

.sidebar-section-title {
    margin: 18px 0 8px 0;
    font-size: 12px;
    font-weight: 800;
    color: #475569 !important;
    text-transform: uppercase;
    letter-spacing: 0.7px;
}

.sidebar-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 13px;
    margin-top: 8px;
}

.sidebar-ready {
    display: flex;
    align-items: center;
    gap: 8px;
    color: #15803d !important;
    font-weight: 700;
    font-size: 13px;
}

.ready-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #22c55e;
    box-shadow: 0 0 0 4px #dcfce7;
}

.sidebar-small {
    margin-top: 7px;
    color: #64748b !important;
    font-size: 12px;
    line-height: 1.55;
}

.disclaimer-card {
    background: #fffbeb;
    border: 1px solid #fde68a;
    border-radius: 12px;
    padding: 12px;
    color: #92400e !important;
    font-size: 11px;
    line-height: 1.55;
}

section[data-testid="stSidebar"] .stButton > button {
    width: 100%;
    min-height: 42px;
    border-radius: 11px !important;
    border: 1px solid #dbe3ef !important;
    background: #ffffff !important;
    color: #1e3a8a !important;
    font-weight: 700 !important;
}

section[data-testid="stSidebar"] .stButton > button:hover {
    border-color: #93c5fd !important;
    background: #eff6ff !important;
    color: #1d4ed8 !important;
}

/* TOP BAR */
.topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: #ffffff;
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 13px 18px;
    margin-bottom: 20px;
    box-shadow: 0 2px 10px rgba(15, 23, 42, 0.03);
}

.topbar-left {
    display: flex;
    align-items: center;
    gap: 11px;
}

.topbar-icon {
    width: 38px;
    height: 38px;
    border-radius: 11px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: #eff6ff;
    color: #2563eb !important;
    font-size: 20px;
}

.topbar-title {
    font-size: 16px;
    font-weight: 800;
    color: #172033 !important;
}

.topbar-subtitle {
    font-size: 11px;
    color: #64748b !important;
    margin-top: 2px;
}

.secure-pill {
    padding: 6px 10px;
    border-radius: 999px;
    background: #ecfdf5;
    color: #047857 !important;
    border: 1px solid #bbf7d0;
    font-size: 11px;
    font-weight: 700;
}

/* WELCOME */
.welcome-card {
    background: #ffffff;
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 48px 28px 34px 28px;
    text-align: center;
    box-shadow: 0 5px 22px rgba(15, 23, 42, 0.04);
}

.welcome-icon {
    width: 72px;
    height: 72px;
    margin: 0 auto 16px auto;
    border-radius: 20px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: linear-gradient(135deg, #eff6ff, #dbeafe);
    font-size: 34px;
}

.welcome-title {
    color: #172033 !important;
    font-size: 32px;
    line-height: 1.15;
    font-weight: 800;
}

.welcome-subtitle {
    max-width: 620px;
    margin: 10px auto 0 auto;
    color: #64748b !important;
    font-size: 15px;
    line-height: 1.6;
}

/* EXAMPLES */
.example-heading {
    text-align: center;
    margin: 24px 0 12px 0;
    color: #475569 !important;
    font-size: 12px;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}

.example-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 13px;
    padding: 14px 15px;
    margin-bottom: 10px;
    color: #334155 !important;
    font-size: 13px;
    line-height: 1.45;
    min-height: 47px;
    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.025);
}

/* CHAT */
[data-testid="stChatMessage"] {
    border: 1px solid #e2e8f0 !important;
    border-radius: 16px !important;
    padding: 10px 14px !important;
    margin: 9px 0 !important;
    background: #ffffff !important;
    box-shadow: 0 2px 9px rgba(15, 23, 42, 0.025);
}

[data-testid="stChatMessageContent"] {
    color: #172033 !important;
}

[data-testid="stChatMessageContent"] p,
[data-testid="stChatMessageContent"] li,
[data-testid="stChatMessageContent"] span {
    color: #172033 !important;
    line-height: 1.65 !important;
}

/* CHAT INPUT */
[data-testid="stChatInput"] {
    background: transparent !important;
}

[data-testid="stChatInput"] > div {
    background: #ffffff !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 16px !important;
    box-shadow: 0 8px 25px rgba(15, 23, 42, 0.08) !important;
}

[data-testid="stChatInput"] textarea {
    color: #172033 !important;
    background: #ffffff !important;
    caret-color: #2563eb !important;
    -webkit-text-fill-color: #172033 !important;
    font-size: 14px !important;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: #94a3b8 !important;
    opacity: 1 !important;
}

/* SOURCES */
.source-box {
    margin-top: 12px;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-left: 4px solid #2563eb;
    border-radius: 10px;
    padding: 10px 12px;
    color: #475569 !important;
    font-size: 11px;
    line-height: 1.5;
}

.footer-note {
    text-align: center;
    color: #94a3b8 !important;
    font-size: 10px;
    margin-top: 20px;
}

@media (max-width: 768px) {
    .main .block-container {
        padding: 0.7rem 0.8rem 6rem 0.8rem !important;
    }

    .welcome-card {
        padding: 34px 18px 26px 18px;
    }

    .welcome-title {
        font-size: 25px;
    }

    .secure-pill {
        display: none;
    }
}
</style>
""",
    unsafe_allow_html=True,
)

TOP_K = 4


@st.cache_resource(show_spinner=False)
def load_collection():
    try:
        collection = get_collection()

        # If the deployed/local Chroma collection is empty,
        # automatically build it from the PDFs in data/.
        if collection.count() == 0:

            data_folder = "data"
            pdf_files = []

            if os.path.exists(data_folder):
                pdf_files = [
                    file
                    for file in os.listdir(data_folder)
                    if file.lower().endswith(".pdf")
                ]

            if pdf_files:
                with st.spinner(
                    f"Building healthcare knowledge base from {len(pdf_files)} PDFs..."
                ):
                    build_knowledge_base()

                # Reload after the builder recreates the Chroma collection.
                collection = get_collection()

        return collection

    except Exception as e:
        print("=" * 60)
        print("KNOWLEDGE BASE ERROR")
        print("=" * 60)
        print(type(e).__name__)
        print(repr(e))
        print("=" * 60)
        return None


collection = load_collection()


if "messages" not in st.session_state:
    st.session_state.messages = []


def new_chat():
    st.session_state.messages = []


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-brand-icon">🩺</div>
            <div class="sidebar-brand-title">HealthRAG</div>
            <div class="sidebar-brand-subtitle">
                Healthcare Information Assistant
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.button(
        "＋  New conversation",
        on_click=new_chat,
        use_container_width=True,
    )

    st.markdown(
        '<div class="sidebar-section-title">Knowledge Base</div>',
        unsafe_allow_html=True,
    )

    if collection is not None and collection.count() > 0:
        st.markdown(
            """
            <div class="sidebar-card">
                <div class="sidebar-ready">
                    <span class="ready-dot"></span>
                    Knowledge base ready
                </div>
                <div class="sidebar-small">
                    Healthcare documents are available for semantic search.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="sidebar-card">
                <div class="sidebar-ready" style="color:#b91c1c !important;">
                    <span class="ready-dot"
                          style="background:#ef4444;"></span>
                    Knowledge base unavailable
                </div>
                <div class="sidebar-small">
                    Check the vector database configuration.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="sidebar-section-title">About</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="sidebar-small">
            HealthRAG searches connected healthcare documents and
            uses an AI model to produce concise, document-grounded
            responses.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="sidebar-section-title">Important</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="disclaimer-card">
            This assistant provides general information from its
            knowledge base. It does not replace a qualified healthcare
            professional, diagnosis, or treatment.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# TOP BAR
# ============================================================

st.markdown(
    """
    <div class="topbar">
        <div class="topbar-left">
            <div class="topbar-icon">🩺</div>
            <div>
                <div class="topbar-title">HealthRAG Assistant</div>
                <div class="topbar-subtitle">
                    Document-grounded healthcare information
                </div>
            </div>
        </div>
        <div class="secure-pill">● Knowledge base connected</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# WELCOME SCREEN
# ============================================================

if not st.session_state.messages:

    st.markdown(
        """
        <div class="welcome-card">
            <div class="welcome-icon">🩺</div>
            <div class="welcome-title">
                How can I help you today?
            </div>
            <div class="welcome-subtitle">
                Ask a healthcare question and I’ll search the available
                trusted documents to provide a clear, relevant answer.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="example-heading">Try an example</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            '<div class="example-card">💙 What are the common symptoms of diabetes?</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="example-card">❤️ How can hypertension be prevented?</div>',
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            '<div class="example-card">🩸 What are the symptoms of anemia?</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="example-card">🛡️ What are the recommended ways to prevent infection?</div>',
            unsafe_allow_html=True,
        )


# ============================================================
# CHAT HISTORY
# ============================================================

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ============================================================
# CHAT INPUT
# ============================================================

user_question = st.chat_input("Ask a healthcare question...")


# ============================================================
# PROCESS QUESTION
# ============================================================

if user_question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_question,
        }
    )

    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):

        if collection is None:

            answer = (
                "I couldn't connect to the healthcare knowledge base. "
                "Please check your vector database configuration."
            )

            st.error(answer)

        else:

            try:
                retrieved_chunks = semantic_search(
                    collection,
                    user_question,
                    top_k=TOP_K,
                )

                rag_prompt = build_prompt(
                    user_question,
                    retrieved_chunks,
                )

                result = generate_answer(rag_prompt)

                if isinstance(result, str):
                    answer = result.strip()
                    if answer:
                        st.markdown(answer)
                else:
                    answer_parts = []
                    answer_placeholder = st.empty()

                    for part in result:
                        if part is None:
                            continue

                        answer_parts.append(str(part))

                        answer_placeholder.markdown(
                            "".join(answer_parts)
                        )

                    answer = "".join(answer_parts).strip()

                if not answer:
                    answer = (
                        "I couldn't generate an answer for that question. "
                        "Please try asking it in a different way."
                    )
                    st.warning(answer)

                if retrieved_chunks:
                    sources = []

                    for item in retrieved_chunks:
                        if isinstance(item, dict):
                            source = item.get(
                                "source",
                                "Knowledge base",
                            )
                        else:
                            source = "Knowledge base"

                        if source not in sources:
                            sources.append(source)

                    if sources:
                        st.markdown(
                            '<div class="source-box"><b>Sources:</b> '
                            + ", ".join(sources)
                            + "</div>",
                            unsafe_allow_html=True,
                        )

            except Exception as exc:

                answer = (
                    "I’m sorry, but I couldn't complete that request. "
                    "Please try again."
                )

                st.error(answer)

                print("=" * 60)
                print("HEALTHRAG ERROR")
                print(type(exc).__name__)
                print(repr(exc))
                print("=" * 60)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )

    st.rerun()


st.markdown(
    """
    <div class="footer-note">
        HealthRAG • AI-assisted document search • For informational purposes only
    </div>
    """,
    unsafe_allow_html=True,
)
