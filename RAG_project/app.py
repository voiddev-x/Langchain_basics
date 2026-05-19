import streamlit as st
from dotenv import load_dotenv
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DeepLearning RAG",
    page_icon="🧠",
    layout="centered",
)

# ── Minimal safe CSS ──────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&display=swap');

.rag-title {
    font-family: 'Space Mono', monospace;
    font-size: 1.8rem;
    font-weight: 700;
    text-align: center;
    margin-bottom: 0.2rem;
}
.rag-sub {
    text-align: center;
    font-size: 0.85rem;
    opacity: 0.5;
    margin-bottom: 1.5rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
.msg-user {
    background: #f0c040;
    color: #111;
    border-radius: 16px 16px 4px 16px;
    padding: 0.6rem 1rem;
    margin: 0.4rem 0 0.4rem 20%;
    font-size: 0.92rem;
    line-height: 1.5;
}
.msg-bot {
    background: #1e1e2e;
    color: #e8e4dc;
    border: 1px solid #333;
    border-radius: 4px 16px 16px 16px;
    padding: 0.7rem 1rem;
    margin: 0.4rem 20% 0.4rem 0;
    font-size: 0.92rem;
    line-height: 1.65;
}
.msg-label {
    font-size: 0.7rem;
    opacity: 0.45;
    letter-spacing: 0.06em;
    margin-bottom: 0.15rem;
    text-transform: uppercase;
}
</style>
""", unsafe_allow_html=True)


# ── Build RAG chain (cached) ──────────────────────────────────────────────────
@st.cache_resource(show_spinner="🔧 Loading RAG pipeline…")
def build_chain():
    load_dotenv()

    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    embedding_model = MistralAIEmbeddings(model="mistral-embed")

    vectorstore = Chroma(
        persist_directory="chromaM_db",
        embedding_function=embedding_model,
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.7},
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful AI assistant. "
         "Use only the provided context to answer the question. "
         "If the answer is not present in the context, say: answer not found in doc"),
        ("human", "context:{context}\n\nQuestion:{Question}"),
    ])

    def join_doc(docs):
        return "\n\n".join([d.page_content for d in docs])

    chain = (
        RunnableParallel({
            "Question": RunnablePassthrough(),
            "context": retriever | RunnableLambda(join_doc),
        })
        | prompt
        | model
        | StrOutputParser()
    )
    return chain


# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="rag-title">🧠 DeepLearning QA</div>', unsafe_allow_html=True)
st.markdown('<div class="rag-sub">RAG · Gemini 2.5 Flash · Mistral Embeddings · ChromaDB</div>', unsafe_allow_html=True)

st.divider()

# ── Chat history ──────────────────────────────────────────────────────────────
if not st.session_state.messages:
    st.info("Ask anything about your deep learning document to get started.", icon="💡")
else:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="msg-label">You</div><div class="msg-user">{msg["text"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="msg-label">Assistant</div><div class="msg-bot">{msg["text"]}</div>',
                unsafe_allow_html=True,
            )

st.divider()

# ── Input form ────────────────────────────────────────────────────────────────
with st.form(key="query_form", clear_on_submit=True):
    user_input = st.text_input(
        "Your question",
        placeholder="e.g. What is backpropagation?",
    )
    col1, col2 = st.columns([3, 1])
    with col1:
        submitted = st.form_submit_button("Send →", use_container_width=True)
    with col2:
        clear = st.form_submit_button("Clear", use_container_width=True)

if clear:
    st.session_state.messages = []
    st.rerun()

if submitted and user_input.strip():
    st.session_state.messages.append({"role": "user", "text": user_input.strip()})
    with st.spinner("Thinking…"):
        try:
            chain = build_chain()
            response = chain.invoke(user_input.strip())
        except Exception as e:
            response = f"⚠️ Error: {e}"
    st.session_state.messages.append({"role": "bot", "text": response})
    st.rerun()