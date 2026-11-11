import os
import re
from pathlib import Path

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter

st.set_page_config(page_title="Zyro HR Help Desk", page_icon="💼", layout="centered")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
LLM_MODEL = "openai/gpt-oss-120b"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

REFUSAL = (
    "This question is not covered in the Zyro Dynamics HR policy documents. "
    "Please contact the HR team for help."
)

# Blocks requests for one person's own records (policy documents can't contain them)
PERSONAL = re.compile(
    r"\b(tell me|show me|what is|what's|check)\s+my\s+"
    r"(esop|vesting|stock|shares?|salary|ctc|payslip|pay\s*slip|pf\s+balance|"
    r"leave\s+balance|appraisal\s+(rating|score))\b",
    re.IGNORECASE,
)

PROMPT = ChatPromptTemplate.from_template(
    """You are the Zyro Dynamics HR Help Desk assistant.
Answer the employee's question using the policy excerpts below.

Rules:
- If the excerpts contain relevant information, even partial, answer with the specific
  details (numbers, days, limits, eligibility, conditions). Be concise and factual.
- Do not use outside knowledge or invent details.
- Only reply with exactly: "{refusal}" if the excerpts contain nothing relevant,
  or if the question is about company finances, products, competitors, or other companies.
- If the question asks for an individual employee's personal records or account data,
  reply with exactly: "{refusal}"

Policy excerpts:
{context}

Question: {question}
Answer:"""
)

EXAMPLES = [
    "How much Earned Leave can I carry forward?",
    "What medical insurance does the company provide?",
    "What is the notice period to resign?",
    "Am I eligible to work from home?",
    "What is Zoho's leave policy like?",
]


def get_api_key():
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return os.environ.get("GROQ_API_KEY")


def find_pdfs():
    """Look in data/ (including subfolders, any capitalization), then next to app.py."""
    found = []
    if DATA_DIR.exists():
        found = [p for p in DATA_DIR.rglob("*") if p.is_file() and p.suffix.lower() == ".pdf"]
    if not found:
        found = [p for p in BASE_DIR.glob("*") if p.is_file() and p.suffix.lower() == ".pdf"]
    return sorted(found)


@st.cache_resource(show_spinner="Reading the HR policy documents...")
def load_vectorstore():
    pdfs = find_pdfs()
    if not pdfs:
        return None, 0
    docs = []
    for path in pdfs:
        pages = PyPDFLoader(str(path)).load()
        for page in pages:
            page.metadata["source"] = path.name
        docs.extend(pages)
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=150
    ).split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return FAISS.from_documents(chunks, embeddings), len(pdfs)


@st.cache_resource
def get_llm(api_key):
    return ChatGroq(model=LLM_MODEL, temperature=0, api_key=api_key)


def retrieve(store, question):
    sim = store.similarity_search(question, k=6)
    mmr = store.max_marginal_relevance_search(
        question, k=6, fetch_k=30, lambda_mult=0.7
    )
    seen, out = set(), []
    for d in sim + mmr:
        if d.page_content not in seen:
            seen.add(d.page_content)
            out.append(d)
    return out[:10]


def ask(store, llm, question):
    if PERSONAL.search(question):
        return {"answer": REFUSAL, "sources": [], "refused": True}

    docs = retrieve(store, question)
    context = "\n\n".join(
        f"[{d.metadata.get('source', '')}] {d.page_content}" for d in docs
    )
    messages = PROMPT.format_messages(
        context=context, question=question, refusal=REFUSAL
    )
    text = llm.invoke(messages).content.strip()
    refused = text.startswith("This question is not covered")

    sources = []
    if not refused:
        for d in docs[:4]:
            label = f"{d.metadata.get('source', 'unknown')} (page {d.metadata.get('page', 0) + 1})"
            if label not in sources:
                sources.append(label)
    return {"answer": text, "sources": sources, "refused": refused}


def render_message(msg):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.write(f"• {s}")


# ---------- UI ----------
st.title("💼 Zyro HR Help Desk")
st.caption("Ask about leave, benefits, WFH, travel, conduct and other HR policies.")

api_key = get_api_key()
if not api_key:
    st.error("GROQ_API_KEY is missing. Add it in the app's Secrets settings.")
    st.stop()

store, n_pdfs = load_vectorstore()
if store is None:
    st.error("No PDFs found. Put the HR policy PDFs in a 'data' folder next to app.py.")
    with st.expander("Debug: what the app can see"):
        st.write("App folder:", str(BASE_DIR))
        st.write("Files next to app.py:", sorted(p.name for p in BASE_DIR.iterdir()))
        st.write("data/ exists:", DATA_DIR.exists())
        if DATA_DIR.exists():
            st.write("Inside data/:", sorted(p.name for p in DATA_DIR.iterdir()))
    st.stop()

llm = get_llm(api_key)

with st.sidebar:
    st.header("About")
    st.write(
        "Answers come only from the company's HR policy documents. "
        "Questions outside them are politely declined."
    )
    st.caption(f"{n_pdfs} policy documents loaded")
    st.subheader("Try asking")
    for ex in EXAMPLES:
        if st.button(ex, use_container_width=True):
            st.session_state["pending"] = ex
    if st.button("Clear chat", use_container_width=True):
        st.session_state["messages"] = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for m in st.session_state["messages"]:
    render_message(m)

typed = st.chat_input("Ask an HR policy question...")
question = typed or st.session_state.pop("pending", None)

if question:
    user_msg = {"role": "user", "content": question}
    st.session_state["messages"].append(user_msg)
    render_message(user_msg)

    with st.chat_message("assistant"):
        with st.spinner("Searching the policies..."):
            try:
                result = ask(store, llm, question)
            except Exception as e:
                result = {
                    "answer": f"Sorry, something went wrong: {str(e)[:150]}",
                    "sources": [],
                    "refused": False,
                }
        st.markdown(result["answer"])
        if result["sources"]:
            with st.expander("Sources"):
                for s in result["sources"]:
                    st.write(f"• {s}")

    st.session_state["messages"].append(
        {
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
        }
    )
