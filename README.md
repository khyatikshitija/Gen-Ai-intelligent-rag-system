# 💼 Zyro Dynamics HR Help Desk (RAG Chatbot)

An AI-powered HR Help Desk chatbot for **Zyro Dynamics**, built with Retrieval-Augmented Generation (RAG) over 11 HR policy PDFs. Employees can ask about leave, benefits, work from home, travel, conduct and other policies and get accurate answers with source citations. Questions outside the policy documents are politely declined.

🔗 **Live app:** https://gen-ai-intelligent-rag-system.streamlit.app/
📦 **GitHub repo:** https://github.com/khyatikshitija/Gen-Ai-intelligent-rag-system

---

## Features

- **Answers only from policy documents:** no outside knowledge and no invented details.
- **Source citations:** every answer shows the PDF and page it came from.
- **Out-of-scope handling:** questions about other companies, company finances, competitors and similar topics are refused.
- **Personal-record guard:** requests for an individual's own data (for example "tell me my ESOP vesting schedule") are refused, since policy documents can't contain them.
- **Chat UI:** conversation history, clickable example questions and a clear-chat button.

## How it works

```
HR policy PDFs ──► PyPDF loader ──► Chunking (800 chars, 150 overlap)
                                          │
                                          ▼
                          MiniLM embeddings ──► FAISS vector store
                                                      │
User question ──► Personal-record guard (regex)       │
                         │                            ▼
                         └──► Retrieval: similarity + MMR (merged, de-duplicated, up to 10 chunks)
                                                      │
                                                      ▼
                                   Groq LLM (openai/gpt-oss-120b)
                                   grounded prompt, refuses if not covered
                                                      │
                                                      ▼
                                       Answer + source citations
```

| Component | Choice |
|---|---|
| PDF loading | PyPDF (`PyPDFLoader`) |
| Chunking | `RecursiveCharacterTextSplitter`, size 800, overlap 150 |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector store | FAISS |
| Retrieval | Similarity search + MMR, merged and de-duplicated |
| LLM | Groq `openai/gpt-oss-120b`, temperature 0 |
| UI | Streamlit |

## Project structure

```
.
├── app.py              # Streamlit app (RAG pipeline + chat UI)
├── requirements.txt    # Python dependencies
├── data/               # The 11 HR policy PDFs
└── README.md
```

## Run locally

1. **Clone the repo**
   ```bash
   git clone https://github.com/khyatikshitija/Gen-Ai-intelligent-rag-system.git
   cd Gen-Ai-intelligent-rag-system
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Add your Groq API key** by creating `.streamlit/secrets.toml`:
   ```toml
   GROQ_API_KEY = "your_groq_api_key"
   ```
   You can get a key at [console.groq.com](https://console.groq.com).

4. **Start the app**
   ```bash
   streamlit run app.py
   ```

## Deployment

The app is deployed on [Streamlit Community Cloud](https://share.streamlit.io) straight from this repo.
The `GROQ_API_KEY` is stored in the app's **Secrets** settings and is never committed to the repository.

## Example questions

- How much Earned Leave can I carry forward?
- What medical insurance does the company provide?
- What is the notice period to resign?
- Am I eligible to work from home?
- *(Refused)* What is Zoho's leave policy like?
- *(Refused)* Tell me my ESOP vesting schedule.

## Evaluation

Built for the Intelligent RAG Challenge on Kaggle: 20 test questions, of which 15 are in-scope (answered from policy) and 5 are out-of-scope (must be refused). Kaggle submission score: **88.55 / 100**.

## Tech stack

Python · Streamlit · LangChain · FAISS · Hugging Face Sentence Transformers · Groq
