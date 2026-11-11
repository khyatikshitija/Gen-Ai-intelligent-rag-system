# 💼 Zyro Dynamics HR Help Desk

A Retrieval-Augmented Generation (RAG) chatbot that answers employee questions using only the company's HR policy documents, and politely declines anything outside them.

**Live app:** https://YOUR-APP-NAME.streamlit.app
**Demo video:** https://www.loom.com/share/YOUR-LOOM-LINK

---

## What it does

- Answers questions on leave, benefits, WFH, travel, conduct and other HR policies.
- Shows the **source document and page** for every answer.
- **Refuses** questions that are out of scope (other companies, company finances, products, competitors).
- **Refuses** requests for an individual's personal records (for example, "Tell me my ESOP vesting schedule"), since policy documents cannot contain them.

## Pipeline

| Step | Choice |
|------|--------|
| Document loading | PyPDF (`PyPDFLoader`) on 11 HR policy PDFs |
| Chunking | `RecursiveCharacterTextSplitter`, chunk size 800, overlap 150 |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector store | FAISS |
| Retrieval | MMR + plain similarity search, merged and de-duplicated (up to 10 chunks) |
| LLM | Groq `openai/gpt-oss-120b`, temperature 0 |
| Prompt | Answers only from the retrieved excerpts, otherwise returns a fixed refusal message |
| Guardrail | Regex check that refuses personal-record requests before the LLM is called |

## Results

Scored on a 20-question `submission.csv`: 15 in-scope questions answered from policy and 5 out-of-scope questions that must be refused.

**Kaggle score: 88.55 / 100**

## Project structure

```
.
├── app.py             # Streamlit chat app
├── requirements.txt   # Python dependencies
├── data/              # the 11 HR policy PDFs
└── README.md
```

## Run locally

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPO.git
cd YOUR-REPO
pip install -r requirements.txt
```

Create `.streamlit/secrets.toml`:

```toml
GROQ_API_KEY = "your_groq_api_key"
```

Then start the app:

```bash
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push the repo to GitHub, with the PDFs inside `data/`.
2. On [share.streamlit.io](https://share.streamlit.io), click **Create app** and select the repo, branch `main` and main file `app.py`.
3. Under **Advanced settings → Secrets**, add `GROQ_API_KEY = "your_groq_api_key"`.
4. Deploy.

## Example questions

- How much Earned Leave can I carry forward?
- What medical insurance does the company provide?
- What is the notice period to resign?
- Am I eligible to work from home?

Out-of-scope and personal-record questions, such as "What is Zoho's leave policy like?" or "Tell me my ESOP vesting schedule", are declined.

## Tech stack

Python, Streamlit, LangChain, FAISS, Hugging Face sentence-transformers, Groq API.
