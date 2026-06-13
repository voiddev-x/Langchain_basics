# langchain_basics

A structured learning repo for LangChain — organized by concept as I work through the fundamentals. Actively updated.

---

## structure

```
langchain_basics/
├── models/ChatModels/     # chat models, structured output basics
├── prompts/               # prompt templates, basics of models and prompts
├── structured_output/     # structured output with LangChain
├── chains/                # chains component of LangChain
├── runnables/             # runnables, LCEL, RunnableParallel, RunnablePassthrough
└── RAG_project/           # full RAG pipeline (see below)
    ├── doc_loaders/       # document loaders
    ├── text_splitter/     # text splitting strategies
    ├── vec_store/         # vector store setup (Chroma)
    ├── retrievers/        # retriever types
    ├── docs/              # source documents
    ├── create_database.py # builds and persists the vector store
    ├── main.py            # RAG chain using LCEL
    └── app.py             # Streamlit frontend to interact with the RAG pipeline
```

---

## what's covered

- chat models and structured output with LangChain
- prompt templates and prompt engineering basics
- chains and the LangChain expression language (LCEL)
- runnables — `RunnableParallel`, `RunnablePassthrough`
- full RAG pipeline: document loading → chunking → embedding → vector store → retrieval → generation
- refactored RAG pipeline into a clean LCEL chain

---

## tech stack

| component | tool |
|---|---|
| framework | LangChain |
| llm | Mistral Gemini (via API) |
| embeddings | `mistral-embed`|
| vector store | Chroma |
| frontend | Streamlit |
| package manager | uv |

---

## notes

this is a learning repo — each folder maps to a concept, not a finished product. the RAG project is the most complete piece: handles both in-context answers and out-of-context queries correctly.

standalone projects (multi-agent systems, production RAG apps) will live in separate repos.
