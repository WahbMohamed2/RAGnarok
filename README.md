# ProfilePulse

A **Retrieval-Augmented Generation (RAG)** system that lets you query a personal profile document using natural language — powered by HuggingFace embeddings, ChromaDB, and Zephyr-7B.

---

## What This Project Does

ProfilePulse ingests a structured `.txt` profile document, breaks it into semantically meaningful chunks, embeds them into a persistent vector store, and answers questions about the person using only the document as the source of truth — no hallucinations, no assumptions.

---

## Architecture

```
Profile Document (.txt)
        │
        ▼
 ┌─────────────────┐
 │  Section-Aware  │   Parses sections by === headers, then
 │    Chunking     │   splits each section with RecursiveCharacterTextSplitter
 └────────┬────────┘
          │
          ▼
 ┌─────────────────┐
 │   HuggingFace   │   sentence-transformers/all-mpnet-base-v2
 │   Embeddings    │   (normalized, CPU inference)
 └────────┬────────┘
          │
          ▼
 ┌─────────────────┐
 │    ChromaDB     │   Persistent vector store — only built once,
 │  Vector Store   │   reloaded on subsequent runs
 └────────┬────────┘
          │
          ▼
 ┌─────────────────┐
 │  Similarity     │   Top-K=4 chunks retrieved per query
 │  Retriever      │
 └────────┬────────┘
          │
          ▼
 ┌─────────────────┐
 │  Zephyr-7B LLM  │   HuggingFaceEndpoint — grounded, strict prompt
 │  (QA Chain)     │   answers only from retrieved context
 └─────────────────┘
```

**Stack:** LangChain · ChromaDB · HuggingFace Inference API · sentence-transformers · Zephyr-7B-beta

---

## Project Structure

```
ProfilePulse/
├── rag_system.py            # Main RAG pipeline
├── test.py                  # Quick connectivity test for Zephyr-7B
├── Wahb_Mohamed_Profile.txt # Source profile document
├── requirements.txt         # Dependencies
├── .gitignore
└── chroma_db/               # Auto-generated — persistent vector store (gitignored)
```

---

## Setup

**1. Clone & create a virtual environment**
```bash
git clone https://github.com/your-username/ProfilePulse.git
cd ProfilePulse
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Set your HuggingFace API token**

Create a `.env` file in the project root:
```
HUGGINGFACEHUB_API_TOKEN=your_token_here
```
Get your token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

**4. Run**
```bash
python rag_system.py
```

The first run builds and persists the ChromaDB vector store. Subsequent runs load it instantly.

---

## Sample Output

```
Q: Which machine learning courses has Wahb completed?
────────────────────────────────────────────────────
A: Wahb has completed Supervised Machine Learning: Regression and Classification
(Stanford/Coursera), the first two courses of the Machine Learning Specialization
by Andrew Ng, the Deep Learning Specialization (first five courses), and studied
the first 8 chapters of "Hands-On Machine Learning". He is currently studying
GANs and Transformers.

Sources used:
  [1] [EDUCATION & COURSES] Completed: Supervised Machine Learning...
  [2] [EDUCATION & COURSES] Action" by Jakob Langr & Vladimir Bok...
```

```
Q: What is Wahb's favorite food?
────────────────────────────────
A: The information is not available in the provided document.
```

The system correctly refuses to hallucinate when the answer isn't in the document.

---

## Configuration

All key parameters live at the top of `rag_system.py`:

| Variable | Default | Description |
|---|---|---|
| `EMBEDDING_MODEL` | `all-mpnet-base-v2` | HuggingFace embedding model |
| `CHUNK_SIZE` | `600` | Characters per chunk |
| `CHUNK_OVERLAP` | `80` | Overlap between chunks |
| `TOP_K` | `4` | Retrieved chunks per query |
| `CHROMA_DIR` | `./chroma_db` | Vector store persistence path |

---

## Notes

- The LLM is instructed via a strict prompt to answer **only from retrieved context** — it will say "The information is not available in the provided document." rather than guess.
- ChromaDB persists to disk so embeddings are only computed once.
- To rebuild the vector store from scratch, delete the `chroma_db/` folder and rerun.

---

## Author

**Wahb Mohamed** — Computer Science Student | AI & Data Science  