"""
RAG System for Personal Profile Document
=========================================
Architecture:
  1. Document Loading
  2. Section-Aware Chunking
  3. Embedding Generation
  4. ChromaDB Persistent Vector Store
  5. Similarity-Based Retriever
  6. Grounded QA Chain (LCEL)
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Tuple
from dotenv import load_dotenv

load_dotenv()

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_huggingface import (
    HuggingFaceEmbeddings,
    HuggingFaceEndpoint,
    ChatHuggingFace,
)

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
FILE_PATH = "./Wahb_Mohamed_Profile.txt"
CHROMA_DIR = "./chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
CHUNK_SIZE = 600
CHUNK_OVERLAP = 80
TOP_K = 4
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")


# ──────────────────────────────────────────────
# 1. DOCUMENT LOADING
# ──────────────────────────────────────────────
def load_document(file_path: str) -> str:
    loader = TextLoader(file_path, encoding="utf-8")
    docs = loader.load()
    return docs[0].page_content


# ──────────────────────────────────────────────
# 2. SECTION-AWARE CHUNKING
# ──────────────────────────────────────────────
SECTION_PATTERN = re.compile(r"^={10,}\s+([A-Z][A-Z\s&]+?)\s*={0,30}\s*$", re.MULTILINE)


def parse_sections(raw_text: str) -> List[Tuple[str, str]]:
    matches = list(SECTION_PATTERN.finditer(raw_text))
    sections = []
    for i, match in enumerate(matches):
        section_name = match.group(1).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw_text)
        body = raw_text[start:end].strip()
        sections.append((section_name, body))
    return sections


def chunk_section(section_name: str, body: str) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "],
        length_function=len,
    )
    raw_chunks = splitter.split_text(body)
    documents = []
    for idx, chunk in enumerate(raw_chunks):
        sub_match = re.match(r"^([A-Za-z &:]+):\s*\n", chunk)
        subsection = sub_match.group(1).strip() if sub_match else ""
        documents.append(
            Document(
                page_content=chunk,
                metadata={
                    "section": section_name,
                    "subsection": subsection,
                    "chunk_idx": idx,
                    "source": FILE_PATH,
                },
            )
        )
    return documents


def build_chunks(raw_text: str) -> List[Document]:
    sections = parse_sections(raw_text)
    all_chunks = []
    for section_name, body in sections:
        chunks = chunk_section(section_name, body)
        all_chunks.extend(chunks)
        print(f"  [Section] '{section_name}' → {len(chunks)} chunk(s)")
    return all_chunks


# ──────────────────────────────────────────────
# 3. EMBEDDING MODEL
# ──────────────────────────────────────────────
def build_embeddings() -> HuggingFaceEmbeddings:
    print(f"\n[Embeddings] Loading model: {EMBEDDING_MODEL}")
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


# ──────────────────────────────────────────────
# 4. VECTOR STORE
# ──────────────────────────────────────────────
def build_vector_store(
    chunks: List[Document], embeddings: HuggingFaceEmbeddings
) -> Chroma:
    if Path(CHROMA_DIR).exists() and any(Path(CHROMA_DIR).iterdir()):
        print(f"\n[VectorStore] Loading existing ChromaDB from '{CHROMA_DIR}'")
        return Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    print(f"\n[VectorStore] Creating new ChromaDB at '{CHROMA_DIR}'")
    vs = Chroma.from_documents(
        documents=chunks, embedding=embeddings, persist_directory=CHROMA_DIR
    )
    print(f"[VectorStore] Stored {len(chunks)} chunks.")
    return vs


# ──────────────────────────────────────────────
# 5. RETRIEVER
# ──────────────────────────────────────────────
def build_retriever(vector_store: Chroma):
    return vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": TOP_K}
    )


# ──────────────────────────────────────────────
# 6. QA CHAIN
# ──────────────────────────────────────────────
STRICT_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a factual question-answering assistant.
Your ONLY source of information is the context provided below.

Rules:
- Answer ONLY using the information present in the context.
- Do NOT add, infer, or hallucinate any information not explicitly stated.
- If the answer cannot be found in the context, respond EXACTLY with:
  "The information is not available in the provided document."
- Keep your answer concise.

Context:
{context}

Question: {question}

Answer:""",
)


def format_docs(docs: List[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def build_llm():
    endpoint = HuggingFaceEndpoint(
        repo_id="HuggingFaceH4/zephyr-7b-beta",
        huggingfacehub_api_token=HF_TOKEN,
        task="conversational",
        max_new_tokens=512,
        temperature=0.1,
    )
    return ChatHuggingFace(llm=endpoint)


def build_chain(retriever, llm):
    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | STRICT_PROMPT
        | llm
        | StrOutputParser()
    )


# ──────────────────────────────────────────────
# QUERY INTERFACE
# ──────────────────────────────────────────────
def query(chain, retriever, question: str) -> Dict:
    answer = chain.invoke(question).strip()
    source_docs = retriever.invoke(question)
    sources = [
        {
            "section": d.metadata.get("section", ""),
            "subsection": d.metadata.get("subsection", ""),
            "snippet": d.page_content[:120] + "...",
        }
        for d in source_docs
    ]
    return {"answer": answer, "sources": sources}


def pretty_print(question: str, result: Dict) -> None:
    print("\n" + "=" * 60)
    print(f"Q: {question}")
    print("-" * 60)
    print(f"A: {result['answer']}")
    print("-" * 60)
    print("Sources used:")
    for i, src in enumerate(result["sources"], 1):
        label = src["section"]
        if src["subsection"]:
            label += f" > {src['subsection']}"
        print(f"  [{i}] [{label}] {src['snippet']}")
    print("=" * 60)


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main():
    print("[1/6] Loading document...")
    raw_text = load_document(FILE_PATH)
    print(f"      Loaded {len(raw_text)} characters.")

    print("\n[2/6] Parsing sections and chunking...")
    chunks = build_chunks(raw_text)
    print(f"      Total chunks: {len(chunks)}")

    print("\n[3/6] Building embedding model...")
    embeddings = build_embeddings()

    print("\n[4/6] Building / loading ChromaDB vector store...")
    vector_store = build_vector_store(chunks, embeddings)

    print(f"\n[5/6] Building retriever (top-k={TOP_K})...")
    retriever = build_retriever(vector_store)

    print("\n[6/6] Building QA chain (Zephyr-7B via HuggingFace)...")
    llm = build_llm()
    chain = build_chain(retriever, llm)
    print("      RAG system ready.\n")

    example_questions = [
        "What is Wahb Mohamed's graduation project about?",
        "Which machine learning courses has Wahb completed?",
        "What are Wahb's long-term career goals?",
        "What is Wahb's favorite food?",
        "What programming languages does Wahb know?",
        "What leadership roles has Wahb held?",
    ]

    for question in example_questions:
        result = query(chain, retriever, question)
        pretty_print(question, result)


if __name__ == "__main__":
    main()
