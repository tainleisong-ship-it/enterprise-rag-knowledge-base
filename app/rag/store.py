import logging
from pathlib import Path
from typing import List, Tuple

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import settings

logger = logging.getLogger(__name__)

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "knowledge_base.txt"


def load_and_split() -> List[Document]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Knowledge base not found: {DATA_PATH}")

    raw_text = DATA_PATH.read_text(encoding="utf-8")
    paragraphs = [p.strip() for p in raw_text.split("\n\n") if p.strip()]
    base_docs = [Document(page_content=p, metadata={"source": "knowledge_base"}) for p in paragraphs]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", "。", "；", "，", " ", ""],
    )
    chunks = splitter.split_documents(base_docs)
    logger.info("Loaded %d chunks from %s", len(chunks), DATA_PATH.name)
    return chunks


def _build_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vectorstore() -> Tuple[FAISS, List[Document]]:
    chunks = load_and_split()
    embeddings = _build_embeddings()
    store = FAISS.from_documents(chunks, embeddings)
    logger.info("FAISS index built with %d vectors", len(chunks))
    return store, chunks


vectorstore, documents = build_vectorstore()
