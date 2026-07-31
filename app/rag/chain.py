import logging
from typing import List

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

from app.core.config import settings
from app.rag.reranker import RuleBasedReranker
from app.rag.retriever import HybridRetriever

logger = logging.getLogger(__name__)

PROMPT = ChatPromptTemplate.from_template(
    """你是一个企业知识库助手。请严格根据以下检索到的上下文回答用户问题。
如果上下文中没有相关信息，请明确说明知识库中未找到相关内容，禁止编造。

上下文：
{context}

问题：{question}

回答："""
)


def _format_docs(docs: List[Document]) -> str:
    return "\n\n---\n\n".join(d.page_content for d in docs)


def get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.deepseek_model,
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        temperature=0.3,
    )


class RAGChain:
    def __init__(self) -> None:
        self.retriever = HybridRetriever()
        self.reranker = RuleBasedReranker()

    def run(self, question: str, top_k: int | None = None) -> dict:
        k = top_k or settings.top_k_rerank

        candidates = self.retriever.retrieve(question)
        reranked = self.reranker.rerank(question, candidates, top_k=k)

        docs = [d for d, _ in reranked]
        sources = [
            {"content": d.page_content, "score": round(float(s), 4), "source": "knowledge_base"}
            for d, s in reranked
        ]

        llm = get_llm()
        chain = (
            {"context": lambda _: _format_docs(docs), "question": RunnablePassthrough()}
            | PROMPT
            | llm
            | StrOutputParser()
        )
        answer = chain.invoke(question)

        return {"answer": answer, "sources": sources}


rag_chain = RAGChain()
