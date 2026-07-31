from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="用户问题")
    top_k: int = Field(default=4, ge=1, le=20, description="返回的上下文数量")


class SourceDocument(BaseModel):
    content: str
    score: float
    source: str = "knowledge_base"


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceDocument]
    cached: bool = False


class HealthResponse(BaseModel):
    status: str
    docs: str
    cache_available: bool
