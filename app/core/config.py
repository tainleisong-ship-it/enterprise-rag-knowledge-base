from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    deepseek_model: str = "deepseek-chat"

    redis_url: str = "redis://localhost:6379/0"
    redis_ttl: int = 3600

    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k_retrieval: int = 10
    top_k_rerank: int = 4

    embedding_model: str = "BAAI/bge-small-zh-v1.5"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
