from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """ Application settings loaded from environment variables. """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: str =  Field(default="local", validation_alias="APP_ENV")
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")

    openai_api_key: str = Field(default="", validation_alias="OPENAI_API_KEY")
    openai_model: str= Field(default="gpt-4.1", validation_alias="OPENAI_MODEL")

    openai_embedding_model: str = Field(
        default = "text-embedding-3-small",
        validation_alias="OPENAI_EMBEDDING_MODEL"
    )

    pinecone_api_key: str = Field(default="", validation_alias="PINECONE_API_KEY")
    pinecone_index_name: str = Field(default="insurance_claim_kb", validation_alias="PINECONE_INDEX_NAME")

    pinecone_namespace: str = Field(default="claim-assistant", validation_alias="PINECONE_NAMESPACE")
    pinecone_cloud: str = Field(default="aws", validation_alias="PINECONE_CLOUD")
    pinecone_region: str = Field(default="us-east-1", validation_alias="PINECONE_REGION")
    pinecone_vector_dimension: str = Field(default=1536, validation_alias="PINECONE_VECTOR_DIMENSION")

    max_retrieval_results: str = Field(default=5, validation_alias="MAX_RETRIEVAL_RESULTS")
    max_reflection_loops: str = Field(default=1, validation_alias="MAX_REFLECTION_LOOPS")


    def validate_run_time_secrets(self) -> None:
        missing = []
        if not self.openai_api_key:
            missing.append("OPENAI_API_KEY")
        if not self.pinecone_api_key:
            missing.append("PINECONE_API_KEY")
        if missing:
            raise ValueError(f"Missing required secrets: {', '.join(missing)}")



@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
