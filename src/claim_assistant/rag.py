from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from uuid import uuid5, NAMESPACE_URL

from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec

from .config import Settings
from .schemas import RetrieveDocument


@dataclass(frozen=True)
class DocumentsChunk:
    id: str
    text: str
    source: str
    title: str


class PineconeKnowledgeBase:
    """ Small pinecone wrapper for retrieval and document ingestion """

    def __init__(self, settings: Settings, embeddings: OpenAIEmbeddings | None = None) -> None:
        settings.validate_run_time_secrets()
        self.settings = settings
        self.embeddings = embeddings or OpenAIEmbeddings(
            model=settings.openai_embedding_model,
            api_key=settings.openai_api_key,
        )
        self.client = Pinecone(api_key=settings.pinecone_api_key)
        self.index = None