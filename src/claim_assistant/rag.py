from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Any
from uuid import uuid5, NAMESPACE_URL

from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec

from .config import Settings
from .schemas import RetrieveDocument

# ============================================================
# Data model
# ============================================================

@dataclass(frozen=True)
class DocumentsChunk:
    """
    Represents a single chunk of a source document.

    The object is immutable because a chunk should not change
    after it has been created and assigned an ID.
    """

    id: str
    text: str
    source: str
    title: str


# ============================================================
# Pinecone knowledge base
# ============================================================

class PineconeKnowledgeBase:
    """
    Handles document ingestion and semantic retrieval using
    OpenAI embeddings and Pinecone.

    Responsibilities:
    - Create the Pinecone index when necessary.
    - Convert document chunks into embeddings.
    - Store document vectors in Pinecone.
    - Convert user queries into embeddings.
    - Retrieve semantically similar documents.
    """

    def __init__(self, settings: Settings, embeddings: OpenAIEmbeddings | None = None) -> None:
        """
        Initialize the knowledge base.

        Args:
            settings:
                Application settings containing API keys,
                embedding model configuration, and Pinecone settings.

            embeddings:
                Optional embedding client. If not supplied,
                an OpenAIEmbeddings instance is created automatically.
         """
        # Ensure required API keys and runtime settings are available.
        settings.validate_runtime_secrets()

        self.settings = settings

        # Allow dependency injection for easier testing.
        # If no embeddings object is supplied, create the default one.
        self.embeddings = embeddings or OpenAIEmbeddings(
            model=settings.openai_embedding_model,
            api_key=settings.openai_api_key,
        )

        # Pinecone control-plane client.
        self.client = Pinecone(
            api_key=settings.pinecone_api_key
        )

        # The index connection is created lazily only when required.
        self.index = None


    # --------------------------------------------------------
    # Index management
    # --------------------------------------------------------

    def ensure_index(self) -> None:
        """
        Ensure that the configured Pinecone index exists.

        If the index already exists, nothing is done.
        Otherwise, a new serverless index is created.
        """
        #  Gets all the pinecone index name
        existing_indexes = self._get_indexes_names()

        # if configured already exists, do nothing
        if self.settings.pinecone_index_name in existing_indexes:
            return

        # Otherwise create a new index
        serverless_spec = ServerlessSpec(
            cloud = self.settings.pinecone_cloud,
            region = self.settings.pinecone_region,
        )

        self._create_index(serverless_spec)

        # Reset the cached index client because a new index has just been created.
        self.index = None


    def _create_index(self, spec: ServerlessSpec) -> None:
        """
        Create the Pinecone index.

        The compatibility check allows this code to work with
        different Pinecone SDK versions.
        """

        index_arguments = {
            "name": self.settings.pinecone_index_name,
            "dimension": self.settings.pinecone_vector_dimension,
            "metric": "cosine",
            "spec": spec,
        }

        if hasattr(self.client, "create_index"):
            self.client.create_index(**index_arguments)
        else:
            self.client.indexes.create(**index_arguments)

    def _get_indexes_names(self) -> set[str]:
        """
        Return all available Pinecone index names.
        """

        if hasattr(self.client, "list_indexes"):
            indexes = self.client.list_indexes()
        else:
            indexes = self.client.indexes.list()

        # Some SDK version expose a convenient names() method.
        if hasattr(indexes, "names"):
            return set(indexes.names())

        # Fallback for SDK responses represented as dictionaries or index-description objects.
        return {
            str(_get_value(value=index, key="name", default=index)
                for index in indexes)
        }

    def _get_index_client(self):
        """
        Return a connection to the configured Pinecone index.

        The index is cached after the first lookup.
        """

        if self.index is None:
            self.index = self._create_index_client()

        return self.index

    def _create_index_client(self):
        """
        Create the Pinecone data-plane index client.

        Different Pinecone SDK versions expose this method
        using slightly different names.
        """

        index_name = self.settings.pinecone_index_name

        if hasattr(self.client, "index"):
            return self.client.index(index_name)

        return self.client.Index(index_name)

    # --------------------------------------------------------
    # Document ingestion
    # --------------------------------------------------------

    def upsert_chunk(self, chunks:Iterable[DocumentsChunk], batch_size: int = 100) -> int:
        """
        Convert document chunks into embeddings and store them in Pinecone.

        Args:
            chunks:
                Document chunks to embed and store.

            batch_size:
                Maximum number of vectors sent to Pinecone
                in each batch.

        Returns:
            Number of vectors successfully upserted.
        """

        chunk_list = list(chunks)

        if not chunk_list:
            return 0

        # Extract only the textual content because this is what
        # the embedding model converts into numerical vectors.
        texts = [
            chunk.text for chunk in chunk_list
        ]

        # Generate one embedding vector for every document chunk.
        vectors = self.embeddings.embed_documents(texts)

        # Combine:
        #   chunk ID + embedding vector + searchable metadata
        #
        # Pinecone stores the vector for similarity search while
        # metadata allows us to recover the original document text.
        pinecone_vectors = [
            (
                chunk.id,
                vector,
                {
                    "text": chunk.text,
                    "source": chunk.source,
                    "title": chunk.title
                },
            )
            for chunk, vector in zip(chunk_list, vectors, strict=True)
        ]

        response = self._get_index_client().upsert(
            vectors=pinecone_vectors,
            namespace=self.settings.pinecone_namespace,
            batch_size=batch_size,
        )

        # --------------------------------------------------------
        # Retrieval
        # --------------------------------------------------------



# ============================================================
# SDK compatibility helpers
# ============================================================

def _get_value(value: Any, key: str, default: Any = None) -> Any:
    """
    Retrieve a value from either a dictionary or an object.

    Pinecone SDK responses may differ between versions:
    some responses behave like dictionaries while others expose
    values as object attributes.
    """

    if isinstance(value, dict):
        return value.get(key, default,)

    return getattr(value,key,default,)