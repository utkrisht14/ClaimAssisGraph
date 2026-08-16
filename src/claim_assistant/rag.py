from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Any
from uuid import uuid5, NAMESPACE_URL

from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec

from .config import Settings
from .schemas import RetrievedDocument

# ============================================================
# Data model
# ============================================================

@dataclass(frozen=True)
class DocumentChunk:
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

    def upsert_chunk(self, chunks:Iterable[DocumentChunk], batch_size: int = 100) -> int:
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

        def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedDocument]:
            """
            Retrieve document chunks that are semantically similar to the user's query.

            Args:
                query:
                    User question or search text.

                top_k:
                    Number of results to retrieve. If omitted,
                    the configured default is used.

            Returns:
                Retrieved documents ordered by Pinecone similarity.
            """

            # Convert the user's question into the same vector space used for the stored document embeddings.
            query_vector = self.embeddings.embed_query(query)

            retrieval_limit = (
                top_k
                if top_k is not None
                else self.settings.pinecone_top_k
            )

            result = self._get_index_client().query(
                vector=query_vector,
                top_k=retrieval_limit,
                namspace=self.settings.pinecone_namespace,
                include_metadata=True,
            )

            matches = _get_value(result, "matches", [])

            return [
                self._match_to_document(match)
                for match in matches
            ]

        def _match_to_document(self, match: Any,) -> RetrievedDocument:
            """
            Convert a raw Pinecone search result into the application's
            RetrievedDocument schema.
            """

            metadata = _get_value(
                match,
                "metadata",
                {},
            ) or {}

            return RetrievedDocument(
                id=str(
                    _get_value(
                        match,
                        "id",
                        metadata.get("id", ""),
                    )
                ),
                text=str(
                    metadata.get("text", "")
                ),
                source=str(
                    metadata.get("source", "unknown")
                ),
                title=str(
                    metadata.get("title", "Untitled")
                ),
                score=_get_value(
                    match,
                    "score",
                    None,
                ),
                metadata=dict(metadata),
            )


# ============================================================
# Document loading and chunk creation
# ============================================================

def load_markdown_chunks(source_dir: Path, max_chars: int = 1800) -> list[DocumentChunk]:
    """
    Load Markdown files recursively and split them into chunks.

    Each chunk receives a deterministic UUID based on:
    - file path
    - chunk position
    - beginning of the chunk text

    Using UUID5 makes re-ingestion stable: the same chunk will
    normally receive the same ID.
    """

    chunks: list[DocumentChunk] = []

    markdown_files = sorted(
        source_dir.rglob("*.md")
    )

    for path in markdown_files:
        # Example:
        #
        # insurance-claims.md
        #        ↓
        # Insurance Claims
        title = (
            path.stem
            .replace("-", " ")
            .title()
        )

        document_text = path.read_text(encoding="utf-8")

        text_chunks = _chunk_text(document_text, max_chars=max_chars)

        for position, chunk_text in enumerate(text_chunks, start=1):
            chunk_id = _create_chunk_id(path =path, position=position, text=chunk_text)

            chunks.append( DocumentChunk(
                    id=chunk_id,
                    text=chunk_text,
                    source=str(path),
                    title=title,
                    )
            )

        return chunks


def _create_chunk_id(path: Path, position:int, text:str) -> str:
    """
    Generate a deterministic ID for a document chunk.

    UUID5 always produces the same UUID when given the same input.
    """
    unique_value = (
        f"{path.as_posix()}:"
        f"{position}:"
        f"{text[:64]}"
    )

    return str(uuid5(NAMESPACE_URL, unique_value))


# ============================================================
# Text chunking
# ============================================================

def _chunk_text(text: str, max_chars: int) -> list[str]:
    """
    Split text into paragraph-based chunks.

    Paragraphs are kept together whenever possible until adding
    another paragraph would exceed max_chars.
    """

    paragraphs = [
        paragraph.strip()
        for paragraph in text.split("\n\n")
        if paragraph.strip()
    ]

    chunks: list[str] = []

    current_chunk = ""

    for paragraph in paragraphs:
        separator_size = 2 if current_chunk else 0

        combined_length = (len(current_chunk) + separator_size + len(paragraph))

        # Keep adding paragraphs while the chunk remains within the configured size
        if combined_length <= max_chars:
            current_chunk = (
                f"{current_chunk}\n\n{paragraph}".strip()
            )
            continue

        # The current chunk is full. So store it.
        if current_chunk:
            chunks.append(current_chunk)

        # Start building the next chunk
        current_chunk = paragraph

    # Add the final partially built chunk
    if current_chunk:
        chunks.append(current_chunk)

    return chunks


# ============================================================
# LLM context formatting
# ============================================================

def format_context(documents: list[RetrievedDocument]) -> str:
    """
    Convert retrieved documents into a readable context string
    that can be inserted into an LLM prompt.
    """
    if not documents:
        return "No relevant documents found."

    formatted_documents = [
        (
            f"[{doc.id}] {doc.title} ({doc.source})\n"
            f"{doc.text}"
        )
        for doc in documents
    ]

    return "\n\n".join(
        formatted_documents
    )







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


def _get_upserted_count(response: Any) -> int:
    """
    Extract the number of successfully upserted vectors from a
    Pinecone response.
    """

    return int(_get_value(response, "upserted_count", 0))