"""
vector_store/indexer.py
────────────────────────
Builds and manages the ChromaDB vector store.

Features
────────
• Multilingual sentence-transformers (Thai + English).
• Persistent ChromaDB with rich metadata filtering.
• Incremental upsert: skips already-indexed chunks.
• Returns LangChain-compatible retriever for chain integration.
"""

import hashlib
from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from loguru import logger

from config.settings import (
    CHROMA_COLLECTION,
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL,
    SIMILARITY_THRESHOLD,
    TOP_K_RETRIEVAL,
    VECTOR_DB_DIR,
)


# ── Embedding model ───────────────────────────────────────────────────────────

def get_embedding_model() -> HuggingFaceEmbeddings:
    logger.info(f"Loading embeddings: {EMBEDDING_MODEL}")
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},        # swap to "cuda" if available
        encode_kwargs={"normalize_embeddings": True, "batch_size": 64},
    )


# ── Stable document ID ────────────────────────────────────────────────────────

def _doc_id(doc: Document) -> str:
    if doc.metadata.get("chunk_id"):
        return str(doc.metadata["chunk_id"])
    key = (
        doc.metadata.get("source_url", "")
        + doc.metadata.get("section", "")
        + str(doc.metadata.get("chunk_index", 0))
        + doc.page_content[:80]
    )
    return hashlib.sha256(key.encode()).hexdigest()[:32]


# ── Manager ───────────────────────────────────────────────────────────────────

class VectorStoreManager:
    """
    Manages the ChromaDB vector store lifecycle.

    Usage
    ─────
        vsm = VectorStoreManager()
        vsm.build(docs)                                    # first-time build
        vsm.upsert(more_docs)                              # incremental add
        hits = vsm.search("Newton's second law",
                           subject="physics", grade="M4")
        retriever = vsm.as_retriever(subject="math")       # for LangChain chains
    """

    def __init__(self):
        self.embedder: Optional[HuggingFaceEmbeddings] = None
        self._client = chromadb.PersistentClient(
            path=str(VECTOR_DB_DIR),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._store: Optional[Chroma] = None

    # ── internal ──────────────────────────────────────────────────────────────

    def _get_embedder(self) -> HuggingFaceEmbeddings:
        if self.embedder is None:
            self.embedder = get_embedding_model()
        return self.embedder

    def _get_collection(self):
        return self._client.get_or_create_collection(
            name=CHROMA_COLLECTION,
            metadata={
                "hnsw:space": "cosine",
                "embedding_model": EMBEDDING_MODEL,
                "embedding_dim": EMBEDDING_DIMENSION,
            },
        )

    def _get_store(self) -> Chroma:
        if self._store is None:
            self._store = Chroma(
                client=self._client,
                collection_name=CHROMA_COLLECTION,
                embedding_function=self._get_embedder(),
                collection_metadata={
                    "hnsw:space": "cosine",
                    "embedding_model": EMBEDDING_MODEL,
                    "embedding_dim": EMBEDDING_DIMENSION,
                },
            )
        return self._store

    def _existing_ids(self) -> set:
        try:
            return set(self._get_store()._collection.get(include=[])["ids"])
        except Exception:
            return set()

    # ── public ────────────────────────────────────────────────────────────────

    def build(self, docs: list, batch_size: int = 500) -> None:
        """Embed and index documents; skip already-present chunks."""
        if not docs:
            logger.warning("No documents to index.")
            return

        existing  = self._existing_ids()
        new_docs  = [d for d in docs if _doc_id(d) not in existing]

        if not new_docs:
            logger.info("All chunks already indexed. Nothing to do.")
            return

        logger.info(
            f"Indexing {len(new_docs)} new chunks "
            f"({len(docs) - len(new_docs)} already present)…"
        )

        store = self._get_store()
        for i in range(0, len(new_docs), batch_size):
            batch = new_docs[i : i + batch_size]
            store.add_documents(documents=batch, ids=[_doc_id(d) for d in batch])
            logger.debug(f"  batch {i // batch_size + 1}: {len(batch)} docs")

        logger.success(f"Indexing complete. {len(new_docs)} chunks added.")

    def upsert(self, docs: list) -> None:
        """Alias for build (incremental)."""
        self.build(docs)

    def stats(self) -> dict:
        return {
            "collection":   CHROMA_COLLECTION,
            "embedding_model": EMBEDDING_MODEL,
            "embedding_dim": EMBEDDING_DIMENSION,
            "total_chunks": self._get_collection().count(),
            "embedding_loaded": self.embedder is not None,
        }

    # ── retrieval ─────────────────────────────────────────────────────────────

    def _build_where(
        self,
        subject: Optional[str] = None,
        grade:   Optional[str] = None,
        source:  Optional[str] = None,
        filter_examples:    Optional[bool] = None,
        filter_definitions: Optional[bool] = None,
    ) -> Optional[dict]:
        clauses = []
        if subject:
            clauses.append({"subject": {"$eq": subject}})
        if grade:
            clauses.append({"grade": {"$eq": grade}})
        if source:
            clauses.append({"source": {"$eq": source}})
        if filter_examples:
            clauses.append({"is_example": {"$eq": True}})
        if filter_definitions:
            clauses.append({"is_definition": {"$eq": True}})

        if not clauses:
            return None
        if len(clauses) == 1:
            return clauses[0]
        return {"$and": clauses}

    def search(
        self,
        query: str,
        subject: Optional[str] = None,
        grade:   Optional[str] = None,
        source:  Optional[str] = None,
        top_k:   int = TOP_K_RETRIEVAL,
        filter_examples:    Optional[bool] = None,
        filter_definitions: Optional[bool] = None,
    ) -> list:
        """
        Semantic search with optional metadata filters.

        Supports Thai and English queries.
        """
        store = self._get_store()
        where = self._build_where(subject, grade, source, filter_examples, filter_definitions)

        try:
            results = store.similarity_search_with_relevance_scores(
                query, k=top_k, filter=where
            )
            filtered = [doc for doc, score in results if score >= SIMILARITY_THRESHOLD]
            logger.debug(
                f"Search '{query[:50]}' → "
                f"{len(filtered)}/{len(results)} above threshold"
            )
            return filtered
        except Exception as exc:
            logger.error(f"Search error: {exc}")
            return []

    def as_retriever(
        self,
        subject: Optional[str] = None,
        grade:   Optional[str] = None,
        source:  Optional[str] = None,
        top_k:   int = TOP_K_RETRIEVAL,
    ):
        """LangChain-compatible retriever for use in chains."""
        where = self._build_where(subject, grade, source)
        kwargs: dict = {"k": top_k}
        if where:
            kwargs["filter"] = where
        return self._get_store().as_retriever(
            search_type="similarity",
            search_kwargs=kwargs,
        )


# ── Singleton ─────────────────────────────────────────────────────────────────

_instance: Optional[VectorStoreManager] = None


def get_vector_store() -> VectorStoreManager:
    global _instance
    if _instance is None:
        _instance = VectorStoreManager()
    return _instance


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    from embeddings.document_processor import DocumentProcessor

    p = argparse.ArgumentParser(description="Build ChromaDB index")
    p.add_argument("--subjects", nargs="*")
    p.add_argument("--grades",   nargs="*")
    args = p.parse_args()

    proc = DocumentProcessor()
    docs = proc.process_all(subjects=args.subjects, grades=args.grades)

    vsm = VectorStoreManager()
    vsm.build(docs)
    print("\nVector store ready:", vsm.stats())
