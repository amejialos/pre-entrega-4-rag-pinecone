"""Pipeline de ingesta: documentos locales -> chunks -> embeddings -> Pinecone.

Uso: uv run python ingest.py [--reset]

- Usa el SDK nativo (`index.upsert`) en lugar de `PineconeVectorStore.add_documents`
  para controlar exactamente qué metadata se guarda y con qué id.
- Los ids son los `chunk_id` deterministas de corpus.py: volver a correr la ingesta
  sobreescribe los mismos vectores en vez de duplicarlos (upsert idempotente).
- Todo va a un namespace (PINECONE_NAMESPACE), nunca al namespace por defecto.
- `--reset` borra el namespace antes de subir, por si cambió el chunking y quedaron
  chunks viejos con ids que ya no existen.
"""

import argparse
import logging
import time
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from config import DIMENSION, build_embeddings, build_pinecone_client, load_settings
from corpus import load_chunks
from init_index import ensure_index

logger = logging.getLogger(__name__)

UPSERT_BATCH_SIZE = 100
METADATA_FIELDS = ("doc_id", "chunk_id", "source", "page", "section", "category", "title", "chunk_index")


def build_records(chunks: list[Document], vectors: list[list[float]]) -> list[dict[str, Any]]:
    """Arma los registros de upsert: id, vector y metadata (incluido el texto original)."""
    if len(chunks) != len(vectors):
        raise ValueError(f"{len(chunks)} chunks pero {len(vectors)} vectores.")
    records = []
    for chunk, vector in zip(chunks, vectors, strict=True):
        if len(vector) != DIMENSION:
            raise ValueError(
                f"El embedding de {chunk.metadata['chunk_id']} tiene {len(vector)} dimensiones; "
                f"el índice espera {DIMENSION}. Revisá EMBEDDING_MODEL."
            )
        metadata = {field: chunk.metadata[field] for field in METADATA_FIELDS}
        # Pinecone no guarda el documento: si el texto no va en la metadata, la
        # búsqueda devuelve ids sin contenido. PineconeVectorStore lo lee de "text".
        metadata["text"] = chunk.page_content
        records.append({"id": chunk.metadata["chunk_id"], "values": vector, "metadata": metadata})
    return records


def upsert_chunks(
    index: Any,
    embeddings: Embeddings,
    chunks: list[Document],
    namespace: str,
    batch_size: int = UPSERT_BATCH_SIZE,
) -> int:
    """Calcula los embeddings y sube los chunks en lotes. Devuelve cuántos subió."""
    vectors = embeddings.embed_documents([chunk.page_content for chunk in chunks])
    records = build_records(chunks, vectors)
    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        index.upsert(vectors=batch, namespace=namespace)
        logger.info("Subidos %d/%d registros al namespace %r.", start + len(batch), len(records), namespace)
    return len(records)


def reset_namespace(index: Any, namespace: str) -> None:
    try:
        index.delete(delete_all=True, namespace=namespace)
        logger.info("Namespace %r vaciado.", namespace)
    except Exception as exc:  # el SDK responde 404 si el namespace todavía no existe
        if "not found" not in str(exc).lower() and "404" not in str(exc):
            raise
        logger.info("El namespace %r no existía: nada que borrar.", namespace)


def namespace_count(index: Any, namespace: str) -> int:
    stats = index.describe_index_stats()
    namespaces = stats["namespaces"] if isinstance(stats, dict) else stats.namespaces
    info = namespaces.get(namespace)
    if info is None:
        return 0
    return info["vector_count"] if isinstance(info, dict) else info.vector_count


def wait_for_count(index: Any, namespace: str, expected: int, timeout: float = 60.0, poll: float = 2.0) -> int:
    """Pinecone es eventualmente consistente: espera a que los vectores sean visibles."""
    deadline = time.monotonic() + timeout
    while True:
        count = namespace_count(index, namespace)
        if count >= expected or time.monotonic() > deadline:
            return count
        time.sleep(poll)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingesta de data/ en Pinecone")
    parser.add_argument("--reset", action="store_true", help="vaciar el namespace antes de subir")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    settings = load_settings()
    pc = build_pinecone_client(settings)
    ensure_index(pc, settings.index_name)
    index = pc.Index(settings.index_name)

    chunks = load_chunks()
    docs = sorted({c.metadata["doc_id"] for c in chunks})
    print(f"{len(chunks)} chunks de {len(docs)} documentos: {', '.join(docs)}")

    if args.reset:
        reset_namespace(index, settings.namespace)
    total = upsert_chunks(index, build_embeddings(settings), chunks, settings.namespace)
    visible = wait_for_count(index, settings.namespace, total)
    print(f"Upsert: {total} vectores | visibles en {settings.index_name}/{settings.namespace}: {visible}")


if __name__ == "__main__":
    main()
