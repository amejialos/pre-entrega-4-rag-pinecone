import pytest

from ingest import upsert_chunks
from rag_system import RAGSystem, tokenize_es

NAMESPACE = "caudal-docs"


@pytest.fixture
def rag(chunks, embeddings, fake_index):
    upsert_chunks(fake_index, embeddings, chunks, namespace=NAMESPACE)
    return RAGSystem.from_pinecone(fake_index, embeddings, NAMESPACE, chunks=chunks)


def _ids(docs):
    return [d.metadata["chunk_id"] for d in docs]


def test_tokenizador_normaliza_y_conserva_codigos():
    tokens = tokenize_es("¿Qué significa el error CAU-417 en la configuración de tamano_lote?")

    assert "cau-417" in tokens
    assert "configuracion" in tokens  # sin tilde
    assert "tamano_lote" in tokens
    assert "el" not in tokens and "que" not in tokens  # stopwords


def test_devuelve_top_5_sin_duplicados(rag):
    docs = rag.retrieve("¿Qué significa el error CAU-417?")

    assert len(docs) == 5
    assert len(set(_ids(docs))) == 5
    for doc in docs:
        assert doc.page_content
        assert {"doc_id", "source", "page", "category"} <= set(doc.metadata)


def test_hibrido_combina_resultados_lexicos_y_semanticos(rag):
    query = "¿Qué significa el error CAU-417?"
    vector_ids = set(_ids(rag.retrieve(query, mode="vector")))
    bm25_ids = _ids(rag.retrieve(query, mode="bm25"))

    hybrid_ids = set(_ids(rag.retrieve(query)))

    assert hybrid_ids & vector_ids, "el híbrido tiene que incluir resultados de Pinecone"
    assert hybrid_ids & set(bm25_ids), "el híbrido tiene que incluir resultados de BM25"
    assert hybrid_ids <= vector_ids | set(bm25_ids)
    # BM25 encuentra el código exacto aunque los embeddings falsos no sepan nada de él
    assert bm25_ids[0] == "caudal-errores-p1-c0"
    assert "caudal-errores-p1-c0" in hybrid_ids


def test_consulta_a_pinecone_en_el_namespace_configurado(rag, fake_index):
    rag.retrieve("backpressure en kafka")

    assert fake_index.query_calls
    assert all(call["namespace"] == NAMESPACE for call in fake_index.query_calls)
    assert all(call["top_k"] == 5 for call in fake_index.query_calls)


def test_texto_recuperado_de_pinecone_viene_de_la_metadata(rag, chunks):
    by_id = {c.metadata["chunk_id"]: c.page_content for c in chunks}

    for doc in rag.retrieve("checkpoints cifrados", mode="vector"):
        assert doc.page_content == by_id[doc.metadata["chunk_id"]]


def test_pagina_vuelve_como_int_aunque_pinecone_la_guarde_como_float(rag):
    for mode in ("vector", "hibrido"):
        for doc in rag.retrieve("checkpoints cifrados", mode=mode):
            assert type(doc.metadata["page"]) is int
            assert type(doc.metadata["chunk_index"]) is int


def test_consulta_vacia_falla(rag):
    with pytest.raises(ValueError):
        rag.retrieve("   ")


def test_sin_chunks_falla(embeddings, fake_index):
    with pytest.raises(ValueError):
        RAGSystem.from_pinecone(fake_index, embeddings, NAMESPACE, chunks=[])
