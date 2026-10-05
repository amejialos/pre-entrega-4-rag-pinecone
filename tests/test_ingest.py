import pytest

from ingest import build_records, namespace_count, reset_namespace, upsert_chunks, wait_for_count


def test_registros_llevan_texto_fuente_pagina_y_categoria(chunks, embeddings):
    vectors = embeddings.embed_documents([c.page_content for c in chunks])

    records = build_records(chunks, vectors)

    assert len(records) == len(chunks)
    record, chunk = records[0], chunks[0]
    assert record["id"] == chunk.metadata["chunk_id"]
    assert len(record["values"]) == 1536
    meta = record["metadata"]
    assert meta["text"] == chunk.page_content
    assert meta["source"] == chunk.metadata["source"]
    assert meta["page"] == chunk.metadata["page"]
    assert meta["category"] == chunk.metadata["category"]
    assert meta["doc_id"] == chunk.metadata["doc_id"]
    assert meta["chunk_id"] == chunk.metadata["chunk_id"]


def test_rechaza_vectores_de_otra_dimension(chunks):
    with pytest.raises(ValueError, match="dimensiones"):
        build_records(chunks[:1], [[0.1] * 768])


def test_upsert_usa_el_namespace_y_lotes(chunks, embeddings, fake_index):
    total = upsert_chunks(fake_index, embeddings, chunks, namespace="caudal-docs", batch_size=5)

    assert total == len(chunks)
    assert {call["namespace"] for call in fake_index.upsert_calls} == {"caudal-docs"}
    assert all(len(call["vectors"]) <= 5 for call in fake_index.upsert_calls)
    assert len(fake_index.upsert_calls) == -(-len(chunks) // 5)
    assert "" not in fake_index.namespaces  # nada en el namespace por defecto
    assert namespace_count(fake_index, "caudal-docs") == len(chunks)


def test_reingesta_es_idempotente(chunks, embeddings, fake_index):
    upsert_chunks(fake_index, embeddings, chunks, namespace="ns")
    upsert_chunks(fake_index, embeddings, chunks, namespace="ns")

    assert namespace_count(fake_index, "ns") == len(chunks)


def test_reset_tolera_namespace_inexistente_y_vacia_el_existente(chunks, embeddings, fake_index):
    reset_namespace(fake_index, "no-existe")  # no lanza

    upsert_chunks(fake_index, embeddings, chunks, namespace="ns")
    reset_namespace(fake_index, "ns")
    assert namespace_count(fake_index, "ns") == 0


def test_wait_for_count_devuelve_el_conteo_visible(chunks, embeddings, fake_index):
    upsert_chunks(fake_index, embeddings, chunks, namespace="ns")

    assert wait_for_count(fake_index, "ns", len(chunks), timeout=0) == len(chunks)
