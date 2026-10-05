from collections import Counter

from corpus import load_documents, split_documents

EXPECTED_DOCS = {
    "caudal-api-flujos", "caudal-changelog", "caudal-conectores", "caudal-errores",
    "caudal-instalacion", "caudal-rendimiento", "caudal-seguridad",
}


def test_carga_markdown_y_json_con_metadata():
    pages = load_documents()

    assert {p.metadata["doc_id"] for p in pages} == EXPECTED_DOCS
    sources = {p.metadata["source"] for p in pages}
    assert "caudal-changelog.json" in sources and "caudal-errores.md" in sources
    for page in pages:
        assert set(page.metadata) >= {"doc_id", "source", "page", "category", "section", "title"}
        assert page.metadata["page"] >= 1
        assert page.page_content.strip()


def test_categorias_vienen_del_front_matter():
    categories = {p.metadata["doc_id"]: p.metadata["category"] for p in load_documents()}

    assert categories["caudal-errores"] == "troubleshooting"
    assert categories["caudal-changelog"] == "changelog"


def test_chunks_tienen_ids_unicos_y_estables(chunks):
    ids = [c.metadata["chunk_id"] for c in chunks]

    assert len(ids) == len(set(ids))
    assert all(c.id == c.metadata["chunk_id"] for c in chunks)
    again = split_documents(load_documents(), chunk_size=2800, chunk_overlap=400, length_function=len)
    assert [c.metadata["chunk_id"] for c in again] == ids


def test_una_pagina_larga_se_parte_en_varios_chunks():
    pages = load_documents()

    small = split_documents(pages, chunk_size=500, chunk_overlap=50, length_function=len)

    per_page = Counter((c.metadata["doc_id"], c.metadata["page"]) for c in small)
    assert max(per_page.values()) > 1
    first = [c for c in small if c.metadata["doc_id"] == "caudal-errores" and c.metadata["page"] == 1]
    assert [c.metadata["chunk_id"] for c in first[:2]] == ["caudal-errores-p1-c0", "caudal-errores-p1-c1"]
    assert all(len(c.page_content) <= 500 for c in small)
