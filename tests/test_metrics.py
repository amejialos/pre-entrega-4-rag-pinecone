import pytest
from langchain_core.documents import Document

from evaluate import evaluate, format_report, load_golden_set
from metrics import max_precision_at_k, mean, precision_at_k, recall_at_k


def test_recall_es_1_si_el_esperado_esta_en_el_top_k():
    assert recall_at_k(["a", "b", "c", "d", "e"], "c") == 1.0
    assert recall_at_k(["a", "b", "c", "d", "e"], "z") == 0.0


def test_recall_solo_mira_los_primeros_k():
    assert recall_at_k(["a", "b", "c", "d", "e", "z"], "z", k=5) == 0.0


def test_precision_cuenta_chunks_utiles_sobre_k():
    retrieved = ["err", "err", "otro", "err", "otro"]
    assert precision_at_k(retrieved, ["err"]) == pytest.approx(3 / 5)
    assert precision_at_k(retrieved, ["err", "otro"]) == 1.0
    assert precision_at_k(retrieved, ["nada"]) == 0.0


def test_precision_con_menos_de_k_resultados_divide_por_k():
    assert precision_at_k(["err", "err"], ["err"], k=5) == pytest.approx(2 / 5)


def test_precision_maxima_alcanzable():
    chunks_per_doc = {"a": 3, "b": 3, "c": 1}
    assert max_precision_at_k(chunks_per_doc, ["a"]) == pytest.approx(3 / 5)
    assert max_precision_at_k(chunks_per_doc, ["a", "b"]) == 1.0
    assert max_precision_at_k(chunks_per_doc, ["c"]) == pytest.approx(1 / 5)


def test_mean():
    assert mean([1.0, 0.0, 1.0, 1.0]) == 0.75
    assert mean([]) == 0.0


def test_golden_set_tiene_5_preguntas_con_documento_existente(chunks):
    golden = load_golden_set()
    doc_ids = {c.metadata["doc_id"] for c in chunks}

    assert len(golden) == 5
    for item in golden:
        assert item["documento_id_esperado"] in doc_ids
        assert set(item.get("documentos_relevantes", [])) <= doc_ids


class _StubRAG:
    """RAG falso: devuelve listas de doc_id fijas por pregunta."""

    def __init__(self, answers, corpus_doc_ids):
        self.answers = answers
        self.bm25_retriever = type("B", (), {"docs": [Document(page_content="x", metadata={"doc_id": d}) for d in corpus_doc_ids]})()

    def retrieve(self, query, mode="hibrido"):
        return [Document(page_content="x", metadata={"doc_id": d}) for d in self.answers[query]]


def test_evaluate_calcula_metricas_por_pregunta():
    golden = [
        {"pregunta": "q1", "documento_id_esperado": "a"},
        {"pregunta": "q2", "documento_id_esperado": "b", "documentos_relevantes": ["b", "c"]},
    ]
    rag = _StubRAG(
        {"q1": ["a", "a", "x", "x", "x"], "q2": ["x", "x", "c", "x", "x"]},
        corpus_doc_ids=["a", "a", "a", "b", "b", "c", "x"],
    )

    results = evaluate(rag, golden)

    assert [r.recall for r in results] == [1.0, 0.0]  # en q2 aparece c, pero el esperado era b
    assert [r.precision for r in results] == [pytest.approx(0.4), pytest.approx(0.2)]
    assert [r.precision_maxima for r in results] == [pytest.approx(0.6), pytest.approx(0.6)]
    report = format_report({"hibrido": results})
    assert "Recall@5" in report and "0.50" in report
