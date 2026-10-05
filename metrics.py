"""Métricas de recuperación sobre ids de documento (funciones puras, sin red).

Definiciones (para una pregunta, con la lista de doc_id de los k chunks recuperados):
- Recall@k: 1 si el documento esperado aparece entre los k chunks, 0 si no.
  Es la pregunta de la consigna: "¿está el documento correcto entre los 5?".
- Precision@k: fracción de los k chunks que son útiles. Un chunk es útil si su
  doc_id pertenece al conjunto de documentos relevantes de la pregunta (por defecto
  solo el esperado; el golden set puede listar más con `documentos_relevantes`).
  Si se recuperan menos de k chunks, el denominador sigue siendo k.
- Precisión máxima alcanzable: si los documentos relevantes tienen en total menos de
  k chunks, ni un recuperador perfecto llega a Precision@k = 1. Se informa para leer
  la precisión en contexto.
"""

from collections.abc import Iterable, Sequence


def recall_at_k(retrieved_doc_ids: Sequence[str], expected_doc_id: str, k: int = 5) -> float:
    return 1.0 if expected_doc_id in retrieved_doc_ids[:k] else 0.0


def precision_at_k(retrieved_doc_ids: Sequence[str], relevant_doc_ids: Iterable[str], k: int = 5) -> float:
    if k <= 0:
        raise ValueError("k tiene que ser positivo.")
    relevant = set(relevant_doc_ids)
    useful = sum(1 for doc_id in retrieved_doc_ids[:k] if doc_id in relevant)
    return useful / k


def max_precision_at_k(chunks_per_doc: dict[str, int], relevant_doc_ids: Iterable[str], k: int = 5) -> float:
    available = sum(chunks_per_doc.get(doc_id, 0) for doc_id in set(relevant_doc_ids))
    return min(available, k) / k


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0
