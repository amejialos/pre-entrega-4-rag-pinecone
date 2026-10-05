"""Fakes de Pinecone para probar sin red ni keys.

FakeIndex implementa lo que usan ingest.py y PineconeVectorStore: upsert, query
(búsqueda por coseno de verdad, en memoria), describe_index_stats, delete y config.
Registra cada llamada para que los tests verifiquen namespaces y metadata.
"""

import copy
import math
from types import SimpleNamespace
from typing import Any


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


class FakeIndex:
    def __init__(self) -> None:
        self.config = SimpleNamespace(host="fake-host", api_key="fake-key")
        self.namespaces: dict[str, dict[str, dict[str, Any]]] = {}
        self.upsert_calls: list[dict[str, Any]] = []
        self.query_calls: list[dict[str, Any]] = []

    def upsert(self, vectors: list[dict[str, Any]], namespace: str) -> dict[str, int]:
        self.upsert_calls.append({"vectors": vectors, "namespace": namespace})
        store = self.namespaces.setdefault(namespace, {})
        for record in vectors:
            stored = copy.deepcopy(record)
            # Como Pinecone: los números de la metadata vuelven como float.
            stored["metadata"] = {
                key: float(value) if isinstance(value, int) and not isinstance(value, bool) else value
                for key, value in stored["metadata"].items()
            }
            store[record["id"]] = stored
        return {"upserted_count": len(vectors)}

    def query(self, vector: list[float], top_k: int, include_metadata: bool, namespace: str | None, filter: Any = None) -> dict:
        self.query_calls.append({"top_k": top_k, "namespace": namespace})
        records = self.namespaces.get(namespace or "", {}).values()
        scored = sorted(records, key=lambda r: _cosine(vector, r["values"]), reverse=True)[:top_k]
        return {
            "matches": [
                # copia: PineconeVectorStore hace metadata.pop("text")
                {"id": r["id"], "score": _cosine(vector, r["values"]), "metadata": dict(r["metadata"])}
                for r in scored
            ]
        }

    def describe_index_stats(self) -> dict:
        return {"namespaces": {ns: {"vector_count": len(recs)} for ns, recs in self.namespaces.items()}}

    def delete(self, delete_all: bool, namespace: str) -> None:
        if namespace not in self.namespaces:
            raise RuntimeError("(404) Reason: Not Found")
        self.namespaces.pop(namespace)


class FakePinecone:
    """Cliente falso: guarda los índices creados y cuenta las llamadas a create_index."""

    def __init__(self, existing: dict[str, dict[str, Any]] | None = None, ready_after: int = 0) -> None:
        self.indexes: dict[str, dict[str, Any]] = dict(existing or {})
        self.create_calls: list[dict[str, Any]] = []
        self.ready_after = ready_after  # cuántos describe_index devuelven "no listo"
        self._fake_index = FakeIndex()

    def has_index(self, name: str) -> bool:
        return name in self.indexes

    def describe_index(self, name: str) -> dict[str, Any]:
        description = self.indexes[name]
        if self.ready_after > 0:
            self.ready_after -= 1
            return {**description, "status": {"ready": False}}
        return {**description, "status": {"ready": True}}

    def create_index(self, name: str, dimension: int, metric: str, spec: Any) -> None:
        self.create_calls.append({"name": name, "dimension": dimension, "metric": metric, "spec": spec})
        self.indexes[name] = {"name": name, "dimension": dimension, "metric": metric}

    def Index(self, name: str) -> FakeIndex:  # noqa: N802 - mismo nombre que el SDK
        return self._fake_index
