"""Evaluación del recuperador con el golden set: Recall@5 y Precision@5.

Uso:
    uv run python evaluate.py                         # híbrido, vector y bm25
    uv run python evaluate.py --modos hibrido         # solo el híbrido
    uv run python evaluate.py --salida docs/evaluacion.txt

Las definiciones de las métricas están en metrics.py.
"""

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from metrics import max_precision_at_k, mean, precision_at_k, recall_at_k
from rag_system import TOP_K, Mode, RAGSystem

GOLDEN_SET_PATH = Path(__file__).parent / "golden_set.json"
ALL_MODES: tuple[Mode, ...] = ("hibrido", "vector", "bm25")


@dataclass
class QuestionResult:
    pregunta: str
    esperado: str
    relevantes: list[str]
    recuperados: list[str]
    recall: float
    precision: float
    precision_maxima: float


def load_golden_set(path: Path = GOLDEN_SET_PATH) -> list[dict[str, Any]]:
    items = json.loads(path.read_text(encoding="utf-8"))
    for item in items:
        if not item.get("pregunta") or not item.get("documento_id_esperado"):
            raise ValueError(f"Item inválido en el golden set: {item}")
    return items


def evaluate(rag: RAGSystem, golden_set: list[dict[str, Any]], mode: Mode = "hibrido", k: int = TOP_K) -> list[QuestionResult]:
    chunks_per_doc = Counter(doc.metadata["doc_id"] for doc in rag.bm25_retriever.docs)
    results = []
    for item in golden_set:
        expected = item["documento_id_esperado"]
        relevant = item.get("documentos_relevantes") or [expected]
        retrieved = [doc.metadata["doc_id"] for doc in rag.retrieve(item["pregunta"], mode=mode)]
        results.append(
            QuestionResult(
                pregunta=item["pregunta"],
                esperado=expected,
                relevantes=relevant,
                recuperados=retrieved,
                recall=recall_at_k(retrieved, expected, k),
                precision=precision_at_k(retrieved, relevant, k),
                precision_maxima=max_precision_at_k(chunks_per_doc, relevant, k),
            )
        )
    return results


def format_report(results_by_mode: dict[str, list[QuestionResult]], k: int = TOP_K) -> str:
    lines: list[str] = []
    for mode, results in results_by_mode.items():
        lines.append(f"=== Modo: {mode} ===")
        for number, r in enumerate(results, start=1):
            mark = "OK " if r.recall else "MISS"
            lines.append(f"[{mark}] P{number}: {r.pregunta}")
            lines.append(f"       esperado={r.esperado} | Recall@{k}={r.recall:.0f} | Precision@{k}={r.precision:.2f} (máx {r.precision_maxima:.2f})")
            lines.append(f"       top-{k}: {', '.join(r.recuperados)}")
        lines.append("")

    lines.append(f"=== Resumen ({len(next(iter(results_by_mode.values())))} preguntas, k={k}) ===")
    lines.append(f"{'modo':<9} | {'Recall@' + str(k):>9} | {'Precision@' + str(k):>12} | {'Precisión máx.':>14}")
    lines.append("-" * 54)
    for mode, results in results_by_mode.items():
        lines.append(
            f"{mode:<9} | {mean([r.recall for r in results]):>9.2f} | "
            f"{mean([r.precision for r in results]):>12.2f} | {mean([r.precision_maxima for r in results]):>14.2f}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Recall@5 y Precision@5 sobre el golden set")
    parser.add_argument("--modos", nargs="+", choices=ALL_MODES, default=list(ALL_MODES))
    parser.add_argument("--golden", type=Path, default=GOLDEN_SET_PATH)
    parser.add_argument("--salida", type=Path, help="además de imprimir, guardar el reporte en este archivo")
    args = parser.parse_args()

    golden_set = load_golden_set(args.golden)
    rag = RAGSystem.from_env()
    report = format_report({mode: evaluate(rag, golden_set, mode) for mode in args.modos})
    print(report)
    if args.salida:
        args.salida.parent.mkdir(parents=True, exist_ok=True)
        args.salida.write_text(report + "\n", encoding="utf-8")
        print(f"\nReporte guardado en {args.salida}")


if __name__ == "__main__":
    main()
