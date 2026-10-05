"""Inicialización del índice: lo crea en Pinecone Serverless solo si no existe.

Uso: uv run python init_index.py

Es idempotente: si el índice ya existe no lo toca, pero verifica que su dimensión y su
métrica coincidan con las del modelo de embeddings (el error más común con Pinecone es
el mismatch de dimensiones, que recién aparece al primer upsert).
"""

import logging
import sys
import time
from typing import Any

from pinecone import ServerlessSpec

from config import CLOUD, DIMENSION, METRIC, REGION, build_pinecone_client, load_settings

logger = logging.getLogger(__name__)


class IndexMismatchError(RuntimeError):
    """El índice existe pero con otra dimensión o métrica."""


def _field(obj: Any, name: str) -> Any:
    """describe_index devuelve un objeto del SDK; los fakes de los tests, un dict."""
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def ensure_index(
    pc: Any,
    name: str,
    dimension: int = DIMENSION,
    metric: str = METRIC,
    cloud: str = CLOUD,
    region: str = REGION,
    wait_timeout: float = 120.0,
    poll_interval: float = 2.0,
) -> bool:
    """Crea el índice serverless si no existe. Devuelve True si lo creó."""
    if pc.has_index(name):
        description = pc.describe_index(name)
        actual_dim = _field(description, "dimension")
        actual_metric = str(_field(description, "metric"))
        if actual_dim != dimension or actual_metric != metric:
            raise IndexMismatchError(
                f"El índice {name!r} existe con dimension={actual_dim} y metric={actual_metric}, "
                f"pero el sistema espera dimension={dimension} y metric={metric}. "
                "Usá otro INDEX_NAME o borralo desde la consola de Pinecone."
            )
        logger.info("El índice %r ya existe (dimension=%s, metric=%s): no se recrea.", name, actual_dim, actual_metric)
        return False

    logger.info("Creando índice serverless %r (%s/%s, dimension=%s, metric=%s)...", name, cloud, region, dimension, metric)
    pc.create_index(
        name=name,
        dimension=dimension,
        metric=metric,
        spec=ServerlessSpec(cloud=cloud, region=region),
    )
    _wait_until_ready(pc, name, wait_timeout, poll_interval)
    logger.info("Índice %r creado y listo.", name)
    return True


def _wait_until_ready(pc: Any, name: str, timeout: float, poll_interval: float) -> None:
    deadline = time.monotonic() + timeout
    while True:
        status = _field(pc.describe_index(name), "status")
        if _field(status, "ready"):
            return
        if time.monotonic() > deadline:
            raise TimeoutError(f"El índice {name!r} no quedó listo en {timeout} s.")
        time.sleep(poll_interval)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stdout)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    settings = load_settings()
    pc = build_pinecone_client(settings)
    created = ensure_index(pc, settings.index_name)
    print(f"Índice: {settings.index_name} | creado ahora: {'sí' if created else 'no (ya existía)'}")
    print(f"Namespace que usa el sistema: {settings.namespace}")


if __name__ == "__main__":
    main()
