import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from config import DIMENSION
from corpus import load_chunks
from tests.fakes import FakeIndex


@pytest.fixture
def embeddings() -> DeterministicFakeEmbedding:
    # Vectores pseudoaleatorios pero estables por texto: sin red y reproducibles.
    return DeterministicFakeEmbedding(size=DIMENSION)


@pytest.fixture(scope="session")
def chunks() -> list[Document]:
    # length_function=len: no depende de que tiktoken tenga el tokenizador descargado.
    # 2800 caracteres ≈ 700 tokens en español.
    return load_chunks(chunk_size=2800, chunk_overlap=400, length_function=len)


@pytest.fixture
def fake_index() -> FakeIndex:
    return FakeIndex()
