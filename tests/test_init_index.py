import pytest
from pinecone import ServerlessSpec

from init_index import IndexMismatchError, ensure_index
from tests.fakes import FakePinecone


def test_crea_el_indice_serverless_si_no_existe():
    pc = FakePinecone()

    created = ensure_index(pc, "pre-entrega-4", poll_interval=0)

    assert created is True
    assert len(pc.create_calls) == 1
    call = pc.create_calls[0]
    assert call["name"] == "pre-entrega-4"
    assert call["dimension"] == 1536
    assert call["metric"] == "cosine"
    assert isinstance(call["spec"], ServerlessSpec)
    assert call["spec"].cloud == "aws"
    assert call["spec"].region == "us-east-1"


def test_no_recrea_un_indice_existente():
    pc = FakePinecone(existing={"pre-entrega-4": {"dimension": 1536, "metric": "cosine"}})

    created = ensure_index(pc, "pre-entrega-4")

    assert created is False
    assert pc.create_calls == []


def test_dos_inicializaciones_crean_una_sola_vez():
    pc = FakePinecone()

    assert ensure_index(pc, "idx", poll_interval=0) is True
    assert ensure_index(pc, "idx", poll_interval=0) is False
    assert len(pc.create_calls) == 1


def test_espera_a_que_el_indice_este_listo():
    pc = FakePinecone(ready_after=2)

    ensure_index(pc, "idx", poll_interval=0)

    assert pc.ready_after == 0  # consultó el estado hasta que quedó listo


@pytest.mark.parametrize("existing", [{"dimension": 768, "metric": "cosine"}, {"dimension": 1536, "metric": "dotproduct"}])
def test_falla_si_el_indice_existe_con_otra_dimension_o_metrica(existing):
    pc = FakePinecone(existing={"idx": existing})

    with pytest.raises(IndexMismatchError):
        ensure_index(pc, "idx")
    assert pc.create_calls == []
