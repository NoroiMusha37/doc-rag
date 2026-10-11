import uuid

from pgvector.sqlalchemy import Vector

from src.doc_rag.db.models import EMBEDDING_DIM, Base, Chunk, Document


def test_tables_are_documents_and_chunks() -> None:
    assert set(Base.metadata.tables) == {"documents", "chunks"}


def test_ids_are_uuids_generated_in_python() -> None:
    for model in (Document, Chunk):
        column = model.__table__.c.id
        assert column.primary_key
        assert column.type.python_type is uuid.UUID
        assert column.default is not None


def test_sha256_is_unique() -> None:
    assert Document.__table__.c.sha256.unique


def test_chunks_are_deleted_with_their_document() -> None:
    (foreign_key,) = Chunk.__table__.c.document_id.foreign_keys
    assert foreign_key.ondelete == "CASCADE"


def test_embedding_has_the_bge_m3_dimension() -> None:
    embedding_type = Chunk.__table__.c.embedding.type
    assert isinstance(embedding_type, Vector)
    assert embedding_type.dim == EMBEDDING_DIM == 1024


def test_tsv_is_generated_from_text() -> None:
    tsv = Chunk.__table__.c.tsv
    assert tsv.computed is not None
    assert tsv.computed.persisted
    assert "to_tsvector" in str(tsv.computed.sqltext)
