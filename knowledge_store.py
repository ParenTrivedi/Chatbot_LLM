
import sqlite3
import json
import hashlib
from pathlib import Path
from datetime import datetime


BASE_DIR = Path(__file__).resolve().parent
DATABASE_DIR = BASE_DIR / "database"
DATABASE_PATH = DATABASE_DIR / "knowledge.db"


def get_connection():
    DATABASE_DIR.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(DATABASE_PATH)
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize_knowledge_db():
    with get_connection() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                file_hash TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL
            )
        """)

        connection.execute("""
            CREATE TABLE IF NOT EXISTS document_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                embedding TEXT NOT NULL,
                FOREIGN KEY(document_id)
                    REFERENCES documents(id)
                    ON DELETE CASCADE
            )
        """)


def calculate_file_hash(file_path):
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:
        for block in iter(
            lambda: file.read(1024 * 1024),
            b""
        ):
            sha256.update(block)

    return sha256.hexdigest()


def store_document(file_path, chunks, embeddings):
    if len(chunks) != len(embeddings):
        raise ValueError(
            "Chunks and embeddings must have equal lengths."
        )

    if not chunks:
        raise ValueError("No chunks to store.")

    if any(
        len(vector) != len(embeddings[0])
        for vector in embeddings
    ):
        raise ValueError("Embedding dimensions do not match.")

    file_path = Path(file_path)
    file_hash = calculate_file_hash(file_path)
    now = datetime.now().isoformat(timespec="seconds")

    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO documents
            (filename, file_hash, created_at)
            VALUES (?, ?, ?)
            ON CONFLICT(file_hash)
            DO UPDATE SET
                filename = excluded.filename,
                created_at = excluded.created_at
            """,
            (file_path.name, file_hash, now)
        )

        row = connection.execute(
            """
            SELECT id
            FROM documents
            WHERE file_hash = ?
            """,
            (file_hash,)
        ).fetchone()

        document_id = row[0]

        # Replace existing chunks if this document
        # has already been indexed.
        connection.execute(
            """
            DELETE FROM document_chunks
            WHERE document_id = ?
            """,
            (document_id,)
        )

        records = [
            (
                document_id,
                index,
                chunk,
                json.dumps(embedding)
            )
            for index, (chunk, embedding)
            in enumerate(zip(chunks, embeddings))
        ]

        connection.executemany(
            """
            INSERT INTO document_chunks
            (document_id, chunk_index, content, embedding)
            VALUES (?, ?, ?, ?)
            """,
            records
        )

    return document_id


def delete_document(document_id):
    """Delete one indexed document and all its chunks."""

    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT id
            FROM documents
            WHERE id = ?
            """,
            (document_id,)
        ).fetchone()

        if row is None:
            return False

        # Explicitly delete chunks before deleting the document.
        connection.execute(
            """
            DELETE FROM document_chunks
            WHERE document_id = ?
            """,
            (document_id,)
        )

        connection.execute(
            """
            DELETE FROM documents
            WHERE id = ?
            """,
            (document_id,)
        )

    return True


def get_document_by_id(document_id):
    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT id, filename, created_at
            FROM documents
            WHERE id = ?
            """,
            (document_id,)
        ).fetchone()

    if row is None:
        return None

    return {
        "id": row[0],
        "filename": row[1],
        "created_at": row[2]
    }


def get_storage_statistics():
    with get_connection() as connection:
        document_count = connection.execute(
            "SELECT COUNT(*) FROM documents"
        ).fetchone()[0]

        chunk_count = connection.execute(
            "SELECT COUNT(*) FROM document_chunks"
        ).fetchone()[0]

    return {
        "documents": document_count,
        "chunks": chunk_count,
        "database": str(DATABASE_PATH)
    }


initialize_knowledge_db()
