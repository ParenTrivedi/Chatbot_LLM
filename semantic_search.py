
import json
import math
import re
from collections import Counter

from knowledge_store import get_connection


# --------------------------------------------------
# 1. TEXT PROCESSING
# --------------------------------------------------

STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were",
    "am", "be", "been", "being", "do", "does", "did",
    "to", "of", "in", "on", "at", "for", "from",
    "with", "by", "and", "or", "but", "if", "then",
    "this", "that", "these", "those", "it", "its",
    "i", "me", "my", "we", "our", "you", "your",
    "he", "she", "they", "them", "their", "what",
    "which", "who", "when", "where", "why", "how",
    "can", "could", "would", "should", "will",
    "please", "document", "documents"
}


def normalize_token(token):
    token = token.lower()

    if len(token) > 5 and token.endswith("ies"):
        return token[:-3] + "y"

    if len(token) > 5 and token.endswith("ing"):
        return token[:-3]

    if len(token) > 4 and token.endswith("ed"):
        return token[:-2]

    if len(token) > 4 and token.endswith("es"):
        return token[:-2]

    if len(token) > 3 and token.endswith("s"):
        return token[:-1]

    return token


def tokenize(text):
    words = re.findall(r"[a-zA-Z0-9]+", text.lower())

    return [
        normalize_token(word)
        for word in words
        if word not in STOP_WORDS
    ]


# --------------------------------------------------
# 2. DOCUMENT ID VALIDATION
# --------------------------------------------------

def normalize_document_ids(document_ids):

    if document_ids is None:
        return None

    if isinstance(document_ids, int):
        document_ids = [document_ids]

    normalized = []

    for document_id in document_ids:

        try:
            document_id = int(document_id)
        except (TypeError, ValueError):
            continue

        if document_id > 0 and document_id not in normalized:
            normalized.append(document_id)

    return normalized


# --------------------------------------------------
# 3. LIST AVAILABLE DOCUMENTS
# --------------------------------------------------

def list_documents():

    with get_connection() as connection:

        rows = connection.execute("""
            SELECT
                documents.id,
                documents.filename,
                documents.created_at,
                COUNT(document_chunks.id) AS chunk_count

            FROM documents

            LEFT JOIN document_chunks
            ON documents.id = document_chunks.document_id

            GROUP BY
                documents.id,
                documents.filename,
                documents.created_at

            ORDER BY documents.id ASC
        """).fetchall()

    return [
        {
            "id": row[0],
            "filename": row[1],
            "created_at": row[2],
            "chunk_count": row[3]
        }
        for row in rows
    ]


# --------------------------------------------------
# 4. LOAD STORED CHUNKS
# --------------------------------------------------

def load_document_chunks(document_ids=None):

    document_ids = normalize_document_ids(document_ids)

    if document_ids == []:
        return []

    with get_connection() as connection:

        if document_ids is None:

            rows = connection.execute("""
                SELECT
                    documents.id,
                    documents.filename,
                    document_chunks.chunk_index,
                    document_chunks.content,
                    document_chunks.embedding

                FROM document_chunks

                JOIN documents
                ON documents.id = document_chunks.document_id

                ORDER BY
                    documents.id,
                    document_chunks.chunk_index
            """).fetchall()

        else:

            placeholders = ",".join(
                "?" for _ in document_ids
            )

            query = f"""
                SELECT
                    documents.id,
                    documents.filename,
                    document_chunks.chunk_index,
                    document_chunks.content,
                    document_chunks.embedding

                FROM document_chunks

                JOIN documents
                ON documents.id = document_chunks.document_id

                WHERE documents.id IN ({placeholders})

                ORDER BY
                    documents.id,
                    document_chunks.chunk_index
            """

            rows = connection.execute(
                query,
                document_ids
            ).fetchall()

    chunks = []

    for document_id, filename, chunk_index, content, embedding in rows:

        chunks.append({
            "document_id": document_id,
            "filename": filename,
            "chunk_index": chunk_index,
            "content": content,
            "embedding": json.loads(embedding)
        })

    return chunks


# --------------------------------------------------
# 5. LOAD ALL CHUNKS FOR A DOCUMENT
# --------------------------------------------------

def get_document_chunks(document_id):

    document_id = int(document_id)

    return load_document_chunks([document_id])


# --------------------------------------------------
# 6. COSINE SIMILARITY
# --------------------------------------------------

def cosine_similarity(vector_a, vector_b):

    if len(vector_a) != len(vector_b):
        return 0.0

    dot_product = sum(
        a * b
        for a, b in zip(vector_a, vector_b)
    )

    magnitude_a = math.sqrt(
        sum(a * a for a in vector_a)
    )

    magnitude_b = math.sqrt(
        sum(b * b for b in vector_b)
    )

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (
        magnitude_a * magnitude_b
    )


# --------------------------------------------------
# 7. SEMANTIC SEARCH
# --------------------------------------------------

def semantic_search(
    query_embedding,
    top_k=5,
    document_ids=None
):

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")

    if not query_embedding:
        raise ValueError("Query embedding cannot be empty.")

    chunks = load_document_chunks(document_ids)

    results = []

    for chunk in chunks:

        embedding = chunk["embedding"]

        if len(embedding) != len(query_embedding):
            continue

        score = cosine_similarity(
            query_embedding,
            embedding
        )

        results.append({
            "document_id": chunk["document_id"],
            "filename": chunk["filename"],
            "chunk_index": chunk["chunk_index"],
            "content": chunk["content"],
            "similarity": score
        })

    results.sort(
        key=lambda item: item["similarity"],
        reverse=True
    )

    return results[:top_k]


# --------------------------------------------------
# 8. KEYWORD SEARCH - BM25 STYLE
# --------------------------------------------------

def keyword_search(
    query,
    top_k=5,
    document_ids=None
):

    if not query or not query.strip():
        raise ValueError("Search query cannot be empty.")

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")

    query_tokens = tokenize(query)

    if not query_tokens:
        return []

    query_terms = set(query_tokens)

    chunks = load_document_chunks(document_ids)

    if not chunks:
        return []

    tokenized_chunks = []

    total_length = 0

    document_frequency = Counter()

    for chunk in chunks:

        tokens = tokenize(chunk["content"])

        frequencies = Counter(tokens)

        tokenized_chunks.append({
            **chunk,
            "tokens": tokens,
            "frequencies": frequencies,
            "length": len(tokens)
        })

        total_length += len(tokens)

        for term in set(tokens):
            document_frequency[term] += 1

    total_documents = len(tokenized_chunks)

    average_length = (
        total_length / total_documents
        if total_documents else 1
    )

    k1 = 1.5
    b = 0.75

    results = []

    for chunk in tokenized_chunks:

        score = 0.0

        frequencies = chunk["frequencies"]
        length = chunk["length"]

        for term in query_terms:

            frequency = frequencies.get(term, 0)

            if frequency == 0:
                continue

            df = document_frequency[term]

            idf = math.log(
                1 + (
                    total_documents - df + 0.5
                ) / (
                    df + 0.5
                )
            )

            denominator = frequency + k1 * (
                1 - b + b * length / average_length
            )

            score += idf * (
                frequency * (k1 + 1)
            ) / denominator

        normalized_content = " ".join(
            chunk["tokens"]
        )

        normalized_query = " ".join(query_tokens)

        if (
            normalized_query
            and normalized_query in normalized_content
        ):
            score += 2.0

        if score > 0:

            results.append({
                "document_id": chunk["document_id"],
                "filename": chunk["filename"],
                "chunk_index": chunk["chunk_index"],
                "content": chunk["content"],
                "similarity": 0.0,
                "keyword_score": score
            })

    results.sort(
        key=lambda item: item["keyword_score"],
        reverse=True
    )

    return results[:top_k]


# --------------------------------------------------
# 9. HYBRID SEARCH
# --------------------------------------------------

def hybrid_search(
    query,
    query_embedding,
    top_k=5,
    document_ids=None
):

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")

    candidate_count = max(top_k * 4, 20)

    semantic_results = semantic_search(
        query_embedding,
        top_k=candidate_count,
        document_ids=document_ids
    )

    keyword_results = keyword_search(
        query,
        top_k=candidate_count,
        document_ids=document_ids
    )

    combined = {}

    rank_constant = 60

    for rank, result in enumerate(
        semantic_results,
        start=1
    ):

        key = (
            result["document_id"],
            result["chunk_index"]
        )

        if key not in combined:

            combined[key] = {
                **result,
                "hybrid_score": 0.0,
                "semantic_similarity": result["similarity"],
                "keyword_score": 0.0
            }

        combined[key]["hybrid_score"] += (
            1 / (rank_constant + rank)
        )

    for rank, result in enumerate(
        keyword_results,
        start=1
    ):

        key = (
            result["document_id"],
            result["chunk_index"]
        )

        if key not in combined:

            combined[key] = {
                **result,
                "hybrid_score": 0.0,
                "semantic_similarity": 0.0,
                "keyword_score": result["keyword_score"]
            }

        combined[key]["hybrid_score"] += (
            1 / (rank_constant + rank)
        )

        combined[key]["keyword_score"] = (
            result["keyword_score"]
        )

    results = list(combined.values())

    results.sort(
        key=lambda item: item["hybrid_score"],
        reverse=True
    )

    return results[:top_k]


# --------------------------------------------------
# 10. EXACT DOCUMENT TEXT SEARCH
# --------------------------------------------------

def exact_document_search(
    query,
    document_ids=None
):

    if not query or not query.strip():
        raise ValueError("Search query cannot be empty.")

    query_tokens = set(tokenize(query))

    if not query_tokens:
        return "Please enter searchable words."

    chunks = load_document_chunks(document_ids)

    output = []

    for chunk in chunks:

        lines = chunk["content"].splitlines()

        matched_positions = []

        for index, line in enumerate(lines):

            line_tokens = set(tokenize(line))

            if query_tokens.intersection(line_tokens):
                matched_positions.append(index)

        if not matched_positions:
            continue

        selected_positions = set()

        for position in matched_positions:

            start = max(0, position - 2)
            end = min(len(lines), position + 6)

            for line_index in range(start, end):
                selected_positions.add(line_index)

        selected_lines = [
            lines[index]
            for index in sorted(selected_positions)
            if lines[index].strip()
        ]

        if selected_lines:

            output.append(
                f"Source: {chunk['filename']} "
                f"(Document ID: {chunk['document_id']}, "
                f"Chunk {chunk['chunk_index']})\n"
                + "\n".join(selected_lines)
            )

    if not output:
        return "No matching text found in the selected documents."

    unique_output = list(dict.fromkeys(output))

    return "\n\n--------------------\n\n".join(
        unique_output
    )


# --------------------------------------------------
# 11. BACKWARD-COMPATIBLE SEARCH
# --------------------------------------------------

def search_knowledge(query_embedding, top_k=5):

    return semantic_search(
        query_embedding,
        top_k=top_k
    )
