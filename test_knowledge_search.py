
from document_reader import extract_text
from text_chunker import chunk_text
from embedding_engine import generate_embeddings

from knowledge_store import (
    store_document,
    get_storage_statistics
)

from semantic_search import search_knowledge


# 1. Select document
file_path = input(
    "Enter complete document path: "
).strip().strip('"')


try:

    # 2. Read document
    print("\nReading document...")

    document_text = extract_text(file_path)

    # 3. Create chunks
    print("\nCreating chunks...")

    chunks = chunk_text(
        document_text,
        chunk_size=1200,
        overlap=200
    )

    # 4. Generate document embeddings
    print("\nGenerating document embeddings...")

    embeddings = generate_embeddings(chunks)

    # 5. Store in knowledge database
    print("\nStoring document in knowledge database...")

    document_id = store_document(
        file_path,
        chunks,
        embeddings
    )

    print("Document ID:", document_id)

    # 6. Display database statistics
    print("\n========== DATABASE STATISTICS ==========")

    statistics = get_storage_statistics()

    for key, value in statistics.items():
        print(f"{key}: {value}")

    # 7. Ask a question
    query = input(
        "\nEnter your question about the document: "
    ).strip()

    if not query:
        raise ValueError("Please enter a search question.")

    # 8. Generate query embedding
    print("\nConverting question into embedding...")

    query_embedding = generate_embeddings([query])[0]

    # 9. Search relevant chunks
    print("\nSearching knowledge database...")

    results = search_knowledge(
        query_embedding,
        top_k=5
    )

    # 10. Display results
    print("\n========== SEARCH RESULTS ==========")

    if not results:
        print("No matching chunks found.")

    for index, result in enumerate(results, start=1):

        print(f"\nResult {index}")

        print("Document:", result["filename"])

        print("Chunk:", result["chunk_index"])

        print(
            "Similarity:",
            round(result["similarity"], 4)
        )

        print("\nContent:")
        print(result["content"])

    print("\nSUCCESS: Storage and semantic search completed.")

except Exception as error:

    print("\nTEST FAILED")
    print(error)
