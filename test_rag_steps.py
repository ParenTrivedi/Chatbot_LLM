
from document_reader import extract_text
from text_chunker import chunk_text
from embedding_engine import generate_embeddings


file_path = input(
    "Enter complete document path: "
).strip().strip('"')


try:
    # STEP 1: Extract document text
    print("\nReading document...")

    document_text = extract_text(file_path)

    print("Document reading successful.")

    # STEP 2: Divide text into chunks
    print("\nCreating text chunks...")

    chunks = chunk_text(
        document_text,
        chunk_size=1200,
        overlap=200
    )

    print(f"Total chunks created: {len(chunks)}")

    # STEP 3: Generate embeddings
    print("\nGenerating Nomic embeddings...")

    embeddings = generate_embeddings(chunks)

    print("\n========== TEST RESULTS ==========")

    print("Document characters:", len(document_text))
    print("Total chunks:", len(chunks))
    print("Total embeddings:", len(embeddings))

    if embeddings:
        print(
            "Embedding dimensions:",
            len(embeddings[0])
        )

    print("\nSUCCESS: Chunking and embeddings completed.")

except Exception as error:
    print("\nTEST FAILED")
    print(error)
