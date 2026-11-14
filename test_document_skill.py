
from skills.document_skill import DocumentSkill

document_skill = DocumentSkill()

print("DOCUMENT INTELLIGENCE TEST")
print("-" * 35)

documents = document_skill.handle("list")

for document in documents:
    print(
        f"ID: {document['id']} | "
        f"Name: {document['filename']} | "
        f"Chunks: {document['chunk_count']}"
    )

if documents:
    document_id = documents[0]["id"]

    print("\nTesting document summary...\n")

    result = document_skill.handle(
        action="summarize",
        document_ids=[document_id]
    )

    print(result)

else:
    print("\nNo documents found. Upload a document first.")
