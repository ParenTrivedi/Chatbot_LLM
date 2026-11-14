
from document_reader import extract_text, get_document_info

file_path = input("Enter complete document path: ").strip().strip('"')

try:
    info = get_document_info(file_path)
    text = extract_text(file_path)

    print("\n========== DOCUMENT INFORMATION ==========")
    print(info)

    print("\n========== EXTRACTED CONTENT ==========")
    print(text[:5000])

    print("\n========== TEST RESULT ==========")
    print("SUCCESS: Document text extracted.")

except Exception as error:
    print("\nDOCUMENT READING FAILED")
    print(error)
