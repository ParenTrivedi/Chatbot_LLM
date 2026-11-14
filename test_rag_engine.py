
import re

from rag_engine import (
    generate_rag_answer,
    find_exact_document_lines,
    get_available_documents,
    generate_document_summary,
    generate_document_topic_answer,
    generate_document_comparison,
)


# --------------------------------------------------
# 1. CHATBOT SETTINGS
# --------------------------------------------------

print("========== RAG CHATBOT ==========")

mode = "hybrid"

conversation_history = []

selected_document_ids = []

active_document_id = None


# --------------------------------------------------
# 2. DOCUMENT REFERENCE DETECTION
# --------------------------------------------------

ORDINALS = {
    "first": 1,
    "1st": 1,
    "second": 2,
    "2nd": 2,
    "third": 3,
    "3rd": 3,
    "fourth": 4,
    "4th": 4,
    "fifth": 5,
    "5th": 5,
    "sixth": 6,
    "6th": 6,
    "seventh": 7,
    "7th": 7,
    "eighth": 8,
    "8th": 8,
    "ninth": 9,
    "9th": 9,
    "tenth": 10,
    "10th": 10,
}


def find_referenced_documents(question, documents):

    question_lower = question.lower()

    selected = []

    has_document_word = bool(
        re.search(
            r"\b(paper|papers|document|documents)\b",
            question_lower
        )
    )

    if has_document_word:

        for word, number in ORDINALS.items():

            if re.search(
                rf"\b{re.escape(word)}\b",
                question_lower
            ):

                if number <= len(documents):

                    document = documents[number - 1]

                    if document["id"] not in selected:
                        selected.append(document["id"])

        numeric_matches = re.findall(
            r"\b(?:paper|document)\s*(?:number\s*)?(\d+)\b",
            question_lower
        )

        for number_text in numeric_matches:

            number = int(number_text)

            if 1 <= number <= len(documents):

                document = documents[number - 1]

                if document["id"] not in selected:
                    selected.append(document["id"])

    # Match filenames.

    for document in documents:

        filename = document["filename"].lower()

        filename_without_extension = re.sub(
            r"\.[^.]+$",
            "",
            filename
        )

        if (
            filename in question_lower
            or (
                len(filename_without_extension) >= 4
                and filename_without_extension in question_lower
            )
        ):

            if document["id"] not in selected:
                selected.append(document["id"])

    return selected


# --------------------------------------------------
# 3. DOCUMENT LIST DISPLAY
# --------------------------------------------------

def show_documents():

    documents = get_available_documents()

    if not documents:

        print("\nNo documents have been uploaded yet.")

        return

    print("\n========== UPLOADED DOCUMENTS ==========")

    for number, document in enumerate(documents, start=1):

        print(f"\n{number}. {document['filename']}")

        print(f"   Document ID: {document['id']}")

        print(f"   Chunks: {document['chunk_count']}")

        print(f"   Added: {document['created_at']}")

    print("\n========================================")


# --------------------------------------------------
# 4. DETECT DOCUMENT READING REQUESTS
# --------------------------------------------------

def is_read_request(question):

    text = question.lower()

    read_patterns = [
        r"\bread\b",
        r"\bsummarize\b",
        r"\bsummarise\b",
        r"\bsummary\b",
        r"\boverview\b",
        r"\bexplain this document\b",
        r"\bexplain this paper\b",
        r"\bexplain the document\b",
        r"\bexplain the paper\b",
        r"\bcomplete document\b",
        r"\bwhole document\b",
        r"\bfull document\b",
    ]

    return any(
        re.search(pattern, text)
        for pattern in read_patterns
    )


# --------------------------------------------------
# 5. DETECT COMPARISON REQUESTS
# --------------------------------------------------

def is_comparison_request(question):

    text = question.lower()

    comparison_patterns = [
        r"\bcompare\b",
        r"\bcomparison\b",
        r"\bdifferences between\b",
        r"\bdifference between\b",
        r"\bsimilarities between\b",
        r"\bsimilarity between\b",
    ]

    return any(
        re.search(pattern, text)
        for pattern in comparison_patterns
    )


# --------------------------------------------------
# 6. DETECT TOPIC SEARCH REQUESTS
# --------------------------------------------------

TOPIC_SEARCH_PHRASES = [
    "find from document",
    "find in document",
    "explain from document",
    "explain in document",
    "search document for",
    "search in document",
    "what does the document say about",
    "what does this paper say about",
]


def extract_topic_question(question):

    text_lower = question.lower()

    for phrase in TOPIC_SEARCH_PHRASES:

        position = text_lower.find(phrase)

        if position >= 0:

            topic = question[
                position + len(phrase):
            ].strip(" :,-")

            return topic

    return None


# --------------------------------------------------
# 7. DETECT EXACT SEARCH
# --------------------------------------------------

EXACT_SEARCH_PHRASES = [
    "find exact lines",
    "search exact lines",
    "find exact text",
    "show exact lines",
    "quote from document",
    "find exact sentence",
    "search exact phrase",
]


def extract_exact_search_question(question):

    text_lower = question.lower()

    for phrase in EXACT_SEARCH_PHRASES:

        position = text_lower.find(phrase)

        if position >= 0:

            search_question = question[
                position + len(phrase):
            ].strip(" :,-")

            return search_question

    return None


# --------------------------------------------------
# 8. DETECT DOCUMENT SELECTION ONLY
# --------------------------------------------------

def is_selection_only_request(question):

    text = question.lower().strip()

    patterns = [
        r"^document\s*:",
        r"^select\s+(the\s+)?(first|second|third|\d+)",
        r"^(first|second|third)\s+(paper|document)$",
        r"^(paper|document)\s+\d+$",
    ]

    return any(
        re.search(pattern, text)
        for pattern in patterns
    )


# --------------------------------------------------
# 9. CHATBOT COMMANDS
# --------------------------------------------------

print("\nDefault behaviour: Hybrid Mode")

print("\nAvailable commands:")

print("1. Read the first paper")
print("2. Read the second paper")
print("3. Compare both papers")
print("4. Find from document: methodology")
print("5. Find exact lines about a topic")
print("6. List documents")
print("7. Documents Only")
print("8. Hybrid Mode")
print("9. All Documents")
print("10. Exit")

print("\nAdditional examples:")

print("Read this document and summarize")
print("Find from document: results")
print("Find exact lines about throughput")
print("Explain the methodology")
print("What are the limitations?")

print("\nYou can change modes anytime.")


# --------------------------------------------------
# 10. MAIN CHAT LOOP
# --------------------------------------------------

while True:

    question = input("\nYou: ").strip()

    if not question:
        continue

    command = question.lower().strip()

    # ----------------------------------------------
    # EXIT
    # ----------------------------------------------

    if command in ["exit", "quit", "stop"]:

        print("\nChatbot stopped.")

        break

    # ----------------------------------------------
    # LIST DOCUMENTS
    # ----------------------------------------------

    if command in [
        "list documents",
        "show documents",
        "my documents",
        "uploaded documents",
        "show uploaded documents",
    ]:

        show_documents()

        continue

    # ----------------------------------------------
    # MODE SWITCHING
    # ----------------------------------------------

    if command in [
        "documents only",
        "document only",
        "pdf only",
        "only from documents",
        "only documents",
    ]:

        mode = "documents_only"

        print("\nSwitched to Documents Only mode.")

        continue

    if command in [
        "hybrid",
        "hybrid mode",
        "general mode",
        "normal mode",
    ]:

        mode = "hybrid"

        print("\nSwitched to Hybrid mode.")

        continue

    # ----------------------------------------------
    # SEARCH ALL DOCUMENTS
    # ----------------------------------------------

    if command in [
        "all documents",
        "search all documents",
        "clear document selection",
        "search everything",
    ]:

        selected_document_ids = []

        active_document_id = None

        print(
            "\nDocument selection cleared. "
            "Future searches will include all documents."
        )

        continue

    # ----------------------------------------------
    # LOAD DOCUMENT LIST
    # ----------------------------------------------

    try:

        documents = get_available_documents()

    except Exception as error:

        print("\nCould not load documents:")
        print(error)

        continue

    if not documents:

        print(
            "\nNo documents found. "
            "Process a document before asking document questions."
        )

        continue

    # ----------------------------------------------
    # DETECT DOCUMENT REFERENCES
    # ----------------------------------------------

    referenced_ids = find_referenced_documents(
        question,
        documents
    )

    if referenced_ids:

        for document_id in referenced_ids:

            if document_id not in selected_document_ids:

                selected_document_ids.append(document_id)

        active_document_id = referenced_ids[-1]

        for document in documents:

            if document["id"] == active_document_id:

                print(
                    f"\nSelected document: {document['filename']}"
                )

                break

    # ----------------------------------------------
    # HANDLE SELECTION-ONLY REQUEST
    # ----------------------------------------------

    if (
        referenced_ids
        and is_selection_only_request(question)
        and not is_read_request(question)
        and not is_comparison_request(question)
    ):

        print(
            "\nDocument selected successfully. "
            "You can now ask questions about this document."
        )

        continue

    # ----------------------------------------------
    # DETECT COMPARISON REQUEST
    # ----------------------------------------------

    comparison_request = is_comparison_request(question)

    document_ids_for_question = None

    if comparison_request:

        if len(selected_document_ids) >= 2:

            document_ids_for_question = (
                selected_document_ids[-2:]
            )

        elif len(documents) >= 2:

            document_ids_for_question = [
                documents[0]["id"],
                documents[1]["id"],
            ]

        else:

            print(
                "\nPlease select at least two documents first."
            )

            continue

    # ----------------------------------------------
    # SELECT DOCUMENT FOR NORMAL QUESTIONS
    # ----------------------------------------------

    elif active_document_id is not None:

        document_ids_for_question = [
            active_document_id
        ]

    # ----------------------------------------------
    # HANDLE AMBIGUOUS READ REQUESTS
    # ----------------------------------------------

    if (
        is_read_request(question)
        and active_document_id is None
        and len(documents) > 1
    ):

        print("\nWhich document should I read?")

        show_documents()

        print("\nExample: Read the first paper.")

        continue

    # ----------------------------------------------
    # ANSWER GENERATION
    # ----------------------------------------------

    try:

        # ------------------------------------------
        # COMPARE DOCUMENTS
        # ------------------------------------------

        if comparison_request:

            print("\nComparing selected documents...")

            answer = generate_document_comparison(
                document_ids_for_question,
                conversation_history,
            )

        # ------------------------------------------
        # READ / SUMMARIZE DOCUMENT
        # ------------------------------------------

        elif (
            is_read_request(question)
            and active_document_id is not None
        ):

            print("\nReading and summarizing the complete document...")

            answer = generate_document_summary(
                active_document_id,
                conversation_history,
            )

        # ------------------------------------------
        # EXACT DOCUMENT SEARCH
        # ------------------------------------------

        elif any(
            phrase in command
            for phrase in EXACT_SEARCH_PHRASES
        ):

            search_question = extract_exact_search_question(
                question
            )

            if not search_question:

                search_question = input(
                    "\nWhat should I find in the document? "
                ).strip()

            if not search_question:

                print("\nPlease enter a search topic.")

                continue

            print("\nSearching for exact document text...")

            answer = find_exact_document_lines(
                search_question,
                document_ids=document_ids_for_question,
            )

        # ------------------------------------------
        # TOPIC SEARCH AND EXPLANATION
        # ------------------------------------------

        elif any(
            phrase in command
            for phrase in TOPIC_SEARCH_PHRASES
        ):

            topic_question = extract_topic_question(
                question
            )

            if not topic_question:

                print("\nPlease provide a topic to search for.")

                continue

            print("\nFinding and explaining the requested topic...")

            answer = generate_document_topic_answer(
                topic_question,
                document_ids=document_ids_for_question,
                conversation_history=conversation_history,
            )

        # ------------------------------------------
        # NORMAL RAG QUESTION
        # ------------------------------------------

        else:

            print("\nSearching documents...")

            answer = generate_rag_answer(
                question,
                mode=mode,
                top_k=5,
                document_ids=document_ids_for_question,
                conversation_history=conversation_history,
            )

        # ------------------------------------------
        # DISPLAY ANSWER
        # ------------------------------------------

        print("\n========== ANSWER ==========\n")

        print(answer)

        # ------------------------------------------
        # SAVE CONVERSATION
        # ------------------------------------------

        conversation_history.append({
            "role": "user",
            "content": question,
        })

        conversation_history.append({
            "role": "assistant",
            "content": answer,
        })

    except Exception as error:

        print("\nERROR:")
        print(error)
