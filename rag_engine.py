
import re
import requests

from embedding_engine import generate_embeddings

from semantic_search import (
    hybrid_search,
    exact_document_search,
    list_documents,
    get_document_chunks,
)


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
CHAT_MODEL = "qwen3:1.7b"
CHAT_TIMEOUT = 180

# Larger batches reduce the number of Ollama requests.
SUMMARY_BATCH_SIZE = 8

RETRIEVAL_TOP_K = 7


# --------------------------------------------------
# 1. QWEN CHAT
# --------------------------------------------------

def ask_qwen(prompt):

    payload = {
        "model": CHAT_MODEL,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0.2,
        },
    }

    try:
        response = requests.post(
            OLLAMA_CHAT_URL,
            json=payload,
            timeout=CHAT_TIMEOUT,
        )

        response.raise_for_status()

    except requests.exceptions.ConnectionError as error:
        raise RuntimeError(
            "Cannot connect to Ollama. Make sure Ollama is running."
        ) from error

    except requests.exceptions.Timeout as error:
        raise RuntimeError(
            "Qwen3 response timed out. Try again."
        ) from error

    except requests.exceptions.HTTPError as error:
        raise RuntimeError(
            f"Ollama chat error: {error}"
        ) from error

    result = response.json()
    answer = result.get("message", {}).get("content", "")

    if not answer:
        raise RuntimeError("Qwen3 returned an empty answer.")

    return answer.strip()


# --------------------------------------------------
# 2. CONVERSATION HISTORY
# --------------------------------------------------

def format_conversation_history(history, max_messages=6):

    if not history:
        return "No previous conversation."

    formatted = []

    for message in history[-max_messages:]:

        role = message.get("role", "user")
        content = message.get("content", "")

        if content:
            formatted.append(
                f"{role.upper()}: {content[:1500]}"
            )

    return "\n".join(formatted)


# --------------------------------------------------
# 3. FOLLOW-UP QUESTION HANDLING
# --------------------------------------------------

def rewrite_followup_query(question, conversation_history=None):

    if not conversation_history:
        return question

    followup_patterns = [
        r"\bit\b",
        r"\bthey\b",
        r"\bthem\b",
        r"\bthis\b",
        r"\bthat\b",
        r"\bthese\b",
        r"\bthose\b",
        r"\btheir\b",
        r"\bwhat about\b",
        r"\bhow about\b",
        r"\bexplain more\b",
        r"\btell me more\b",
        r"\bwhy\b",
        r"\bhow does it\b",
        r"\bwhat is its\b",
        r"\bwhat are its\b",
    ]

    question_lower = question.lower().strip()

    is_short_question = len(question.split()) <= 7

    is_context_dependent = any(
        re.search(pattern, question_lower)
        for pattern in followup_patterns
    )

    if not is_short_question and not is_context_dependent:
        return question

    history = format_conversation_history(
        conversation_history,
        max_messages=6,
    )

    prompt = f"""
You are a search-query rewriting assistant.

Rewrite the latest user question into one standalone search query.

CONVERSATION:
{history}

LATEST QUESTION:
{question}

RULES:
1. Preserve the user's actual intent.
2. Use previous conversation only to resolve references.
3. Do not answer the question.
4. Do not invent facts or details.
5. Return only the rewritten search query.
"""

    rewritten = ask_qwen(prompt).strip()

    if not rewritten or len(rewritten) > 500:
        return question

    return rewritten


# --------------------------------------------------
# 4. RETRIEVE RELEVANT DOCUMENT CHUNKS
# --------------------------------------------------

def get_relevant_chunks(
    question,
    top_k=5,
    document_ids=None,
):

    query_embedding = generate_embeddings([question])[0]

    return hybrid_search(
        query=question,
        query_embedding=query_embedding,
        top_k=top_k,
        document_ids=document_ids,
    )


# --------------------------------------------------
# 5. BUILD DOCUMENT CONTEXT
# --------------------------------------------------

def build_document_context(results):

    if not results:
        return "No relevant document content was found."

    context_parts = []

    for index, result in enumerate(results, start=1):

        context_parts.append(
            f"""
SOURCE {index}
Document ID: {result['document_id']}
Filename: {result['filename']}
Chunk: {result['chunk_index']}

CONTENT:
{result['content']}
"""
        )

    return "\n".join(context_parts)


# --------------------------------------------------
# 6. HYBRID ANSWER GENERATION
# --------------------------------------------------

def generate_hybrid_answer(
    question,
    results,
    conversation_history=None,
):

    context = build_document_context(results)

    history = format_conversation_history(
        conversation_history
    )

    prompt = f"""
You are an accurate general-purpose AI assistant.

Answer the user's latest question.

DOCUMENT RULES:

1. Prioritize relevant information from supplied documents.
2. Never invent document content.
3. If information is missing, clearly say so.
4. You may use general knowledge when useful.
5. Clearly distinguish document information from general knowledge.
6. Mention source filenames when using document information.
7. Include chunk numbers where useful.
8. Do not claim a document says something unless the supplied content supports that statement.
9. Do not confuse information from different documents.
10. If uncertain, say so.

CONVERSATION HISTORY:
{history}

RETRIEVED DOCUMENT CONTENT:
{context}

CURRENT USER QUESTION:
{question}

Provide a clear, structured answer.
"""

    return ask_qwen(prompt)


# --------------------------------------------------
# 7. DOCUMENTS-ONLY ANSWER
# --------------------------------------------------

def generate_documents_only_answer(results, top_k=5):

    if not results:
        return "No matching document content was found."

    answer_parts = []

    for result in results[:top_k]:

        source = (
            f"Source: {result['filename']} "
            f"(Document ID: {result['document_id']}, "
            f"Chunk {result['chunk_index']})"
        )

        answer_parts.append(
            f"{source}\n{result['content']}"
        )

    return "\n\n--------------------\n\n".join(answer_parts)


# --------------------------------------------------
# 8. MAIN RAG FUNCTION
# --------------------------------------------------

def generate_rag_answer(
    question,
    mode="hybrid",
    top_k=5,
    document_ids=None,
    conversation_history=None,
):

    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    if mode not in ("hybrid", "documents_only"):
        raise ValueError("Invalid mode.")

    retrieval_question = rewrite_followup_query(
        question,
        conversation_history,
    )

    results = get_relevant_chunks(
        retrieval_question,
        top_k=top_k,
        document_ids=document_ids,
    )

    if mode == "documents_only":

        return generate_documents_only_answer(
            results,
            top_k=top_k,
        )

    return generate_hybrid_answer(
        question,
        results,
        conversation_history,
    )


# --------------------------------------------------
# 9. LIST DOCUMENTS
# --------------------------------------------------

def get_available_documents():

    return list_documents()


# --------------------------------------------------
# 10. DOCUMENT SUMMARY HELPERS
# --------------------------------------------------

def _format_chunk_batch(chunks):

    parts = []

    for chunk in chunks:

        parts.append(
            f"""
CHUNK {chunk['chunk_index']}:

{chunk['content']}
"""
        )

    return "\n".join(parts)


def _summarize_chunk_batch(
    filename,
    chunks,
    batch_number,
    total_batches,
):

    context = _format_chunk_batch(chunks)

    prompt = f"""
You are an accurate document analysis assistant.

DOCUMENT:
{filename}

BATCH:
{batch_number} of {total_batches}

DOCUMENT CONTENT:
{context}

TASK:

Summarize the information in this batch.

Preserve:
1. Important technical details.
2. Definitions and terminology.
3. Methodology and experimental procedures.
4. Numerical values, measurements and equations.
5. Results and observations.
6. Important limitations.
7. Conclusions and recommendations.

RULES:

- Use only the supplied content.
- Do not invent missing details.
- Preserve important numerical values accurately.
- Do not assume that a section is a conclusion unless the text supports that interpretation.
- Mention the chunk numbers covered by this summary.
- Do not omit important technical information merely to make the summary shorter.

Write a structured summary.
"""

    return ask_qwen(prompt)


def _reduce_summaries(filename, summaries):

    # Skip intermediate LLM reduction calls.
    # The final synthesis will combine the batch summaries.

    return summaries


# --------------------------------------------------
# 11. FULL DOCUMENT SUMMARY
# --------------------------------------------------

def generate_document_summary(
    document_id,
    conversation_history=None,
    max_chunks=None,
):

    """
    Summarize the complete document using batch processing.

    max_chunks is retained for compatibility with older callers.
    It is intentionally not used to truncate the document.
    """

    chunks = get_document_chunks(document_id)

    if not chunks:
        return "No content was found for this document."

    # Always obtain the filename from the database.
    filename = chunks[0]["filename"]

    total_chunks = len(chunks)

    batch_summaries = []

    batches = [
        chunks[start:start + SUMMARY_BATCH_SIZE]
        for start in range(0, total_chunks, SUMMARY_BATCH_SIZE)
    ]

    total_batches = len(batches)

    for index, batch in enumerate(batches, start=1):

        print(
            f"Summarizing {filename}: "
            f"batch {index}/{total_batches}"
        )

        summary = _summarize_chunk_batch(
            filename,
            batch,
            index,
            total_batches,
        )

        batch_summaries.append(summary)

    consolidated = _reduce_summaries(
        filename,
        batch_summaries,
    )

    history = format_conversation_history(
        conversation_history
    )

    final_context = "\n\n".join(consolidated)

    prompt = f"""
You are an accurate document-reading assistant.

You have received summaries covering all chunks of one document.

DOCUMENT CONTENT SOURCE:
{filename}

TOTAL DOCUMENT CHUNKS:
{total_chunks}

CONSOLIDATED CONTENT:
{final_context}

CONVERSATION HISTORY:
{history}

TASK:

Explain the complete document clearly.

Use these sections where applicable:

1. Main subject and central argument.
2. Purpose and objectives.
3. Important sections and concepts.
4. Methodology.
5. Experimental setup or implementation.
6. Results and numerical findings.
7. Discussion and interpretation.
8. Limitations.
9. Conclusion.
10. Important technical details.

RULES:

- Cover the entire document, not just its beginning.
- Do not invent information.
- If a section is absent or unclear, say so.
- Preserve important numerical values.
- Distinguish the authors' findings from your interpretation.
- Do not describe a result as experimentally validated unless the supplied content supports that claim.
- Use clear headings and explanations.
- Do not generate, guess, or substitute a document filename.
- Do not mention any filename in your answer.
- Use only the supplied document summaries as evidence.

Provide the complete document explanation.
"""

    answer = ask_qwen(prompt)

    # Add the authoritative database filename ourselves.
    # The LLM does not control this metadata.

    return (
        f"**Document:** {filename}\n\n"
        f"{answer}"
    )


# --------------------------------------------------
# 12. DOCUMENT SECTION EXTRACTION
# --------------------------------------------------

SECTION_ALIASES = {
    "methodology": [
        "methodology",
        "research methodology",
        "methods",
        "materials and methods",
        "method and materials",
        "research methods"
    ],
    "results": [
        "results",
        "experimental results",
        "findings",
        "results and discussion"
    ],
    "discussion": [
        "discussion",
        "results and discussion"
    ],
    "conclusion": [
        "conclusion",
        "conclusions",
        "concluding remarks",
        "summary and conclusion"
    ],
    "limitations": [
        "limitations",
        "study limitations",
        "research limitations"
    ],
    "experience": [
        "experience",
        "work experience",
        "professional experience",
        "employment history",
        "work history"
    ],
    "education": [
        "education",
        "educational qualifications",
        "academic qualifications",
        "academic background"
    ],
    "certifications": [
        "certifications",
        "certificates",
        "licenses and certifications"
    ],
    "projects": [
        "projects",
        "academic projects",
        "personal projects",
        "professional projects"
    ],
    "skills": [
        "skills",
        "technical skills",
        "core competencies",
        "key skills"
    ],
    "abstract": [
        "abstract"
    ],
    "introduction": [
        "introduction",
        "background"
    ],
    "references": [
        "references",
        "bibliography",
        "works cited"
    ]
}


def normalize_heading(text):
    text = text.lower().strip()

    text = re.sub(r"[*_#]", "", text)

    text = re.sub(
        r"^\s*(?:\d+(?:\.\d+)*\.?)\s*",
        "",
        text
    )

    text = re.sub(r"[:\-–—]+$", "", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip(" .:")


def get_heading_level(line):
    """
    Detect likely document headings.

    Numbered headings retain their hierarchy:
    1. Methodology       -> level 1
    1.1 Research Design  -> level 2
    1.1.1 Data           -> level 3
    """

    stripped = line.strip()

    if not stripped:
        return None

    numbered = re.match(
        r"^(\d+(?:\.\d+)*)(?:\.|\s+)\s*\S",
        stripped
    )

    if numbered:
        return numbered.group(1).count(".") + 1

    if stripped.startswith("#"):
        return len(stripped) - len(stripped.lstrip("#"))

    if stripped.startswith("**") and stripped.endswith("**"):
        return 1

    clean = re.sub(r"[*_]", "", stripped).strip()

    if len(clean) > 100:
        return None

    words = clean.split()

    if len(words) > 12:
        return None

    # Short uppercase headings.
    if clean.isupper() and len(words) <= 10:
        return 1

    return None


def merge_document_chunks(chunks):
    """
    Reconstruct document text in chunk order.

    Removes repeated overlap between adjacent chunks
    when the overlap can be identified.
    """

    if not chunks:
        return ""

    ordered_chunks = sorted(
        chunks,
        key=lambda item: item["chunk_index"]
    )

    document_text = ""

    for chunk in ordered_chunks:

        content = chunk["content"].strip()

        if not content:
            continue

        if not document_text:
            document_text = content
            continue

        # Find the largest exact overlap between
        # the previous text ending and current text beginning.
        max_overlap = min(
            1200,
            len(document_text),
            len(content)
        )

        overlap = 0

        for size in range(max_overlap, 30, -1):

            if document_text[-size:] == content[:size]:
                overlap = size
                break

        if overlap:
            document_text += content[overlap:]
        else:
            document_text += "\n\n" + content

    return document_text


def find_requested_heading(lines, requested_section):
    """
    Find an exact or recognized alias heading.
    Avoid matching ordinary sentences containing the topic.
    """

    requested = normalize_heading(requested_section)

    aliases = SECTION_ALIASES.get(
        requested,
        [requested]
    )

    normalized_aliases = {
        normalize_heading(alias)
        for alias in aliases
    }

    matches = []

    for index, line in enumerate(lines):

        clean_line = line.strip()

        if not clean_line:
            continue

        normalized = normalize_heading(clean_line)

        if normalized not in normalized_aliases:
            continue

        heading_level = get_heading_level(clean_line)

        # Accept exact standalone headings.
        # Plain standalone headings are also allowed.
        if (
            heading_level is not None
            or len(clean_line.split()) <= 8
        ):
            matches.append({
                "index": index,
                "level": heading_level or 1,
                "heading": clean_line
            })

    return matches


def extract_document_section(document_text, requested_section):
    """
    Extract only the requested section.

    The section ends at the next heading of the same
    or higher hierarchy level.
    """

    lines = document_text.splitlines()

    matches = find_requested_heading(
        lines,
        requested_section
    )

    if not matches:
        return None

    # Use the first matching section heading.
    selected = matches[0]

    start_index = selected["index"]
    start_level = selected["level"]

    end_index = len(lines)

    for index in range(start_index + 1, len(lines)):

        line = lines[index]

        level = get_heading_level(line)

        if level is not None and level <= start_level:
            end_index = index
            break

        # Recognize other known standalone section headings.
        normalized = normalize_heading(line)

        known_headings = {
            normalize_heading(alias)
            for aliases in SECTION_ALIASES.values()
            for alias in aliases
        }

        if (
            normalized in known_headings
            and len(line.strip().split()) <= 8
        ):
            end_index = index
            break

    section_lines = lines[
        start_index + 1:end_index
    ]

    section_content = "\n".join(section_lines).strip()

    if not section_content:
        return None

    return {
        "heading": selected["heading"],
        "content": section_content,
        "start_line": start_index + 1,
        "end_line": end_index
    }


def generate_document_topic_answer(
    question,
    document_ids=None,
    conversation_history=None,
    top_k=8
):
    """
    Extract and explain only the requested document section.
    """

    if not question or not question.strip():
        return "Please provide a section name."

    # Extract the requested topic.
    requested_section = question.strip()

    requested_section = re.sub(
        r"^(find from document|find in document|"
        r"from document|in document)\s*:\s*",
        "",
        requested_section,
        flags=re.IGNORECASE
    ).strip()

    if not requested_section:
        return "Please specify the section you want to read."

    # Load only the selected document(s).
    chunks = []

    if document_ids is not None:

        for document_id in document_ids:

            chunks.extend(
                get_document_chunks(document_id)
            )

    else:
        return (
            "Please select a document before requesting "
            "a specific section."
        )

    if not chunks:
        return "No document content was found."

    # Separate sections for each document.
    documents = {}

    for chunk in chunks:

        document_id = chunk["document_id"]

        if document_id not in documents:
            documents[document_id] = {
                "filename": chunk["filename"],
                "chunks": []
            }

        documents[document_id]["chunks"].append(chunk)

    extracted_sections = []

    for document_id, document in documents.items():

        full_text = merge_document_chunks(
            document["chunks"]
        )

        section = extract_document_section(
            full_text,
            requested_section
        )

        if section:

            extracted_sections.append({
                "document_id": document_id,
                "filename": document["filename"],
                "heading": section["heading"],
                "content": section["content"],
                "start_line": section["start_line"],
                "end_line": section["end_line"]
            })

    if not extracted_sections:

        return (
            f"The section '{requested_section}' "
            "could not be identified in the selected document. "
            "No full-document summary was generated."
        )

    history = format_conversation_history(
        conversation_history
    )

    answers = []

    for section in extracted_sections:

        prompt = f"""
You are a document section analysis assistant.

USER REQUESTED SECTION:
{requested_section}

DOCUMENT:
{section['filename']}

ACTUAL SECTION HEADING:
{section['heading']}

EXTRACTED SECTION CONTENT:
{section['content']}

TASK:

Explain only the extracted section.

RULES:

1. Use exclusively the extracted section above.
2. Do not use information from other document sections.
3. Do not create a complete document summary.
4. Do not add unrelated background information.
5. Preserve the original technical details.
6. Preserve numerical values, equations, names and terminology.
7. Explain the section clearly in natural language.
8. Do not invent missing information.
9. If the section contains subsections, explain those subsections.
10. Do not claim that something is present if it is absent.
11. If the section is incomplete, clearly mention that.
12. Do not use conversation history as evidence.

CONVERSATION HISTORY:
{history}

Write an explanation limited strictly to this section.
"""

        answer = ask_qwen(prompt)

        answers.append(
            f"**Document:** {section['filename']}\n"
            f"**Section:** {section['heading']}\n\n"
            f"{answer}"
        )

    return "\n\n--------------------\n\n".join(answers)

# --------------------------------------------------
# 13. COMPARE DOCUMENTS
# --------------------------------------------------

def generate_document_comparison(
    document_ids,
    conversation_history=None,
    max_chunks_per_document=None,
):

    if not document_ids or len(document_ids) < 2:
        return "Please select at least two documents to compare."

    document_summaries = []

    for document_id in document_ids:

        summary = generate_document_summary(
            document_id,
            conversation_history=None,
        )

        document_summaries.append(
            f"""
DOCUMENT ID: {document_id}

DOCUMENT SUMMARY:
{summary}
"""
        )

    history = format_conversation_history(
        conversation_history
    )

    documents_context = "\n\n".join(document_summaries)

    prompt = f"""
You are an accurate document comparison assistant.

CONVERSATION HISTORY:
{history}

DOCUMENTS:
{documents_context}

COMPARE:

1. Main objective of each document.
2. Methodology.
3. Tools and technologies.
4. Experimental setup or implementation.
5. Results and findings.
6. Conclusions.
7. Similarities.
8. Differences.
9. Limitations supported by the documents.
10. Whether the documents complement or contradict each other.

RULES:

- Identify each document using the document information provided.
- Do not invent missing information.
- Clearly state when information is unavailable.
- Do not assume all documents are research papers.
- Preserve important numerical details.
- Use a comparison table where useful.

Provide the comparison.
"""

    return ask_qwen(prompt)


# --------------------------------------------------
# 14. EXACT DOCUMENT SEARCH
# --------------------------------------------------

def find_exact_document_lines(
    question,
    document_ids=None,
):

    return exact_document_search(
        question,
        document_ids=document_ids,
    )
