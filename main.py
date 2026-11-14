
import subprocess
import sys
import os
import sqlite3
import re
import tempfile
import json

from pathlib import Path
from datetime import datetime


# ==========================================================
# AUTOMATIC STREAMLIT LAUNCH
# ==========================================================

if __name__ == "__main__" and not os.environ.get("STREAMLIT_RUNNING"):
    env = os.environ.copy()
    env["STREAMLIT_RUNNING"] = "1"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            os.path.abspath(__file__)
        ],
        env=env
    )

else:

    import streamlit as st
    import requests

    from document_reader import extract_text
    from text_chunker import chunk_text
    from embedding_engine import generate_embeddings

    from knowledge_store import (
        store_document,
        delete_document,
        get_storage_statistics
    )

    from rag_engine import (
        generate_rag_answer,
        find_exact_document_lines,
        get_available_documents,
        generate_document_summary,
        generate_document_topic_answer,
        generate_document_comparison
    )

    from skills.skill_manager import SkillManager
    from skills.document_skill import DocumentSkill
    from skills.research_skill import ResearchSkill
    from skills.coding_skill import CodingSkill
    from skills.reasoning_skill import ReasoningSkill
    from skills.memory_skill import MemorySkill
    from skills.tools_skill import ToolsSkill
    from skills.planning_skill import PlanningSkill
    from skills.skill_creator import SkillCreator

    # ======================================================
    # CONFIGURATION
    # ======================================================

    MODEL = "qwen3:1.7b"
    OLLAMA_URL = "http://localhost:11434/api/chat"

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATABASE_DIR = os.path.join(BASE_DIR, "database")

    DATABASE_PATH = os.path.join(
        DATABASE_DIR,
        "chat_history.db"
    )

    KNOWLEDGE_DATABASE_PATH = os.path.join(
        DATABASE_DIR,
        "knowledge.db"
    )

    os.makedirs(DATABASE_DIR, exist_ok=True)

    st.set_page_config(
        page_title="Axon.ai",
        page_icon="🤖",
        layout="wide"
    )

    # ======================================================
    # CHAT DATABASE
    # Existing database schema preserved
    # ======================================================

    def get_connection():
        connection = sqlite3.connect(DATABASE_PATH)
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def initialize_database():

        with get_connection() as connection:

            connection.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)

            connection.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (conversation_id)
                        REFERENCES conversations(id)
                )
            """)

    def create_conversation():

        now = datetime.now().isoformat(timespec="seconds")

        with get_connection() as connection:

            cursor = connection.execute(
                """
                INSERT INTO conversations
                (title, created_at, updated_at)
                VALUES (?, ?, ?)
                """,
                ("New Chat", now, now)
            )

            return cursor.lastrowid

    def get_conversations():

        with get_connection() as connection:

            rows = connection.execute(
                """
                SELECT id, title
                FROM conversations
                ORDER BY updated_at DESC, id DESC
                """
            ).fetchall()

        return rows

    def save_message(conversation_id, role, content):

        now = datetime.now().isoformat(timespec="seconds")

        with get_connection() as connection:

            connection.execute(
                """
                INSERT INTO messages
                (conversation_id, role, content, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (conversation_id, role, str(content), now)
            )

            connection.execute(
                """
                UPDATE conversations
                SET updated_at = ?
                WHERE id = ?
                """,
                (now, conversation_id)
            )

            if role == "user":

                conversation = connection.execute(
                    """
                    SELECT title
                    FROM conversations
                    WHERE id = ?
                    """,
                    (conversation_id,)
                ).fetchone()

                if conversation and conversation[0] == "New Chat":

                    title = content.strip().replace("\n", " ")[:40]

                    connection.execute(
                        """
                        UPDATE conversations
                        SET title = ?
                        WHERE id = ?
                        """,
                        (title or "New Chat", conversation_id)
                    )

    def load_messages(conversation_id):

        with get_connection() as connection:

            rows = connection.execute(
                """
                SELECT id, role, content
                FROM messages
                WHERE conversation_id = ?
                ORDER BY id ASC
                """,
                (conversation_id,)
            ).fetchall()

        return [
            {
                "id": row[0],
                "role": row[1],
                "content": row[2]
            }
            for row in rows
        ]

    def load_messages_before(message_id, conversation_id):

        with get_connection() as connection:

            rows = connection.execute(
                """
                SELECT role, content
                FROM messages
                WHERE conversation_id = ?
                  AND id < ?
                ORDER BY id ASC
                """,
                (conversation_id, message_id)
            ).fetchall()

        return [
            {
                "role": role,
                "content": content
            }
            for role, content in rows
        ]

    def delete_conversation(conversation_id):

        with get_connection() as connection:

            connection.execute(
                """
                DELETE FROM messages
                WHERE conversation_id = ?
                """,
                (conversation_id,)
            )

            connection.execute(
                """
                DELETE FROM conversations
                WHERE id = ?
                """,
                (conversation_id,)
            )

    def delete_messages_from(message_id, conversation_id):

        now = datetime.now().isoformat(timespec="seconds")

        with get_connection() as connection:

            connection.execute(
                """
                DELETE FROM messages
                WHERE conversation_id = ?
                  AND id >= ?
                """,
                (conversation_id, message_id)
            )

            first_user = connection.execute(
                """
                SELECT content
                FROM messages
                WHERE conversation_id = ?
                  AND role = 'user'
                ORDER BY id ASC
                LIMIT 1
                """,
                (conversation_id,)
            ).fetchone()

            title = (
                first_user[0].strip().replace("\n", " ")[:40]
                if first_user else "New Chat"
            )

            connection.execute(
                """
                UPDATE conversations
                SET title = ?, updated_at = ?
                WHERE id = ?
                """,
                (title or "New Chat", now, conversation_id)
            )

    # Initialize database without changing existing records.
    initialize_database()

    # ======================================================
    # SKILL SYSTEM
    # ======================================================

    skill_manager = SkillManager()

    skill_manager.register_skill("document", DocumentSkill())
    skill_manager.register_skill("research", ResearchSkill())
    skill_manager.register_skill("coding", CodingSkill())
    skill_manager.register_skill("reasoning", ReasoningSkill())
    skill_manager.register_skill("memory", MemorySkill())
    skill_manager.register_skill("tools", ToolsSkill())
    skill_manager.register_skill("planning", PlanningSkill())
    skill_manager.register_skill("skill_creator", SkillCreator())

    # ======================================================
    # SESSION STATE
    # ======================================================

    if "active_conversation_id" not in st.session_state:

        conversations = get_conversations()

        if conversations:
            st.session_state.active_conversation_id = conversations[0][0]
        else:
            st.session_state.active_conversation_id = create_conversation()

    if "selected_document_ids" not in st.session_state:
        st.session_state.selected_document_ids = []

    if "rag_mode" not in st.session_state:
        st.session_state.rag_mode = "hybrid"

    if "editing_message_id" not in st.session_state:
        st.session_state.editing_message_id = None

    if "confirm_delete_chat" not in st.session_state:
        st.session_state.confirm_delete_chat = False

    # ======================================================
    # DOCUMENT PROCESSING
    # ======================================================

    def process_uploaded_file(uploaded_file):

        suffix = Path(uploaded_file.name).suffix.lower()
        temporary_path = None

        try:

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix
            ) as temporary_file:

                temporary_file.write(uploaded_file.getvalue())
                temporary_path = temporary_file.name

            extracted_text = extract_text(temporary_path)

            if not extracted_text or not extracted_text.strip():
                raise ValueError("No readable text found.")

            chunks = chunk_text(
                extracted_text,
                chunk_size=1200,
                overlap=200
            )

            if not chunks:
                raise ValueError("No document chunks were created.")

            embeddings = generate_embeddings(chunks)

            if len(chunks) != len(embeddings):
                raise ValueError(
                    "Chunk count and embedding count do not match."
                )

            document_id = store_document(
                temporary_path,
                chunks,
                embeddings
            )

            # Store the original uploaded filename.
            with sqlite3.connect(KNOWLEDGE_DATABASE_PATH) as connection:

                connection.execute(
                    """
                    UPDATE documents
                    SET filename = ?
                    WHERE id = ?
                    """,
                    (uploaded_file.name, document_id)
                )

            return {
                "document_id": document_id,
                "filename": uploaded_file.name,
                "chunks": len(chunks)
            }

        finally:

            if temporary_path and os.path.exists(temporary_path):
                os.remove(temporary_path)

    # ======================================================
    # AXON IDENTITY
    # ======================================================

    def is_axon_identity_question(question):

        normalized = question.lower().strip()

        normalized = re.sub(
            r"[?!.,]+$",
            "",
            normalized
        ).strip()

        patterns = [
            r"^(?:hi[\s,]*)?who are you$",
            r"^(?:hello[\s,]*)?who are you$",
            r"^(?:hi[\s,]*)?what is your name$",
            r"^(?:hello[\s,]*)?what is your name$",
            r"^(?:hi[\s,]*)?what's your name$",
            r"^(?:hello[\s,]*)?what's your name$",
            r"^introduce yourself$",
            r"^tell me about yourself$"
        ]

        return any(
            re.search(pattern, normalized)
            for pattern in patterns
        )

    def get_ai_response(conversation_messages):

        clean_messages = [
            {
                "role": message["role"],
                "content": message["content"]
            }
            for message in conversation_messages
            if message.get("role") in ("user", "assistant")
        ]

        payload = {
            "model": MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are Axon, the AI assistant behind Axon.ai. "
                        "Your name is Axon. You are not the user. "
                        "Never describe the user's resume or background "
                        "as your own identity. "
                        "When asked who you are, introduce yourself as Axon. "
                        "Answer accurately and clearly. "
                        "Do not invent facts. If uncertain, say so."
                    )
                },
                *clean_messages
            ],
            "stream": False,
            "think": False
        }

        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=180
        )

        response.raise_for_status()

        result = response.json()

        if "message" not in result:
            raise ValueError("Ollama returned an unexpected response.")

        return result["message"]["content"].strip()

    # ======================================================
    # SKILL RESULT FORMATTER
    # ======================================================

    def format_skill_result(result):

        if isinstance(result, str):
            return result

        return json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
            default=str
        )

    # ======================================================
    # SKILL REQUEST ROUTER
    # ======================================================

    def handle_skill_request(question):

        text = question.strip()
        lower = text.lower()

        # List skills
        if re.search(
            r"\b(list|show|what are the|available)\s+"
            r"(all\s+)?skills\b",
            lower
        ):

            return (
                "Available skills:\n\n"
                + "\n".join(
                    f"- {name.replace('_', ' ').title()}"
                    for name in skill_manager.list_skills()
                )
            )

        # Create a custom skill
        creator_match = re.search(
            r"\b(?:skill\s+)?(?:called|named)\s+"
            r"([a-zA-Z0-9_ -]+?)"
            r"(?=\s+(?:for|that|to)\s+|$)",
            text,
            re.IGNORECASE
        )

        is_skill_creation = bool(
            re.search(
                r"\b(create|make|build|generate)\s+"
                r"(?:a\s+|an\s+|new\s+)?"
                r"(?:python\s+)?skill\b",
                lower
            )
        )

        if is_skill_creation:

            if not creator_match:
                return (
                    "Please specify a skill name and its purpose.\n\n"
                    "Example: Create a skill named translator "
                    "for translating English into Gujarati."
                )

            skill_name = creator_match.group(1).strip()
            description = text[creator_match.end():].strip()

            description = re.sub(
                r"^(?:for|that|to)\s+",
                "",
                description,
                flags=re.IGNORECASE
            ).strip()

            if not description:
                return "Please describe what the new skill should do."

            creator = skill_manager.get_skill("skill_creator")

            try:

                if re.search(r"\bpython\s+skill\b", lower):

                    result = creator.handle(
                        action="create_python",
                        name=skill_name,
                        description=description
                    )

                else:

                    result = creator.handle(
                        action="create_instruction",
                        name=skill_name,
                        description=description
                    )

                return (
                    "Skill creation result:\n\n"
                    + format_skill_result(result)
                    + "\n\nPython skills are saved as drafts. "
                    "They are not automatically executed."
                )

            except Exception as error:
                return f"Skill creation failed: {error}"

        # List custom skills
        if re.search(
            r"\b(list|show)\s+(my\s+)?custom\s+skills\b",
            lower
        ):

            result = skill_manager.execute_skill(
                "skill_creator",
                action="list"
            )

            return format_skill_result(result)

        # Memory
        if re.match(
            r"^(remember|save to memory|store in memory)\b",
            lower
        ):

            content = re.sub(
                r"^(remember|save to memory|store in memory)"
                r"[\s,:-]*(that\s+)?",
                "",
                text,
                flags=re.IGNORECASE
            ).strip()

            if not content:
                return "Please specify what you want me to remember."

            result = skill_manager.execute_skill(
                "memory",
                action="add",
                content=content
            )

            return f"Saved to memory: {result['content']}"

        if re.match(
            r"^(what do you remember|search memory|find in memory)\b",
            lower
        ):

            keyword = re.sub(
                r"^(what do you remember about|what do you remember|"
                r"search memory for|search memory|find in memory)"
                r"[\s,:-]*",
                "",
                text,
                flags=re.IGNORECASE
            ).strip()

            if keyword:

                result = skill_manager.execute_skill(
                    "memory",
                    action="search",
                    keyword=keyword
                )

            else:

                result = skill_manager.execute_skill(
                    "memory",
                    action="list"
                )

            return format_skill_result(result)

        if re.match(r"^(list|show) memories\b", lower):

            result = skill_manager.execute_skill(
                "memory",
                action="list"
            )

            return format_skill_result(result)

        # Tools
        if re.match(
            r"^(list|show)\s+(available\s+)?tools\b",
            lower
        ):

            result = skill_manager.execute_skill(
                "tools",
                action="list"
            )

            return format_skill_result(result)

        # Research
        if re.match(r"^(research|investigate|study)\b", lower):

            topic = re.sub(
                r"^(research|investigate|study)"
                r"[\s,:-]*(the\s+topic\s+)?",
                "",
                text,
                flags=re.IGNORECASE
            ).strip()

            if not topic:
                return "Please provide a research topic."

            result = skill_manager.execute_skill(
                "research",
                action="research",
                topic=topic
            )

            return format_skill_result(result)

        # Coding
        coding_patterns = [
            r"\bdebug\b",
            r"\bfix\s+(this\s+)?(python\s+)?code\b",
            r"\bexplain\s+(this\s+)?code\b",
            r"\breview\s+(this\s+)?code\b",
            r"\bwrite\s+(some\s+)?code\b",
            r"\bgenerate\s+(some\s+)?code\b",
            r"\bprogramming\s+error\b"
        ]

        if (
            any(re.search(pattern, lower) for pattern in coding_patterns)
            or "```" in text
        ):

            result = skill_manager.execute_skill(
                "coding",
                action="coding_task",
                task=text
            )

            return format_skill_result(result)

        # Planning
        if re.match(
            r"^(plan|create a plan|make a plan|"
            r"prepare a plan|break down this project)\b",
            lower
        ):

            goal = re.sub(
                r"^(plan|create a plan|make a plan|"
                r"prepare a plan|break down this project)"
                r"[\s,:-]*",
                "",
                text,
                flags=re.IGNORECASE
            ).strip()

            if not goal:
                return "Please describe what you want to plan."

            result = skill_manager.execute_skill(
                "planning",
                action="plan",
                goal=goal
            )

            return format_skill_result(result)

        # Reasoning
        if re.match(
            r"^(solve|calculate|work out|reason through|"
            r"solve this problem)\b",
            lower
        ):

            result = skill_manager.execute_skill(
                "reasoning",
                action="solve",
                problem=text
            )

            return format_skill_result(result)

        return None

    # ======================================================
    # DOCUMENT REFERENCE DETECTION
    # ======================================================

    def find_referenced_documents(question, documents):

        question_lower = question.lower()
        referenced = []

        ordinal_words = {
            "first": 0,
            "1st": 0,
            "second": 1,
            "2nd": 1,
            "third": 2,
            "3rd": 2,
            "fourth": 3,
            "4th": 3,
            "fifth": 4,
            "5th": 4
        }

        for word, index in ordinal_words.items():

            pattern = (
                rf"\b{word}\s+"
                rf"(?:paper|document|file|article)\b"
            )

            if re.search(pattern, question_lower):

                if index < len(documents):
                    referenced.append(documents[index]["id"])

        number_matches = re.findall(
            r"\b(?:paper|document|file|article)\s*(\d+)\b",
            question_lower
        )

        for number in number_matches:

            index = int(number) - 1

            if 0 <= index < len(documents):
                referenced.append(documents[index]["id"])

        for document in documents:

            filename = document["filename"].lower()

            if filename in question_lower:
                referenced.append(document["id"])

        return list(dict.fromkeys(referenced))

    # ======================================================
    # DOCUMENT COMMAND DETECTION
    # ======================================================

    def is_read_request(question):

        patterns = [
            r"\bread\b",
            r"\bsummarize\b",
            r"\bsummary\b",
            r"\boverview\b",
            r"\bexplain\s+(?:the\s+)?document\b",
            r"\bexplain\s+(?:the\s+)?paper\b",
            r"\bfull\s+document\b",
            r"\bwhole\s+document\b"
        ]

        return any(
            re.search(pattern, question, re.IGNORECASE)
            for pattern in patterns
        )

    def is_comparison_request(question):

        patterns = [
            r"\bcompare\b",
            r"\bcomparison\b",
            r"\bdifferences\b",
            r"\bsimilarities\b"
        ]

        return any(
            re.search(pattern, question, re.IGNORECASE)
            for pattern in patterns
        )

    def is_topic_search_request(question):

        phrases = [
            "find from document",
            "find in document",
            "explain from document",
            "explain in document",
            "search document for",
            "search in document",
            "what does the document say about",
            "what does this paper say about"
        ]

        question_lower = question.lower()

        return any(
            phrase in question_lower
            for phrase in phrases
        )

    def extract_topic_question(question):

        phrases = [
            "what does the document say about",
            "what does this paper say about",
            "search document for",
            "explain from document",
            "explain in document",
            "search in document",
            "find from document",
            "find in document"
        ]

        question_lower = question.lower()

        for phrase in phrases:

            position = question_lower.find(phrase)

            if position != -1:

                return question[
                    position + len(phrase):
                ].strip(" :?.,\n")

        return question.strip()

    def is_exact_search_request(question):

        phrases = [
            "find exact lines",
            "search exact lines",
            "find exact text",
            "show exact lines",
            "quote from document",
            "find exact sentence",
            "search exact phrase"
        ]

        question_lower = question.lower()

        return any(
            phrase in question_lower
            for phrase in phrases
        )

    def extract_exact_search_question(question):

        phrases = [
            "find exact lines",
            "search exact lines",
            "find exact text",
            "show exact lines",
            "quote from document",
            "find exact sentence",
            "search exact phrase"
        ]

        question_lower = question.lower()

        for phrase in phrases:

            position = question_lower.find(phrase)

            if position != -1:

                return question[
                    position + len(phrase):
                ].strip(" :?.,\n")

        return question

    def is_list_documents_request(question):

        patterns = [
            r"\blist\s+documents\b",
            r"\bshow\s+documents\b",
            r"\bavailable\s+documents\b",
            r"\blist\s+files\b"
        ]

        return any(
            re.search(pattern, question, re.IGNORECASE)
            for pattern in patterns
        )

    def is_selection_only_request(question):

        patterns = [
            r"^\s*(?:select\s+)?(?:first|second|third|fourth|fifth)\s+"
            r"(?:paper|document|file)\s*$",
            r"^\s*(?:select\s+)?(?:paper|document|file)\s*\d+\s*$"
        ]

        return any(
            re.search(pattern, question, re.IGNORECASE)
            for pattern in patterns
        )

    # ======================================================
    # DOCUMENT LIST
    # ======================================================

    def format_document_list(documents):

        if not documents:
            return "No documents have been uploaded yet."

        output = ["Available documents:"]

        for index, document in enumerate(documents, start=1):

            output.append(
                f"{index}. {document['filename']} "
                f"(ID: {document['id']}, "
                f"Chunks: {document['chunk_count']})"
            )

        return "\n".join(output)

    # ======================================================
    # RESPONSE ROUTER
    # ======================================================

    def generate_response(
        question,
        conversation_history,
        documents,
        selected_document_ids,
        mode
    ):

        # Identity
        if is_axon_identity_question(question):

            return (
                "Hi! I'm Axon, the AI assistant behind Axon.ai. "
                "I can help with coding, research, document analysis, "
                "problem-solving, and everyday tasks."
            )

        # Document list
        if is_list_documents_request(question):
            return format_document_list(documents)

        # Skill routing
        skill_response = handle_skill_request(question)

        if skill_response is not None:
            return skill_response

        # Normal chat without documents
        if not documents:

            if (
                is_read_request(question)
                or is_comparison_request(question)
                or is_topic_search_request(question)
                or is_exact_search_request(question)
            ):

                return (
                    "No documents are available yet. "
                    "Upload a document before using document search."
                )

            conversation_messages = [
                {
                    "role": item["role"],
                    "content": item["content"]
                }
                for item in conversation_history
            ]

            conversation_messages.append({
                "role": "user",
                "content": question
            })

            return get_ai_response(conversation_messages)

        referenced_ids = find_referenced_documents(
            question,
            documents
        )

        target_ids = (
            referenced_ids
            if referenced_ids
            else selected_document_ids or None
        )

        # Document selection
        if is_selection_only_request(question):

            if referenced_ids:

                names = [
                    document["filename"]
                    for document in documents
                    if document["id"] in referenced_ids
                ]

                return (
                    "Document identified: "
                    + ", ".join(names)
                    + "\nYou can select it in the sidebar."
                )

            return (
                "I could not identify the requested document. "
                "Please check the available document list."
            )

        # Document comparison
        if is_comparison_request(question):

            if referenced_ids:
                comparison_ids = referenced_ids

            elif selected_document_ids:
                comparison_ids = selected_document_ids[-2:]

            elif len(documents) >= 2:
                comparison_ids = [
                    documents[0]["id"],
                    documents[1]["id"]
                ]

            else:
                return (
                    "At least two documents are required "
                    "for comparison."
                )

            if len(comparison_ids) < 2:
                return "Please select at least two documents to compare."

            return generate_document_comparison(
                comparison_ids,
                conversation_history=conversation_history
            )

        # Read document
        if is_read_request(question):

            if referenced_ids:
                document_id = referenced_ids[0]

            elif len(selected_document_ids) == 1:
                document_id = selected_document_ids[0]

            elif len(documents) == 1:
                document_id = documents[0]["id"]

            else:
                return (
                    "Which document would you like me to read?\n\n"
                    + format_document_list(documents)
                    + "\n\nSelect one document from the sidebar."
                )

            return generate_document_summary(
                document_id,
                conversation_history=conversation_history
            )

        # Exact search
        if is_exact_search_request(question):

            exact_query = extract_exact_search_question(question)

            if not exact_query:
                return "Please provide the exact text or phrase to search for."

            return find_exact_document_lines(
                exact_query,
                document_ids=target_ids
            )

        # Topic search
        if is_topic_search_request(question):

            topic = extract_topic_question(question)

            if not topic:
                return "Please specify the topic you want to find."

            return generate_document_topic_answer(
                topic,
                document_ids=target_ids,
                conversation_history=conversation_history,
                top_k=8
            )

        # Normal RAG
        return generate_rag_answer(
            question,
            mode=mode,
            top_k=5,
            document_ids=target_ids,
            conversation_history=conversation_history
        )

    # ======================================================
    # EDIT AND REGENERATE
    # ======================================================

    def regenerate_edited_message(
        message_id,
        conversation_id,
        edited_text
    ):

        edited_text = edited_text.strip()

        if not edited_text:
            raise ValueError("Message cannot be empty.")

        # Make sure we are editing the active conversation.
        if st.session_state.active_conversation_id != conversation_id:
            raise ValueError(
                "Conversation mismatch. Please reopen the correct chat."
            )

        # Check that the selected message is actually a user message.
        with get_connection() as connection:

            original_message = connection.execute(
                """
                SELECT id, role
                FROM messages
                WHERE id = ?
                  AND conversation_id = ?
                """,
                (message_id, conversation_id)
            ).fetchone()

        if not original_message or original_message[1] != "user":
            raise ValueError("The selected user message was not found.")

        # Load context before the edited message.
        previous_history = load_messages_before(
            message_id,
            conversation_id
        )

        # Load current documents and settings.
        documents = get_available_documents()

        selected_ids = st.session_state.get(
            "selected_document_ids",
            []
        )

        mode = st.session_state.get("rag_mode", "hybrid")

        # Generate a completely new answer.
        answer = generate_response(
            question=edited_text,
            conversation_history=previous_history,
            documents=documents,
            selected_document_ids=selected_ids,
            mode=mode
        )

        answer = format_skill_result(answer)

        if not answer.strip():
            raise ValueError("Axon generated an empty response.")

        now = datetime.now().isoformat(timespec="seconds")

        # Update the conversation only after generation succeeds.
        with get_connection() as connection:

            cursor = connection.execute(
                """
                UPDATE messages
                SET content = ?
                WHERE id = ?
                  AND conversation_id = ?
                  AND role = 'user'
                """,
                (edited_text, message_id, conversation_id)
            )

            if cursor.rowcount != 1:
                raise ValueError("Could not update the original message.")

            # Remove old messages after the edited message.
            connection.execute(
                """
                DELETE FROM messages
                WHERE conversation_id = ?
                  AND id > ?
                """,
                (conversation_id, message_id)
            )

            # Save the newly generated answer.
            connection.execute(
                """
                INSERT INTO messages
                (conversation_id, role, content, created_at)
                VALUES (?, 'assistant', ?, ?)
                """,
                (conversation_id, answer, now)
            )

            # Refresh the conversation title.
            first_user = connection.execute(
                """
                SELECT content
                FROM messages
                WHERE conversation_id = ?
                  AND role = 'user'
                ORDER BY id ASC
                LIMIT 1
                """,
                (conversation_id,)
            ).fetchone()

            title = (
                first_user[0].strip().replace("\n", " ")[:40]
                if first_user else "New Chat"
            )

            connection.execute(
                """
                UPDATE conversations
                SET title = ?, updated_at = ?
                WHERE id = ?
                """,
                (title or "New Chat", now, conversation_id)
            )

        # IMPORTANT:
        # Keep the same conversation selected after rerun.
        st.session_state.active_conversation_id = conversation_id
        st.session_state.pending_conversation_selection = conversation_id

        return answer

    # ======================================================
    # DELETE ACTIVE CONVERSATION
    # ======================================================

    def delete_active_conversation():

        conversation_id = st.session_state.get(
            "active_conversation_id"
        )

        if conversation_id is None:
            return

        delete_conversation(conversation_id)

        remaining_conversations = get_conversations()

        if remaining_conversations:
            new_id = remaining_conversations[0][0]
        else:
            new_id = create_conversation()

        # Do not modify the selectbox key here.
        st.session_state.active_conversation_id = new_id
        st.session_state.pending_conversation_selection = new_id

        st.session_state.editing_message_id = None
        st.session_state.confirm_delete_chat = False

        st.rerun()

    # ======================================================
    # DELETE DOCUMENT
    # ======================================================

    def remove_document_from_app(document_id):

        deleted = delete_document(document_id)

        if deleted:

            current_selection = st.session_state.get(
                "document_selector",
                []
            )

            new_selection = [
                item
                for item in current_selection
                if item != document_id
            ]

            # Store pending selection under a different key.
            st.session_state.pending_document_selection = new_selection
            st.session_state.selected_document_ids = new_selection

        return deleted

    # ======================================================
    # SIDEBAR
    # ======================================================

    with st.sidebar:

        st.title("🤖 Axon.ai")
        st.caption("Your personal AI assistant")

        # --------------------------------------------------
        # NEW CHAT
        # --------------------------------------------------

        if st.button("＋ New Chat", use_container_width=True):

            new_id = create_conversation()

            st.session_state.active_conversation_id = new_id

            # Never directly change an instantiated widget key.
            st.session_state.pending_conversation_selection = new_id

            st.session_state.editing_message_id = None
            st.session_state.confirm_delete_chat = False

            st.rerun()

        st.divider()

        # --------------------------------------------------
        # CHAT HISTORY
        # --------------------------------------------------

        st.subheader("Chat History")

        conversations = get_conversations()

        conversation_ids = [
            item[0] for item in conversations
        ]

        conversation_titles = {
            item[0]: item[1]
            for item in conversations
        }

        if conversation_ids:

            # Apply pending selection BEFORE creating selectbox.
            if "pending_conversation_selection" in st.session_state:

                pending_id = st.session_state.pop(
                    "pending_conversation_selection"
                )

                if pending_id in conversation_ids:
                    st.session_state["conversation_selector"] = pending_id

            current_id = st.session_state.get(
                "active_conversation_id"
            )

            if current_id not in conversation_ids:
                current_id = conversation_ids[0]

            if (
                st.session_state.get("conversation_selector")
                not in conversation_ids
            ):
                st.session_state["conversation_selector"] = current_id

            selected_conversation = st.selectbox(
                "Previous conversations",
                options=conversation_ids,
                format_func=lambda item: conversation_titles[item],
                key="conversation_selector"
            )

            # The selected widget value becomes the active conversation.
            st.session_state.active_conversation_id = selected_conversation

            if st.button(
                "🗑️ Delete Selected Chat",
                use_container_width=True
            ):
                st.session_state.confirm_delete_chat = True

            if st.session_state.confirm_delete_chat:

                st.warning(
                    "Delete this conversation and all its messages?"
                )

                confirm_col, cancel_col = st.columns(2)

                with confirm_col:

                    if st.button(
                        "Confirm Delete",
                        type="primary",
                        use_container_width=True
                    ):
                        delete_active_conversation()

                with cancel_col:

                    if st.button(
                        "Cancel",
                        use_container_width=True
                    ):
                        st.session_state.confirm_delete_chat = False
                        st.rerun()

        else:
            st.caption("No conversations yet.")

        st.divider()

        # --------------------------------------------------
        # DOCUMENT UPLOAD
        # --------------------------------------------------

        st.subheader("📁 Documents")

        uploaded_files = st.file_uploader(
            "Upload documents",
            type=[
                "pdf", "docx", "doc",
                "xlsx", "xls", "pptx", "ppt",
                "txt", "csv", "json", "md",
                "rtf", "odt", "ods", "odp",
                "html", "htm", "epub"
            ],
            accept_multiple_files=True,
            key="document_uploader"
        )

        if st.button(
            "Process Documents",
            use_container_width=True,
            disabled=not uploaded_files
        ):

            for uploaded_file in uploaded_files:

                try:

                    with st.spinner(
                        f"Processing {uploaded_file.name}..."
                    ):

                        result = process_uploaded_file(uploaded_file)

                    st.success(
                        f"Processed: {result['filename']} "
                        f"({result['chunks']} chunks)"
                    )

                except Exception as error:

                    st.error(
                        f"Could not process {uploaded_file.name}: {error}"
                    )

            st.rerun()

        st.divider()

        # --------------------------------------------------
        # DOCUMENT SELECTION
        # --------------------------------------------------

        st.subheader("Select Documents")

        documents = get_available_documents()

        document_options = {
            document["id"]: (
                f"{document['filename']} "
                f"({document['chunk_count']} chunks)"
            )
            for document in documents
        }

        available_ids = list(document_options.keys())

        # Apply pending selection BEFORE creating multiselect.
        if "pending_document_selection" in st.session_state:

            pending_documents = st.session_state.pop(
                "pending_document_selection"
            )

            st.session_state["document_selector"] = [
                item
                for item in pending_documents
                if item in available_ids
            ]

        # Remove IDs that no longer exist.
        current_document_selection = st.session_state.get(
            "document_selector",
            st.session_state.selected_document_ids
        )

        valid_document_selection = [
            item
            for item in current_document_selection
            if item in available_ids
        ]

        # This assignment happens before the widget is instantiated.
        st.session_state["document_selector"] = valid_document_selection

        selected_document_ids = st.multiselect(
            "Documents for questions",
            options=available_ids,
            format_func=lambda item: document_options[item],
            key="document_selector"
        )

        st.session_state.selected_document_ids = selected_document_ids

        if selected_document_ids:

            st.caption(
                f"{len(selected_document_ids)} document(s) selected."
            )

        else:

            st.caption(
                "No specific selection: searches can use all documents."
            )

        # --------------------------------------------------
        # DOCUMENT MANAGEMENT
        # --------------------------------------------------

        if documents:

            with st.expander("Manage Documents"):

                for document in documents:

                    st.write(f"**{document['filename']}**")

                    st.caption(
                        f"ID: {document['id']} | "
                        f"Chunks: {document['chunk_count']}"
                    )

                    confirm_key = (
                        f"confirm_delete_document_{document['id']}"
                    )

                    if confirm_key not in st.session_state:
                        st.session_state[confirm_key] = False

                    if not st.session_state[confirm_key]:

                        if st.button(
                            "🗑️ Delete Document",
                            key=f"delete_document_{document['id']}",
                            use_container_width=True
                        ):

                            st.session_state[confirm_key] = True
                            st.rerun()

                    else:

                        st.warning(
                            f"Delete {document['filename']} "
                            "from the knowledge database?"
                        )

                        confirm_col, cancel_col = st.columns(2)

                        with confirm_col:

                            if st.button(
                                "Confirm",
                                key=f"confirm_doc_{document['id']}",
                                type="primary"
                            ):

                                deleted = remove_document_from_app(
                                    document["id"]
                                )

                                st.session_state[confirm_key] = False

                                if deleted:
                                    st.success("Document deleted.")
                                else:
                                    st.error("Document was not found.")

                                st.rerun()

                        with cancel_col:

                            if st.button(
                                "Cancel",
                                key=f"cancel_doc_{document['id']}"
                            ):

                                st.session_state[confirm_key] = False
                                st.rerun()

        else:
            st.caption("No documents uploaded yet.")

        st.divider()

        # --------------------------------------------------
        # SEARCH MODE
        # --------------------------------------------------

        st.subheader("Search Mode")

        selected_mode = st.radio(
            "Choose response mode",
            options=["hybrid", "documents_only"],
            format_func=lambda value: (
                "Hybrid Mode"
                if value == "hybrid"
                else "Documents Only"
            ),
            index=(
                0
                if st.session_state.rag_mode == "hybrid"
                else 1
            ),
            key="mode_selector"
        )

        st.session_state.rag_mode = selected_mode

        st.divider()

        # --------------------------------------------------
        # SKILLS
        # --------------------------------------------------

        st.subheader("🧠 AI Skills")

        for skill_name in skill_manager.list_skills():

            st.write(
                f"✓ {skill_name.replace('_', ' ').title()}"
            )

        st.caption(
            f"{len(skill_manager.list_skills())} registered skills"
        )

        with st.expander("Custom Skill Files"):

            try:

                custom_skills = skill_manager.execute_skill(
                    "skill_creator",
                    action="list"
                )

                st.write("Instruction skills:")
                st.write(custom_skills["instruction_skills"])

                st.write("Python drafts:")
                st.write(custom_skills["python_drafts"])

            except Exception as error:
                st.error(str(error))

        st.divider()

        # --------------------------------------------------
        # STORAGE INFORMATION
        # --------------------------------------------------

        st.subheader("Storage Information")

        try:

            stats = get_storage_statistics()

            st.write(f"Documents: {stats['documents']}")
            st.write(f"Document chunks: {stats['chunks']}")

        except Exception as error:

            st.error(
                f"Could not read storage statistics: {error}"
            )

        st.divider()

        st.caption("Chat Model: Qwen3 1.7B")
        st.caption("Embedding Model: Nomic Embed Text")
        st.caption("Runtime: Ollama")
        st.caption("Storage: SQLite")
        st.caption("Application: Axon.ai")

    # ======================================================
    # LOAD ACTIVE CONVERSATION
    # ======================================================

    conversation_id = st.session_state.active_conversation_id

    messages = load_messages(conversation_id)

    # ======================================================
    # MAIN CHAT AREA
    # ======================================================

    st.title("Axon.ai")

    st.caption(
        "Chat normally, use AI skills, or upload documents "
        "to ask questions about them."
    )

    for message in messages:

        with st.chat_message(message["role"]):

            is_editing = (
                message["role"] == "user"
                and st.session_state.editing_message_id == message["id"]
            )

            if is_editing:

                with st.form(
                    key=f"edit_form_{message['id']}"
                ):

                    edited_text = st.text_area(
                        "Edit your message",
                        value=message["content"],
                        key=f"edit_text_{message['id']}",
                        height=120
                    )

                    save_edit = st.form_submit_button(
                        "Save and Regenerate",
                        type="primary"
                    )

                    cancel_edit = st.form_submit_button("Cancel")

                    if cancel_edit:

                        st.session_state.editing_message_id = None
                        st.rerun()

                    elif save_edit:

                        try:

                            with st.spinner(
                                "Axon is generating a new response..."
                            ):

                                regenerate_edited_message(
                                    message_id=message["id"],
                                    conversation_id=conversation_id,
                                    edited_text=edited_text
                                )

                            st.session_state.editing_message_id = None

                            st.rerun()

                        except requests.exceptions.ConnectionError:

                            st.error(
                                "Cannot connect to Ollama. "
                                "Please make sure Ollama is running."
                            )

                        except requests.exceptions.Timeout:

                            st.error(
                                "The AI request timed out. Please try again."
                            )

                        except requests.exceptions.RequestException as error:

                            st.error(f"AI connection error: {error}")

                        except Exception as error:

                            st.error(
                                f"Regeneration failed: "
                                f"{type(error).__name__}: {error}"
                            )

            else:

                st.markdown(message["content"])

                if message["role"] == "user":

                    edit_col, delete_col = st.columns([1, 1])

                    with edit_col:

                        if st.button(
                            "✏️ Edit",
                            key=f"edit_message_{message['id']}"
                        ):

                            st.session_state.editing_message_id = message["id"]
                            st.rerun()

                    with delete_col:

                        if st.button(
                            "🗑️ Delete from here",
                            key=f"delete_message_{message['id']}"
                        ):

                            delete_messages_from(
                                message["id"],
                                conversation_id
                            )

                            st.session_state.editing_message_id = None
                            st.rerun()

    # ======================================================
    # USER INPUT
    # ======================================================

    if prompt := st.chat_input("Message Axon..."):

        # Capture history before saving the new question.
        conversation_history = [
            {
                "role": item["role"],
                "content": item["content"]
            }
            for item in load_messages(conversation_id)
        ]

        # Save user message.
        save_message(
            conversation_id,
            "user",
            prompt
        )

        with st.chat_message("assistant"):

            with st.spinner("Axon is thinking..."):

                try:

                    current_documents = get_available_documents()

                    answer = generate_response(
                        question=prompt,
                        conversation_history=conversation_history,
                        documents=current_documents,
                        selected_document_ids=(
                            st.session_state.selected_document_ids
                        ),
                        mode=st.session_state.rag_mode
                    )

                    answer = format_skill_result(answer)

                except requests.exceptions.ConnectionError:

                    answer = (
                        "Cannot connect to Ollama. "
                        "Please make sure Ollama is running."
                    )

                except requests.exceptions.Timeout:

                    answer = (
                        "The AI request timed out. Please try again."
                    )

                except requests.exceptions.RequestException as error:

                    answer = f"AI connection error: {error}"

                except Exception as error:

                    answer = (
                        "An error occurred while processing your request:\n\n"
                        f"{type(error).__name__}: {error}"
                    )

                st.markdown(answer)

        # Save AI response to the same conversation.
        save_message(
            conversation_id,
            "assistant",
            answer
        )

        # Keep the same conversation selected on rerun.
        st.session_state.active_conversation_id = conversation_id
        st.session_state.pending_conversation_selection = conversation_id

        st.rerun()
