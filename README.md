# Chatbot-LLM
# Axon.ai — AI-Powered Personal Assistant

Axon.ai is a Python-based AI assistant designed to understand documents, answer questions, perform research, assist with coding, and solve problems using Large Language Models (LLMs).

It combines Retrieval-Augmented Generation (RAG), semantic search, document processing, and a modular skill-based architecture to provide an extensible AI assistant.

## Features

### 1. AI Chat Assistant
- Conversational AI powered by Qwen3.
- Context-aware follow-up questions.
- Conversation history management.
- Edit messages and regenerate responses.

### 2. Document Intelligence
- Upload and process multiple document formats.
- Extract and analyze document content.
- Ask questions about uploaded documents.
- Generate document summaries.
- Compare multiple documents.
- Search for specific information and sections.

**Supported formats include:**
PDF, DOCX, DOC, TXT, MD, CSV, XLSX, XLS, PPTX, PPT, JSON, HTML, EPUB, RTF, ODT, ODS and ODP.

### 3. Retrieval-Augmented Generation (RAG)
- Semantic search using document embeddings.
- Keyword-based search.
- Hybrid document retrieval.
- Context-based answer generation.
- Local knowledge storage using SQLite.

### 4. Modular AI Skills

Axon.ai includes a modular skill architecture:

| Skill | Function |
|---|---|
| Document Skill | Document reading, analysis and comparison |
| Research Skill | Structured research and information analysis |
| Coding Skill | Code generation, explanation and debugging |
| Reasoning Skill | Logical reasoning and problem-solving |
| Memory Skill | Persistent assistant memory |
| Planning Skill | Task and project planning |
| Tools Skill | Tool registration and execution |
| Skill Creator | Create new skill definitions and drafts |

### 5. Technology Stack

- **Python** — Core programming language
- **Qwen3 1.7B** — Conversational language model
- **Nomic Embed Text** — Text embedding model
- **Ollama** — Local model execution
- **Streamlit** — Web application interface
- **SQLite** — Chat history and knowledge storage
- **RAG** — Retrieval-Augmented Generation

## Project Structure

```text
Chatbot LLM/
│
├── main.py
├── document_reader.py
├── text_chunker.py
├── embedding_engine.py
├── knowledge_store.py
├── semantic_search.py
├── rag_engine.py
├── requirements.txt
│
├── skills/
│   ├── __init__.py
│   ├── skill_manager.py
│   ├── document_skill.py
│   ├── research_skill.py
│   ├── coding_skill.py
│   ├── reasoning_skill.py
│   ├── memory_skill.py
│   ├── planning_skill.py
│   ├── tools_skill.py
│   └── skill_creator.py
│
└── database/
    └── Local databases are generated automatically.
```

## Installation and Setup

### Prerequisites

- Python 3.11 or newer (Python 3.13 is used in development).
- Ollama.
- Git.

### 1. Clone the repository

```bash
git clone https://github.com/ParenTrivedi/Chatbot_LLM.git
cd Chatbot_LLM
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install and download Ollama models

Install Ollama from:

https://ollama.com/download

Download the required models:

```bash
ollama pull qwen3:1.7b
ollama pull nomic-embed-text
```

### 5. Run Axon.ai

Start Ollama, then run:

```bash
streamlit run main.py
```

Open the local application in your browser at:

```text
http://localhost:8501
```

## How It Works

1. The user interacts with Axon.ai through the Streamlit interface.
2. The application processes the request and identifies the appropriate task.
3. For document-related questions, the RAG engine retrieves relevant document chunks.
4. The embedding engine generates vector representations for semantic retrieval.
5. Retrieved context is passed to the language model.
6. The model generates a response based on the available context.

## Current Development Status

Axon.ai is an actively developed project.

Planned improvements include:
- Improved retrieval accuracy and response quality.
- Advanced conversation memory.
- More AI skills and tool integrations.
- Optimized document processing.
- Cloud deployment and public web access.
- Scalable model inference.

## Contributing

Contributions, suggestions and improvements are welcome.

## License

License information will be added in a future release.

---

**Axon.ai — One assistant. Multiple capabilities.**
