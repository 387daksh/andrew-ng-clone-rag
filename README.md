# Digital Twin of Andrew Ng

This repository contains a lightweight, direct, and completely reproducible AI-agent project that emulates the teaching style, timeline, and core knowledge of Andrew Ng. It combines Gemini 3.1 Flash Lite with hybrid search retrieval-augmented generation (RAG) and a plain-text JSON long-term memory system.

---

## 1. Project Overview

*   **RAG Pipeline**: Integrates PyMuPDF (`fitz`) and JSONL document loading. It chunks texts via `RecursiveCharacterTextSplitter` and embeds them using sentence-transformers.
*   **Hybrid Search**: Executes keyword query matching (`rank-bm25`) alongside dense vector lookup (ChromaDB), combining results cleanly via **Reciprocal Rank Fusion (RRF)**.
*   **SQLite-Free Memory**: Automatically tracks user facts, experience levels, and learning preferences directly inside a transparent, plain-text `database/memories.json` document.
*   **Encouraging Persona**: Enforces Andrew Ng's structured style (Intuition $\rightarrow$ Example $\rightarrow$ Technical Details $\rightarrow$ Key Takeaways) using simple system rules.
*   **Polished Front-End**: Interactive chatbot running entirely on standard, out-of-the-box Streamlit widgets.

---

## 2. Installation & Setup

### Install Dependencies
Clone this repository and install all required python libraries:
```bash
pip install -r requirements.txt
```

### Configure the Environment
Set your Gemini API Key in your terminal shell:

*   **Windows PowerShell**:
    ```powershell
    $env:GEMINI_API_KEY="your-gemini-api-key-here"
    ```
*   **macOS / Linux Shell**:
    ```bash
    export GEMINI_API_KEY="your-gemini-api-key-here"
    ```

---

## 3. Running Instructions

### Step 1: Ingest Data
Place all educational PDFs and JSONL articles inside the `data/` directory. Then compile the local RAG index by running:
```bash
python ingest.py --reset
```

### Step 2: Start the Web App
Launch the interactive Streamlit chatbot portal:
```bash
streamlit run app.py
```

---

## 4. Codebase Navigation

The repository contains 6 core self-documenting files:
*   [app.py](file:///e:/andrewng%20rag/digital_twin_andrew_ng/app.py): The main execution pipeline. Hosts the Streamlit chatbot pages, voice processing helpers, and routing.
*   [ingest.py](file:///e:/andrewng%20rag/digital_twin_andrew_ng/ingest.py): Document scanning, text extraction, recursive splitting, and ChromaDB indexing.
*   [rag.py](file:///e:/andrewng%20rag/digital_twin_andrew_ng/rag.py): Vector & keyword hybrid search using Reciprocal Rank Fusion (RRF).
*   [memory.py](file:///e:/andrewng%20rag/digital_twin_andrew_ng/memory.py): Transparent JSON-based memory loading, saving, search, and dual-engine fact extraction.
*   [prompts.py](file:///e:/andrewng%20rag/digital_twin_andrew_ng/prompts.py): Dialogue state formatters and LLM prompt compilers.
*   [persona.py](file:///e:/andrewng%20rag/digital_twin_andrew_ng/persona.py): Prompt guidelines capturing the patient Andrew Ng persona.
*   [timeline.py](file:///e:/andrewng%20rag/digital_twin_andrew_ng/timeline.py): Timeline loader.

---

## 5. Documentation & Core Features

*   **Architecture Flowchart**: See [docs/architecture.md](file:///e:/andrewng%20rag/digital_twin_andrew_ng/docs/architecture.md) for the complete data flow.
*   **Sample Scenarios**: Read [docs/sample_conversations.md](file:///e:/andrewng%20rag/digital_twin_andrew_ng/docs/sample_conversations.md) for 10 realistic conversations showcasing the twin's persona, precision citation accuracy, and persistent memory recall.
