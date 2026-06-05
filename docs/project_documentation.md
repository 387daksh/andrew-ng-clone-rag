# Project Architecture & RAG Pipeline Documentation

This documentation provides an in-depth walkthrough of the system flow, chunking strategy, hybrid retrieval mechanics, and database structures designed for the Digital Twin of Andrew Ng.

---

## 1. Document Collection & Processing

The RAG pipeline operates on a dataset containing Andrew Ng's personal lecture notes and chronological archives from deeplearning.ai:
*   **Lectures & Transcripts**: Stored inside `.pdf` files.
*   **The Batch Articles**: Saved inside a structured `.jsonl` document where each line holds an individual article containing a `title`, `body`, and original `url`.

---

## 2. Ingestion & Chunking Strategy

To prepare the documents for retrieval, the pipeline applies a clean text-processing loop:
1.  **Text Extraction**: 
    *   PDF text is extracted using PyMuPDF (`fitz`), which provides fast, layout-aware reading.
    *   JSONL files are read line-by-line using Python's standard `json` library, merging the title and article body into a single block.
2.  **Recursive Paragraph Splitting**:
    *   We use LangChain's `RecursiveCharacterTextSplitter` with a `chunk_size` of `1000` characters and a `chunk_overlap` of `200` characters.
    *   The splitter targets specific separator sequences: double line breaks (`\n\n`), single breaks (`\n`), periods (`. `), and spaces (` `). This ensures that sentences and paragraphs remain cohesive, maintaining full contextual integrity.
3.  **Local Embedding**:
    *   Each chunk is converted into a 384-dimensional vector embedding using the `"all-MiniLM-L6-v2"` model from the `sentence-transformers` library.
    *   The resulting arrays and document metadata are stored in a persistent local ChromaDB index located at `database/chroma`.

---

## 3. Hybrid Retrieval & Fusion Model (RAG)

To maximize search accuracy, the system queries the database using two distinct search methodologies in parallel, merging results via **Reciprocal Rank Fusion (RRF)**:

```
                  ┌──────────────────────┐
                  │   User Query Input   │
                  └──────────┬───────────┘
                             │
              ┌──────────────┴──────────────┐
              ▼                             ▼
   ┌────────────────────┐        ┌────────────────────┐
   │ Chroma Vector      │        │ rank-bm25 Keyword  │
   │ Similarity Search  │        │ Score Matching     │
   └──────────┬─────────┘        └──────────┬─────────┘
              │                             │
              ▼                             ▼
         [Top 50 Hits]                 [Top 50 Hits]
              │                             │
              └──────────────┬──────────────┘
                             ▼
                  ┌──────────────────────┐
                  │ Reciprocal Rank      │
                  │ Fusion (RRF) Scorer  │
                  └──────────┬───────────┘
                             ▼
                 [Final Top-K Context Block]
```

### Vector Similarity Search
Queries the local ChromaDB index to find semantic matches, capturing conceptual meaning even if the user uses different words than the source texts.

### Keyword Search (`rank-bm25`)
Pulls all documents from the database, tokenizes them, and calculates exact keyword matches using the standard `BM25Okapi` algorithm. This ensures that technical terms, acronyms, and specific numbers are never missed.

### Reciprocal Rank Fusion (RRF)
Vector scores and BM25 scores operate on entirely different mathematical scales. To merge them reliably, we calculate a unified rank score for each chunk based on its positional rank in both search lists:
$$\text{Score}(d) = \sum_{m \in M} \frac{1}{60 + r_m(d)}$$
The top $k$ chunks with the highest RRF scores are compiled into the final prompt context, providing an exceptionally grounded reference list.

---

## 4. Short-Term & Long-Term Memory Architecture

To make the agent feel interactive and personalized, it features two distinct memory layers:
*   **Short-Term Session Memory**: Handled by Streamlit's `st.session_state` to track active messages within the current browser tab.
*   **Long-Term Memory**: Stored in a highly readable, flat-file JSON document at `database/memories.json`. 

### Dual-Engine Memory Extraction
When a user types a message, the memory system parses the input using two cooperative engines:
1.  **LLM-Based Parser (Primary)**: Queries `gemini-3.1-flash-lite` to extract key facts, preferences, experience levels, or interests, returning them in a standard string format: `KEY: VALUE (IMPORTANCE)`.
2.  **String-Slicing Parser (Fallback)**: If the API key is not configured or the network is offline, a standard Python string manipulation pipeline (`.find()`, `.split()`) extracts basic user facts and learning profiles instantly with zero dependencies.

---

## 5. User Interface & Demo Orchestration

The front-end is served by a highly optimized, linear Streamlit script ([app.py](file:///e:/andrewng%20rag/digital_twin_andrew_ng/app.py)):
1.  **Sidebar Configuration**: Allows adjusting the model name, retrieval parameters (`top-k`), temperature, and learning profiles. Includes the Voice Clone settings to switch between cloned voice synthesis and standard text-to-speech output.
2.  **Chat Canvas**: Renders message bubbles dynamically from session state.
3.  **Integrated Voice Portal**: Enables recording questions directly or uploading audio files (`.wav`/`.flac`), transcribing them via standard speech-recognition, and sending them directly through the RAG cycle.
4.  **Grounding Panels**: Expandable widgets display the retrieved chunks, matching source metadata, and long-term memory logs so the entire execution is completely transparent and easy to demonstrate.

---

## 6. Voice Cloning & Synthesis

To provide vocal feedback, the app integrates dual-mode text-to-speech synthesis:
*   **Standard TTS**: Uses `edge-tts` to generate high-quality default speech streams matching standard assistant voices.
*   **Voice Cloning**: Employs a local `chatterbox-tts` model (`ChatterboxTurboTTS`) executing on CPU/CUDA to synthesize custom voice replicas.
    *   **Reference File**: The model reads a target 10-20 second WAV audio recording located strictly at `database/voices/my_voice.wav`.
    *   **Execution**: When a user selects "My cloned voice", the app passes the assistant response text and the reference recording to the chatterbox engine to synthesize and output a matched wav file. If the reference WAV file is missing, the app defaults back to standard voice generation.
