import json
from pathlib import Path

import fitz
import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter


DATA_DIR = Path("data")
DB_PATH = "database/chroma"
COLLECTION_NAME = "andrew_ng"


def read_pdf(path):
    doc = fitz.open(path)
    text = []
    for page in doc:
        text.append(page.get_text())
    return "\n".join(text)

def load_documents():

    documents = []
    if not DATA_DIR.exists():
        print("Data folder not found")
        return documents
    for path in DATA_DIR.rglob("*"):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            try:
                text = read_pdf(path)
                if text.strip():
                    documents.append(
                        {
                            "text": text,
                            "metadata": {
                                "source": str(path.relative_to(DATA_DIR)),
                                "title": path.stem,
                                "url": "",
                                "doc_type": "pdf",
                            },
                        }
                    )

            except Exception as e:
                print(f"Error reading {path}: {e}")

        elif suffix == ".jsonl":
            try:
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        item = json.loads(line)
                        title = item.get("title", "")
                        body = item.get("body", "")
                        url = item.get("url", "")
                        text = f"{title}\n\n{body}".strip()

                        if text:
                            documents.append(
                                {
                                    "text": text,
                                    "metadata": {
                                        "source": url
                                        or str(path.relative_to(DATA_DIR)),
                                        "title": title or path.stem,
                                        "url": url,
                                        "doc_type": "jsonl",
                                    },
                                }
                            )

            except Exception as e:
                print(f"Error reading {path}: {e}")

    return documents


def chunk_documents(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    metadatas = []

    for doc in documents:

        split_texts = splitter.split_text(doc["text"])
        for i, chunk in enumerate(split_texts):
            metadata = dict(doc["metadata"])
            metadata["chunk_index"] = i
            chunks.append(chunk)
            metadatas.append(metadata)
    return chunks, metadatas


def ingest(reset=False):
    documents = load_documents()
    if not documents:
        print("No documents found")
        return 0
    chunks, metadatas = chunk_documents(documents)
    client = chromadb.PersistentClient(path=DB_PATH)
    embedding_fn = (
        embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
    )

    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
            print("Old collection deleted")
        except:
            pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
    )

    ids = [f"chunk_{i}" for i in range(len(chunks))]

    batch_size = 100

    for i in range(0, len(chunks), batch_size):

        collection.add(
            documents=chunks[i : i + batch_size],
            metadatas=metadatas[i : i + batch_size],
            ids=ids[i : i + batch_size],
        )

        print(
            f"Added {min(i + batch_size, len(chunks))}/{len(chunks)} chunks"
        )

    return len(chunks)


if __name__ == "__main__":

    total_chunks = ingest(reset=True)

    print(f"\nSuccessfully ingested {total_chunks} chunks")