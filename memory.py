import json
from pathlib import Path
from datetime import datetime
import google.generativeai as genai

DEFAULT_JSON_PATH = Path(__file__).resolve().parent / "database" / "memories.json"

def init_db(db_path=DEFAULT_JSON_PATH):
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    if not Path(db_path).exists():
        with open(db_path, "w", encoding="utf-8") as f:
            json.dump([], f)

def load_memories(db_path=DEFAULT_JSON_PATH):
    init_db(db_path)
    try:
        with open(db_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def save_memories(memories, db_path=DEFAULT_JSON_PATH):
    with open(db_path, "w", encoding="utf-8") as f:
        json.dump(memories, f, indent=4, ensure_ascii=False)

def save_memory(key, value, importance=0.5, db_path=DEFAULT_JSON_PATH):
    if not key or not value:
        return None
    memories = load_memories(db_path)
    memory_id = len(memories) + 1
    new_memory = {
        "id": memory_id,
        "key": key,
        "value": value,
        "importance": float(importance),
        "created_at": datetime.utcnow().isoformat()
    }
    memories.append(new_memory)
    save_memories(memories, db_path)
    return memory_id

def retrieve_memory(key, limit=5, db_path=DEFAULT_JSON_PATH):
    memories = load_memories(db_path)
    matched = [m for m in memories if m.get("key") == key]
    matched.sort(key=lambda m: m.get("id", 0), reverse=True)
    return matched[:limit]

def get_latest_memory_value(key, db_path=DEFAULT_JSON_PATH):
    rows = retrieve_memory(key, limit=1, db_path=db_path)
    return rows[0]["value"] if rows else ""

def search_memory(query, limit=5, db_path=DEFAULT_JSON_PATH):
    if not query:
        return []
    memories = load_memories(db_path)
    query_lower = query.lower()
    matched = []
    for m in memories:
        key_match = query_lower in str(m.get("key", "")).lower()
        val_match = query_lower in str(m.get("value", "")).lower()
        if key_match or val_match:
            matched.append(m)
    matched.sort(key=lambda m: (m.get("importance", 0.0), m.get("id", 0)), reverse=True)
    return matched[:limit]

def list_memories(limit=200, db_path=DEFAULT_JSON_PATH):
    memories = load_memories(db_path)
    memories.sort(key=lambda m: m.get("id", 0), reverse=True)
    return memories[:limit]

def delete_memory(memory_id, db_path=DEFAULT_JSON_PATH):
    memories = load_memories(db_path)
    initial_len = len(memories)
    memories = [m for m in memories if m.get("id") != memory_id]
    if len(memories) < initial_len:
        save_memories(memories, db_path)
        return True
    return False

def extract_long_term_memories_fallback(text):
    memories = []
    lower = text.lower()
    
    if "beginner" in lower or "new to" in lower or "just starting" in lower:
        memories.append(("skill_level", "Beginner", 0.9))
    elif "intermediate" in lower:
        memories.append(("skill_level", "Intermediate", 0.9))
    elif "advanced" in lower or "expert" in lower:
        memories.append(("skill_level", "Advanced", 0.9))
        
    if "my name is " in lower:
        idx = lower.find("my name is ") + len("my name is ")
        name = text[idx:].strip().split(".")[0].split("!")[0].split("?")[0].strip()
        if name:
            memories.append(("user_name", name, 0.8))
            
    if "remember that " in lower:
        idx = lower.find("remember that ") + len("remember that ")
        fact = text[idx:].strip().split(".")[0].split("!")[0].split("?")[0].strip()
        if fact:
            memories.append(("user_fact", fact, 0.7))
            
    for pref_key in ["i prefer ", "i like ", "i want ", "please "]:
        if pref_key in lower:
            idx = lower.find(pref_key) + len(pref_key)
            preference = text[idx:].strip().split(".")[0].split("!")[0].split("?")[0].strip()
            if preference:
                memories.append(("preference", f"{pref_key.strip()} {preference}", 0.6))
            break
            
    topics = [
        "machine learning",
        "deep learning",
        "neural networks",
        "ai",
        "data science",
        "linear regression",
        "logistic regression",
        "backpropagation",
        "cnn",
        "rnn"
    ]
    for topic in topics:
        if topic in lower:
            memories.append((f"topic:{topic}", "interested", 0.4))
            
    return memories

def extract_long_term_memories(text, api_key):
    if not text:
        return []
        
    if not api_key:
        return extract_long_term_memories_fallback(text)
        
    prompt = f"""
Analyze the following user message and extract any long-term memories or key facts about the user that are worth remembering (such as their name, learning preferences, skill level, topics of interest, or specific facts they share).

Format each extracted memory on a new line as:
KEY: VALUE (IMPORTANCE)

Guidelines:
- KEY should be a simple snake_case string (e.g., user_name, skill_level, preference, topic_interest, user_fact).
- VALUE should be the exact fact or detail.
- IMPORTANCE should be a float between 0.0 and 1.0.
- If there are no long-term memories worth saving, respond with ONLY the word "NONE".

User message:
"{text}"

Response:
""".strip()

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-3.1-flash-lite")
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        if not response_text or "NONE" in response_text:
            return []
            
        memories = []
        for line in response_text.splitlines():
            line = line.strip()
            if not line or ":" not in line or "(" not in line or ")" not in line:
                continue
                
            parts = line.split(":", 1)
            key = parts[0].strip()
            
            value_part = parts[1].rsplit("(", 1)
            value = value_part[0].strip()
            
            importance_str = value_part[1].replace(")", "").strip()
            try:
                importance = float(importance_str)
            except ValueError:
                importance = 0.5
                
            memories.append((key, value, importance))
        return memories
    except Exception:
        return extract_long_term_memories_fallback(text)