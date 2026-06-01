from persona import ANDREW_NG_PERSONA

def format_chat_history(messages, max_turns=6):
    if not messages:
        return "No prior conversation."
    trimmed = messages[-max_turns * 2 :]
    lines = []
    for msg in trimmed:
        role = msg.get("role", "user")
        label = "User" if role == "user" else "Assistant"
        content = msg.get("content", "")
        lines.append(f"{label}: {content}")
    return "\n".join(lines)

def format_memories_for_prompt(memories):
    if not memories:
        return "No long-term memories found."
    lines = []
    for mem in memories:
        key = mem.get("key", "")
        value = mem.get("value", "")
        importance = mem.get("importance", 0.0)
        lines.append(f"- {key}: {value} (importance {importance})")
    return "\n".join(lines)

def build_prompt(user_query, chat_history, retrieved_context, memories, timeline, skill_level):
    return f"""
{ANDREW_NG_PERSONA}

Teaching style rules:
- Patient and encouraging.
- Start with intuition, then a simple example, then technical details.
- End with key takeaways.
- Use clear, practical language.
- Do not be arrogant or overly academic.

Skill level to target: {skill_level}

Conversation history:
{chat_history}

Long-term memories about the user:
{memories}

Andrew Ng timeline:
{timeline}

Sources to ground your response (cite as [1], [2], etc.):
{retrieved_context}

User question:
{user_query}

Response requirements:
- Use the exact headings: "## Intuition", "## Example", "## Technical Details", "## Key Takeaways".
- Cite sources using bracketed numbers like [1] next to statements that use the sources.
- If you do not have relevant sources, say that clearly and avoid making up details.

Now respond as Andrew Ng.
""".strip()
