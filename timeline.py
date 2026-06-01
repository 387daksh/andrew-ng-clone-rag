from pathlib import Path
import json

TIMELINE_PATH = Path(__file__).resolve().parent / "docs" / "timeline.json"

def load_timeline(path=TIMELINE_PATH):
    if not Path(path).exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def format_timeline_for_prompt(timeline):
    if not timeline:
        return "No timeline data available."
    lines = []
    for item in timeline:
        year = str(item.get("year", "")).strip()
        event = str(item.get("event", "")).strip()
        if year and event:
            lines.append(f"{year} - {event}")
        elif event:
            lines.append(event)
    return "\n".join(lines)
