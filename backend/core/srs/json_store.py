import json
from pathlib import Path
from typing import Any

DEFAULT_STATE: dict[str, Any] = {
    "version": 1,
    "items": {},
}


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"version": 1, "items": {}}

    data = json.loads(path.read_text(encoding="utf-8"))
    if "version" not in data:
        data["version"] = 1
    if "items" not in data:
        data["items"] = {}

    return data


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(state, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )


def upsert_items(state: dict[str, Any], items: list[dict[str, Any]]) -> None:
    store = state.setdefault("items", {})
    for item in items:
        store[item["id"]] = item
