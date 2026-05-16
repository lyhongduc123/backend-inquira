from pathlib import Path

_RESPONSES_DIR = Path(__file__).resolve().parent / "responses"


def _load_response_template(key: str, default: str) -> str:
    try:
        path = _RESPONSES_DIR / f"{key}.txt"
        if not path.exists():
            return default

        content = path.read_text(encoding="utf-8").strip()
        return content or default
    except Exception:
        return default