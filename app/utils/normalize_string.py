def normalize_string(string: str) -> str:
    if not string:
        return ""

    normalized = string.lower().strip()
    normalized = (
        normalized.replace(".", "")
        .replace(",", "")
        .replace(":", "")
        .replace(";", "")
    )
    normalized = " ".join(normalized.split())
    return normalized