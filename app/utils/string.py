import re

def surname(name: str) -> str:
    name = re.sub(r"[^\w\s]", "", name.lower())
    return name.split()[-1]