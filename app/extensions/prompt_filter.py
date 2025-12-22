import math
import re
from typing import Set

from wordfreq import zipf_frequency

VOWELS: Set[str] = set("aeiou")


def _vowel_ratio(text: str) -> float:
    """Calculate ratio of vowels in alphabetic characters."""
    letters = re.sub(r"[^a-zA-Z]", "", text.lower())
    if len(letters) == 0:
        return 0.0
    if len(letters) < 3:
        return 1.0  # short strings get a pass on vowel ratio

    vowels = sum(1 for c in letters if c in VOWELS)
    return vowels / len(letters)


def _dictionary_ratio(text: str) -> float:
    """Calculate ratio of valid English words."""
    words = re.findall(r"[a-zA-Z]+", text.lower())
    if not words:
        return 0.0

    valid = sum(1 for w in words if len(w) > 1 and zipf_frequency(w, "en") > 2.5)
    return valid / len(words)


def _shannon_entropy(text: str) -> float:
    """Calculate Shannon entropy of the text."""
    freq = {}
    text = text.replace(" ", "")
    if not text:
        return 0.0

    for c in text:
        freq[c] = freq.get(c, 0) + 1

    entropy = 0.0
    length = len(text)
    for count in freq.values():
        p = count / length
        entropy -= p * math.log2(p)

    return entropy


def _is_common_greeting(text: str) -> bool:
    """Check if text is a common greeting or phrase that should be blocked."""
    common_phrases = {
        "hi", "hello", "hey", "thanks", "thank you", "ok", "okay",
        "yes", "no", "bye", "goodbye", "please", "help",
    }
    normalized = text.lower().strip()
    return normalized in common_phrases


def _looks_like_identifier(text: str) -> bool:
    """Check if text looks like a username, variable, or identifier."""
    # USER123, f_1238dan, admin_user, etc.
    if re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*\d+", text):
        return True
    # Underscores or mixed alphanumeric with low vowel ratio
    if "_" in text or re.search(r"[a-z][0-9]|[0-9][a-z]", text.lower()):
        letters = re.sub(r"[^a-zA-Z]", "", text.lower())
        if len(letters) > 2:
            vowel_ratio = sum(1 for c in letters if c in VOWELS) / len(letters)
            if vowel_ratio < 0.25:
                return True
    return False


def _is_short_nonsense(text: str) -> bool:
    """Check if short text is nonsense (3-5 chars with no valid words)."""
    if len(text) <= 5:
        # Check if it's a valid short word
        return zipf_frequency(text.lower(), "en") < 2.5
    return False


def is_gibberish(text: str) -> bool:
    """
    Detect gibberish input that shouldn't trigger LLM processing.

    Blocks:
    - Single characters or very short strings
    - Random character sequences (jfbdasjbfkabskf, asd)
    - Identifier-like strings (USER123, f_1238dan)
    - Common greetings that don't need LLM responses

    Args:
        text: Input string to check

    Returns:
        True if the text is gibberish and should be blocked
    """
    if not text or not text.strip():
        return True

    text = text.strip()

    # Block single characters or very short inputs
    if len(text) <= 2:
        return True

    # Block common greetings
    if _is_common_greeting(text):
        return True

    # Block identifier-like strings
    if _looks_like_identifier(text):
        return True

    # Block short nonsense words (asd, qwe, etc.)
    if _is_short_nonsense(text):
        return True

    # Check for meaningful content
    score = 0

    # Vowel ratio check (adjusted thresholds)
    vowel_ratio = _vowel_ratio(text)
    if vowel_ratio < 0.2 or vowel_ratio > 0.9:  # too few or too many vowels
        score += 1

    # Dictionary words check
    dict_ratio = _dictionary_ratio(text)
    if dict_ratio < 0.3:  # less than 30% valid words
        score += 1

    # Entropy check (randomness)
    entropy = _shannon_entropy(text)
    if entropy > 4.5:  # very high entropy = random
        score += 1

    # Structure checks
    alpha_chars = re.sub(r"[^a-zA-Z]", "", text)
    if len(alpha_chars) > 5 and not re.search(r"[aeiouAEIOU]", alpha_chars):
        score += 1

    # Repetitive patterns (aaaa, 1111, etc.)
    if re.search(r"(.)\1{4,}", text):
        score += 1

    return score >= 2