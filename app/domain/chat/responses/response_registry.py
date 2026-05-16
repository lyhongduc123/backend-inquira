
from .response_defaults import DEFAULT_RESPONSES
from .loader import _load_response_template


class ResponseRegistry:
    @staticmethod
    def get(key: str) -> str:
        default = DEFAULT_RESPONSES.get(key, "")
        return _load_response_template(key, default)