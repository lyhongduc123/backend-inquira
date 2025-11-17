import requests
from typing import Optional, Dict, Any

class HTTPClient:
    def __init__(self, timeout: float = 10.0, retries: int = 1):
        self.timeout = timeout
        self.retries = retries

    def get(self, 
            url: str, 
            params: Optional[Dict[str, Any]] = None, 
            headers: Optional[Dict[str, str]] = None, 
            timeout: Optional[float] = None) -> Optional[Any]:
        for attempt in range(1, self.retries + 1):
            try:
                response = requests.get(url, params=params, headers=headers, timeout=timeout or self.timeout)
                response.raise_for_status()
                return response
            except requests.Timeout:
                print(f"Timeout on attempt {attempt} for {url}")
            except requests.RequestException as e:
                print(f"Request error on attempt {attempt} for {url}: {e}")
        return None
