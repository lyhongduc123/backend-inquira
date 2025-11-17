import requests
from typing import List, Dict, Any
from bs4 import BeautifulSoup

class Crawler:
    def __init__(self, headers: Dict[str, str] | None = None):
        self.headers = headers or {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
        }


    def fetch_page(self, url: str) -> str:
        """Fetches the content of a web page given its URL."""
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.text

    def parse_content(self, html_content: str) -> List[Dict[str, Any]]:
        """Parses HTML content and extracts relevant information."""

        soup = BeautifulSoup(html_content, 'html.parser')
        results = soup.find_all("div", class_="gs_r")
        print(results)
        papers = []
        for result in results:
            h3 = result.find("h3")
            title = h3.get_text(strip=True) if h3 is not None else "No Title"
            # citation = result.find("div", class_="gs_fl").text if result.find("div", class_="gs_fl") else "No Citation"

            papers.append({"title": title})
        return papers
    
crawler = Crawler()