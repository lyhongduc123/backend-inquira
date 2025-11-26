import requests
from typing import List, Dict, Any
from bs4 import BeautifulSoup
import time
import random

class Crawler:
    def __init__(self, headers: Dict[str, str] | None = None):
        # Rotate User-Agents to avoid detection
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ]
        self.headers = headers or self._get_random_headers()
        self.session = requests.Session()

    def _get_random_headers(self) -> Dict[str, str]:
        return {
            "User-Agent": random.choice(self.user_agents),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1"
        }

    def fetch_page(self, url: str, retry_delay: int = 5, max_retries: int = 3) -> str:
        """Fetches the content of a web page with retry logic and rate limiting."""
        for attempt in range(max_retries):
            try:
                # Random delay to avoid rate limiting (3-7 seconds)
                time.sleep(random.uniform(3, 7))
                
                # Rotate headers for each request
                self.headers = self._get_random_headers()
                
                response = self.session.get(url, headers=self.headers, timeout=10)
                response.raise_for_status()
                
                # Check if we got blocked (CAPTCHA page)
                if "captcha" in response.text.lower() or response.status_code == 429:
                    print(f"⚠️ Rate limited or CAPTCHA detected. Waiting {retry_delay * (attempt + 1)}s...")
                    time.sleep(retry_delay * (attempt + 1))
                    continue
                
                return response.text
                
            except requests.exceptions.RequestException as e:
                print(f"❌ Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    raise
        
        raise Exception("Failed to fetch page after maximum retries")

    def parse_google_scholar(self, html_content: str) -> List[Dict[str, Any]]:
        """Parses Google Scholar HTML content and extracts paper information."""
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Google Scholar uses different class names - check current structure
        # Try multiple possible selectors
        results = soup.find_all("div", class_="gs_ri") or soup.find_all("div", class_="gs_r")
        
        if not results:
            print("⚠️ No results found. HTML structure may have changed or page blocked.")
            print(f"Page preview: {html_content[:500]}")
            return []
        
        papers = []
        for result in results:
            try:
                # Title
                title_tag = result.find("h3", class_="gs_rt") or result.find("h3")
                title = title_tag.get_text(strip=True) if title_tag else "No Title"
                
                # Remove [PDF], [HTML], [CITATION] markers
                title = title.replace('[PDF]', '').replace('[HTML]', '').replace('[CITATION]', '').strip()
                
                # Authors and publication info
                info_tag = result.find("div", class_="gs_a")
                authors = info_tag.get_text(strip=True) if info_tag else "Unknown"
                
                # Abstract/snippet
                snippet_tag = result.find("div", class_="gs_rs")
                snippet = snippet_tag.get_text(strip=True) if snippet_tag else ""
                
                # Link
                link_tag = title_tag.find("a") if title_tag else None
                link = link_tag.get("href") if link_tag else None
                
                # Citation count
                citation_tag = result.find("div", class_="gs_fl")
                citation_text = citation_tag.get_text() if citation_tag else ""
                cited_by = 0
                if "Cited by" in citation_text:
                    try:
                        cited_by = int(citation_text.split("Cited by")[1].split()[0])
                    except:
                        pass
                
                papers.append({
                    "title": title,
                    "authors": authors,
                    "snippet": snippet,
                    "url": link,
                    "cited_by": cited_by
                })
                
            except Exception as e:
                print(f"⚠️ Error parsing result: {e}")
                continue
        
        return papers

    def search_google_scholar(self, query: str, num_results: int = 10) -> List[Dict[str, Any]]:
        """Search Google Scholar for papers matching the query."""
        base_url = "https://scholar.google.com/scholar"
        all_papers = []
        
        for start in range(0, num_results, 10):
            params = {
                "q": query,
                "start": start
            }
            
            # Build URL with parameters
            url = f"{base_url}?q={requests.utils.quote(query)}&start={start}"
            
            try:
                html = self.fetch_page(url)
                papers = self.parse_google_scholar(html)
                all_papers.extend(papers)
                
                if len(papers) < 10:  # No more results
                    break
                    
            except Exception as e:
                print(f"Error fetching page {start}: {e}")
                break
        
        return all_papers[:num_results]

    def parse_content(self, html_content: str) -> List[Dict[str, Any]]:
        """Legacy method - redirects to parse_google_scholar."""
        return self.parse_google_scholar(html_content)
    
crawler = Crawler()