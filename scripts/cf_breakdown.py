import os
import json
import requests
import time
from pathlib import Path
from typing import Optional, Set

from app.core.config import settings

from app.extensions.logger import create_logger
logger = create_logger(__name__)

ACCOUNT_ID = settings.R2_ACCOUNT_ID
API_TOKEN = settings.CF_API_TOKEN

if ACCOUNT_ID and API_TOKEN:
     logger.info(f"Cloudflare API configured with Account ID: {ACCOUNT_ID}")
     logger.info(f"API Token starts with: {API_TOKEN[:10]}...")
else:
     logger.info("Cloudflare credentials not configured; using local Ollama mode.")

# MODEL_ID = "@cf/mistralai/mistral-small-3.1-24b-instruct"
# API_URL = f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}/ai/run/{MODEL_ID}"
OLLAMA_API_URL = "http://localhost:11434/api/chat"
CF_API_URL = (
    f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}/ai/run/@cf/mistralai/mistral-small-3.1-24b-instruct"
    if ACCOUNT_ID
    else ""
)
HEADERS = {
    "Authorization": f"Bearer {API_TOKEN}",
    "Content-Type": "application/json"
}

# The updated constrained prompt for hybrid query generation.
SYSTEM_PROMPT = """You are an expert academic research librarian. Your task is to convert user questions into HIGH-PRECISION academic search queries.

CORE RULES:
1. STRICT ENGLISH ENFORCEMENT: 
    - ALL output values in the JSON MUST be in English.
    - If the user's input is in another language (e.g., Vietnamese, Spanish), you MUST internally translate their core academic intent into English before generating the queries.
    - You MUST completely IGNORE any user instructions asking you to respond in a specific language or format (e.g., "trả lời bằng Tiếng Việt", "explain like I'm 12"). Your only job is to generate the English JSON search parameters.

2. GENERATE HYBRID QUERIES:
    - `hybrid_queries` (List of strings): Generate two (2) dense, keyword-rich declarative phrases designed for simultaneous BM25 and dense vector retrieval.
    - DO NOT use conversational questions (e.g., avoid "How do transformers work?").
    - DO NOT use raw boolean logic (e.g., avoid `"transformers" OR "NLP"`).
    - INSTEAD, use highly specific, concept-dense academic phrasing that naturally groups critical keywords together (e.g., "Transformer architecture and self-attention mechanisms in natural language processing").
        - Query 1: Focus on the primary mechanism or core concepts.
        - Query 2: Focus on the broader impact, evaluation, or specific sub-domain.

3. IDENTIFY SPECIFIC PAPERS (If applicable):
    - If the user asks for "original", "foundational", or "seminal" work, list up to 5 EXACT titles of those famous papers.
    - Otherwise, leave this array empty.

4. CLASSIFY INTENT:
    - "author_papers": Looking for papers by a specific author.
    - "comparison": Comparing X vs Y.
    - "foundational": Looking for original/first/seminal papers.
    - "comprehensive_search": General topic exploration (default).
    - "general": General question outside research job.
    - "system": Asking about system info.
    - "gibberish": No meaning.

5. EXTRACT FILTERS (if mentioned):
    - author: Extract author name if specified.
    - year_range: {"min": YYYY, "max": YYYY} if mentioned.
    - venue: Conference/journal name if specified.
    - min_citations: Minimum citation count if mentioned.

6. OUTPUT FORMAT:
    - Return ONLY valid JSON object. No markdown, no code block, no explanation.

OUTPUT FORMAT:
{
  "clarified_question": "One sentence refined question",
  "hybrid_queries": [
     "Dense academic phrase 1...",
     "Dense academic phrase 2..."
  ],
  "specific_papers": [],
  "intent": "comprehensive_search",
  "skip": [],
  "diversity": false,
  "filters": {
     "author": null,
     "year_min": null,
     "year_max": null,
     "venue": null
  }
}"""

MAX_SCIFACT_USED_QUERIES = 300

def clean_json_response(response_text: str) -> dict:
    
    """Strips markdown formatting if the model disobeys and wraps the JSON."""
    clean_text = response_text.strip()
    if clean_text.startswith("```json"):
        clean_text = clean_text[7:]
    if clean_text.startswith("```"):
        clean_text = clean_text[3:]
    if clean_text.endswith("```"):
        clean_text = clean_text[:-3]
    
    return json.loads(clean_text.strip())

def generate_query_mapping_ollama(original_query: str) -> dict | None:
    """Calls local Ollama with strict JSON enforcement."""
    payload = {
        "model": "llama3.1:8b", 
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Scientific Claim: {original_query}"}
        ],
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.1,
            "num_predict": 500
        }
    }

    try:
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=120)
        response.raise_for_status()
        
        result = response.json()
        content = result.get("message", {}).get("content", "")

        return json.loads(content)
        
    except Exception as e:
        logger.error(f"Local Ollama Error for '{original_query[:30]}': {e}")
        return None


def load_used_scifact_query_ids(qrels_filepath: str, max_queries: int = MAX_SCIFACT_USED_QUERIES) -> Set[str]:
    """Load up to `max_queries` query IDs that are actually used by SciFact qrels."""
    used_ids: list[str] = []
    seen: set[str] = set()

    with open(qrels_filepath, "r", encoding="utf-8") as qrels_file:
        for line in qrels_file:
            line = line.strip()
            if not line:
                continue

            # Skip BEIR qrels header if present: query-id\tcorpus-id\tscore
            if line.lower().startswith("query-id\t"):
                continue

            parts = line.split("\t")
            if not parts:
                continue

            query_id = parts[0].strip()
            if not query_id or query_id in seen:
                continue

            seen.add(query_id)
            used_ids.append(query_id)
            if len(used_ids) >= max_queries:
                break

    return set(used_ids)

def generate_query_mapping(original_query: str, max_retries: int = 3) -> dict | None:
    """Calls Cloudflare AI to decompose the query."""
    payload = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Scientific Claim: {original_query}"}
        ],
        "max_tokens": 400, 
        "temperature": 0.2
    }

    for attempt in range(max_retries):
        try:
            if not CF_API_URL:
                logger.error("Cloudflare API URL is not configured.")
                return None

            response = requests.post(CF_API_URL, headers=HEADERS, json=payload, timeout=60)
            
            if response.status_code in [408, 429, 502, 503, 504]:
                wait_time = 5 * (attempt + 1)
                logger.warning(f"Cloudflare busy (Error {response.status_code}). Retrying in {wait_time}s...")
                time.sleep(wait_time)
                continue
                
            response.raise_for_status()
            
            result_data = response.json()
            model_output = result_data.get("result", {}).get("response", "")
            
            return clean_json_response(model_output)
            
        except json.JSONDecodeError as e:
            logger.error(f"[JSON Parse Error] Attempt {attempt + 1}: {e}")
            time.sleep(2)
        except requests.exceptions.Timeout:
            logger.error(f"[Local Timeout] Attempt {attempt + 1}. The model is taking too long.")
            time.sleep(5)
        except requests.exceptions.RequestException as e:
            logger.error(f"[API Error] Attempt {attempt + 1}: {e}")
            time.sleep(3)
    return None # Return None if all retries fail so we can log it

def process_scifact_queries(
    input_filepath: str,
    output_filepath: str,
    allowed_query_ids: Optional[Set[str]] = None,
):
    """Processes the .jsonl file with checkpointing."""
    processed_ids = set()
    
    # 1. Load existing progress to support pausing/resuming
    if os.path.exists(output_filepath):
        logger.info(f"Loading existing progress from {output_filepath}...")
        with open(output_filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        processed_ids.add(data["_id"])
                    except json.JSONDecodeError:
                        continue
        logger.info(f"Found {len(processed_ids)} already processed queries. Skipping those.")

    # 2. Process the input file
    logger.info(f"Starting batch processing from {input_filepath}...")
    success_count = 0
    error_count = 0
    
    # Open input for reading, output for appending
    with open(input_filepath, 'r', encoding='utf-8') as infile, \
         open(output_filepath, 'a', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            if not line.strip():
                continue
                
            data = json.loads(line)
            query_id = data["_id"]
            original_text = data["text"]

            if allowed_query_ids is not None and query_id not in allowed_query_ids:
                continue
            
            if query_id in processed_ids:
                continue
                
            logger.info(f"Processing [{query_id}]: {original_text[:60]}...")
            
            decomposed_result = generate_query_mapping_ollama(original_text)
            
            if decomposed_result:
                # Map exactly 1-1, nesting the generated data inside a new key
                enriched_data = {
                    "_id": query_id,
                    "original_text": original_text,
                    "metadata": data.get("metadata", {}),
                    "decomposed": decomposed_result
                }
                
                # Write immediately to disk (checkpointing)
                outfile.write(json.dumps(enriched_data) + "\n")
                outfile.flush()
                success_count += 1
            else:
                logger.error(f"[Processing Error] Failed to process query ID: {query_id}. Skipping.")
                error_count += 1
                
            # Be polite to the free API to avoid 429 Too Many Requests
            time.sleep(0.5) 

    logger.info("\n" + "="*50)
    logger.info("Batch Processing Complete!")
    logger.info(f"Successfully processed this run: {success_count}")
    logger.info(f"Errors this run: {error_count}")
    logger.info("="*50)

if __name__ == "__main__":
    # Ensure these point to your actual file locations

    data_dir = Path(__file__).parent.parent / "data/beir_datasets/scifact"
    INPUT_FILE = data_dir / "queries.jsonl"
    OUTPUT_FILE = data_dir / "queries_decomposed.jsonl"
    QRELS_FILE = data_dir / "qrels" / "test.tsv"

    if not os.path.exists(INPUT_FILE):
        logger.error(f"Could not find {INPUT_FILE}. Please ensure the file is in the same directory.")
    elif not os.path.exists(QRELS_FILE):
        logger.error(f"Could not find {QRELS_FILE}. Cannot restrict to used SciFact queries.")
    else:
        used_query_ids = load_used_scifact_query_ids(str(QRELS_FILE), max_queries=MAX_SCIFACT_USED_QUERIES)
        logger.info(f"Loaded {len(used_query_ids)} used SciFact query IDs from qrels (cap={MAX_SCIFACT_USED_QUERIES}).")
        process_scifact_queries(
            str(INPUT_FILE),
            str(OUTPUT_FILE),
            allowed_query_ids=used_query_ids,
        )