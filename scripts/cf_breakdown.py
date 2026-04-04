import os
import json
import requests
import time
from pathlib import Path
from app.core.config import settings
from app.llm.ollama_client import OllamaClient

ACCOUNT_ID = settings.R2_ACCOUNT_ID
API_TOKEN = settings.CF_API_TOKEN

if not ACCOUNT_ID or not API_TOKEN:
    raise ValueError("Missing Cloudflare credentials in environment variables.")

print(f"Using Cloudflare API with Account ID: {ACCOUNT_ID}")
print(f"API Token starts with: {API_TOKEN[:10]}...")

# MODEL_ID = "@cf/mistralai/mistral-small-3.1-24b-instruct"
# API_URL = f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}/ai/run/{MODEL_ID}"
API_URL = "http://localhost:11434/api/chat"
HEADERS = {
    "Authorization": f"Bearer {API_TOKEN}",
    "Content-Type": "application/json"
}

# The highly-constrained prompt we designed
SYSTEM_PROMPT = """You are an expert academic research librarian. Your task is to analyze a scientific claim or query and decompose it for a hybrid RAG database search engine.

CORE RULES:
1. STRICT ENGLISH ENFORCEMENT: 
   - ALL output values in the JSON MUST be in English.
   - If the input is in another language, internally translate the core academic intent into English before generating the queries.

2. FORMAT CONSTRAINTS:
   - Return ONLY a raw, valid JSON object. 
   - DO NOT include markdown formatting (like ```json), conversational text, or explanations.

3. GENERATE EXACTLY THESE 3 FIELDS:
   - `clarified_question`: A fully self-contained, clear version of the original scientific claim. Resolve ambiguities.
   - `bm25_query`: A STRICT boolean search string extracting core entities and synonyms. Use ONLY the `OR` operator to link synonyms. Wrap multi-word concepts in quotes. Do NOT use 'AND'.
   - `semantic_queries`: Generate exactly two (2) natural-language questions for dense vector retrieval. 
     - Query 1: Focus on the broad conceptual relationship.
     - Query 2: Focus on the specific biological/chemical mechanism or outcome.

OUTPUT FORMAT:
{
  "clarified_question": "One sentence refined question",
  "bm25_query": "\"exact phrase\" OR \"synonym\"",
  "semantic_queries": [
    "Synthetic semantic query 1...",
    "Synthetic semantic query 2..."
  ]
}"""

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
        "format": "json", # <--- THE MAGIC BUTTON: Forces valid JSON
        "options": {
            "temperature": 0.1,
            "num_predict": 500
        }
    }

    try:
        # Increase timeout to 120s since it's local and 24B models can be slow
        response = requests.post(API_URL, json=payload, timeout=120)
        response.raise_for_status()
        
        result = response.json()
        content = result.get("message", {}).get("content", "")
        
        # No more "Regex cleaners" needed! Ollama's 'format: json' is perfect.
        return json.loads(content)
        
    except Exception as e:
        print(f"❌ Local Ollama Error for '{original_query[:30]}': {e}")
        return None

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
            response = requests.post(API_URL, headers=HEADERS, json=payload, timeout=60)
            
            if response.status_code in [408, 429, 502, 503, 504]:
                wait_time = 5 * (attempt + 1)
                print(f"  -> ⚠️ Cloudflare busy (Error {response.status_code}). Retrying in {wait_time}s...")
                time.sleep(wait_time)
                continue
                
            response.raise_for_status()
            
            result_data = response.json()
            model_output = result_data.get("result", {}).get("response", "")
            
            return clean_json_response(model_output)
            
        except json.JSONDecodeError as e:
            print(f"  -> [JSON Parse Error] Attempt {attempt + 1}: {e}")
            time.sleep(2)
        except requests.exceptions.Timeout:
            print(f"  -> [Local Timeout] Attempt {attempt + 1}. The model is taking too long.")
            time.sleep(5)
        except requests.exceptions.RequestException as e:
            print(f"  -> [API Error] Attempt {attempt + 1}: {e}")
            time.sleep(3)
            
    return None # Return None if all retries fail so we can log it

def process_scifact_queries(input_filepath: str, output_filepath: str):
    """Processes the .jsonl file with checkpointing."""
    processed_ids = set()
    
    # 1. Load existing progress to support pausing/resuming
    if os.path.exists(output_filepath):
        print(f"Loading existing progress from {output_filepath}...")
        with open(output_filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        processed_ids.add(data["_id"])
                    except json.JSONDecodeError:
                        continue
        print(f"Found {len(processed_ids)} already processed queries. Skipping those.")

    # 2. Process the input file
    print(f"Starting batch processing from {input_filepath}...")
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
            
            if query_id in processed_ids:
                continue
                
            print(f"Processing [{query_id}]: {original_text[:60]}...")
            
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
                print(f"❌ Failed to process query ID: {query_id}. Skipping.")
                error_count += 1
                
            # Be polite to the free API to avoid 429 Too Many Requests
            time.sleep(0.5) 

    print("\n" + "="*50)
    print("Batch Processing Complete!")
    print(f"Successfully processed this run: {success_count}")
    print(f"Errors this run: {error_count}")
    print("="*50)


if __name__ == "__main__":
    # Ensure these point to your actual file locations

    data_dir = Path(__file__).parent.parent / "data/beir_datasets/scifact"
    INPUT_FILE = data_dir / "queries.jsonl"
    OUTPUT_FILE = data_dir / "queries_decomposed.jsonl"

    if not os.path.exists(INPUT_FILE):
        print(f"Could not find {INPUT_FILE}. Please ensure the file is in the same directory.")
    else:
        process_scifact_queries(str(INPUT_FILE), str(OUTPUT_FILE))