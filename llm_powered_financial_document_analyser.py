import requests
import json
import time

# --- CONFIGURATION ---
"""
Replace placeholder with Gemini API Key, if running in a secure environment where the key
is provided via env var, use that instead.
"""
API_KEY = ""
MODEL_NAME = "gemini-2.5-flash-preview-09-2025"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={API_KEY}"

# The System Instruction defines the model's role and output format.
SYSTEM_PROMPT = (
    "Act as a world-class financial analyst. "
    "Provide a concise, single-paragraph summary of the key findings."
)

# --- HELPER FUNCTIONS ---

def exponential_backoff_fetch(url, payload, max_retries=5):
    """
    Performs an HTTP POST request with exponential backoff for retries.
    This handles potential transient network or API throttling errors.
    """
    headers = {'Content-Type': 'application/json'}

    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, data=json.dumps(payload))
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
            return response.json()
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                # print(f"Request failed ({e}). Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"Final attempt failed. Error: {e}")
                return None
    return None

def analyze_financial_query(user_query):
    """
    Calls the Gemini API with grounding and system instructions.
    """
    print(f"\n--- Analyzing Query: {user_query} ---\n")

    # 1. Construct the API Payload
    payload = {
        "contents": [{
            "parts": [{"text": user_query}]
        }],

        # 2. Add System Instruction to define the analyst persona
        "systemInstruction": {
            "parts": [{"text": SYSTEM_PROMPT}]
        },

        # 3. Enable Google Search Grounding Tool for real-time data
        "tools": [{"google_search": {}}],
    }

    # 4. Make the API Call with Backoff
    response_data = exponential_backoff_fetch(API_URL, payload)

    if not response_data:
        print("Failed to get a response from the API.")
        return

    # 5. Process the Response
    try:
        candidate = response_data['candidates'][0]

        # Extract the generated text
        generated_text = candidate['content']['parts'][0]['text']
        print("✅ Generated Summary:\n")
        print(generated_text)
        print("\n" + "="*50)

        # Extract Grounding Attributions (Citations)
        grounding_metadata = candidate.get('groundingMetadata')
        if grounding_metadata and grounding_metadata.get('groundingAttributions'):
            sources = []
            for attribution in grounding_metadata['groundingAttributions']:
                web_info = attribution.get('web')
                if web_info:
                    sources.append({
                        "title": web_info.get('title'),
                        "uri": web_info.get('uri')
                    })

            print("🌐 Sources Used (Grounding Attributions):")
            for i, source in enumerate(sources):
                print(f"  {i+1}. Title: {source['title']}")
                print(f"     URI: {source['uri']}")
        else:
            print("No external sources were cited for grounding.")

    except (KeyError, IndexError) as e:
        print(f"Error parsing API response structure: {e}")
        print("Raw Response:", json.dumps(response_data, indent=2))

# --- MAIN EXECUTION ---

if __name__ == '__main__':
    if not API_KEY:
        print("="*70)
        print("SETUP REQUIRED: Please provide your API_KEY in the script.")
        print("The code structure is correct, but the API call will fail without a key.")
        print("="*70)

    # Example Query using Google Search Grounding
    query = "Summarize the key takeaways from the latest quarterly earnings report for Apple Inc. (AAPL)."
    analyze_financial_query(query)
