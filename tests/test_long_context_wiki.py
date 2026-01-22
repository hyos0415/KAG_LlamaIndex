import asyncio
import json
import random
import os
import sys

# Add project root to path
sys.path.append(os.getcwd())

from core.rlm_evaluator import RLMEvaluator

async def main():
    file_path = "wikipedia_documents.json"
    print(f"Loading {file_path}...")
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: {file_path} not found.")
        return

    # Data is a dict of dicts: {"0": {"text": "...", "title": "..."}, "1": ...}
    if isinstance(data, dict):
        doc_ids = list(data.keys())
        print(f"Loaded {len(doc_ids)} documents.")
        
        # Pick a random document to test retrieval
        target_id = random.choice(doc_ids)
        target_doc = data[target_id]
        
        title = target_doc.get('title', 'Unknown Title')
        text = target_doc.get('text', '')
        
        # Get the first sentence for exact quote verification
        first_sentence = text.split('.')[0] if text else ""
    else:
        print(f"Unexpected data type: {type(data)}")
        return

    print(f"Selected Target Document ID: {target_id}, Title: {title}")
    print(f"Ground Truth First Sentence (approx): {first_sentence}")
    
    query = f"Find the document titled '{title}' in the context and summarize its main content. Also, quote the first sentence exactly."
    
    # Initialize RLM
    from core.lib.rlm.rlm_repl import RLM_REPL
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found.")
        return

    rlm = RLM_REPL(api_key=api_key, model="gpt-5.1", enable_logging=True)
    
    print("\n--- Starting RLM Verification on Long Context ---")
    print(f"Query: {query}")
    
    # We pass the WHOLE dataset as context. 
    response = await asyncio.to_thread(rlm.completion, context=data, query=query)
    
    print("\n--- Final Answer ---")
    print(response)
    
    print("\n--- Evaluation ---")
    # Simple check: Does the response contain the title and some overlap with the first sentence?
    # Ideally, we would use an LLM-as-judge here for "summarization quality", but a string check is a good sanity test.
    
    if title in response:
        print(f"[PASS] Response contains the document title: '{title}'")
    else:
        print(f"[FAIL] Response does NOT contain the document title: '{title}'")
        
    # Check for first sentence overlap (allowing for some formatting diffs)
    # Taking a substantial substring of the first sentence to check
    check_phrase = first_sentence[:20] if len(first_sentence) > 20 else first_sentence
    if check_phrase in response:
         print(f"[PASS] Response appears to quote the first sentence (found '{check_phrase}...')")
    else:
         print(f"[WARN] Response might not have quoted the first sentence exactly. Expected to find '{check_phrase}...'")

if __name__ == "__main__":
    asyncio.run(main())
