import sys
import os

# Add the current directory to sys.path to import from backend
sys.path.append(os.getcwd())

from ddgs import DDGS

def test_search():
    print("Testing DuckDuckGo Search (using ddgs package)...")
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text("What is the current price of Bitcoin?", max_results=3))
            print(f"DEBUG: Found {len(results)} results")
            for i, r in enumerate(results):
                print(f"\nResult {i+1}:")
                print(f"Title: {r.get('title')}")
                print(f"Snippet: {r.get('body')}")
                print(f"URL: {r.get('href')}")
    except Exception as e:
        print(f"Search failed: {e}")

if __name__ == "__main__":
    test_search()
