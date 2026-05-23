from ddgs import DDGS
import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral-medium-latest")
MISTRAL_URL = "https://api.mistral.ai/v1/chat/completions"


def keyword_score(query: str, text: str) -> int:
    query_words = set(query.lower().split())
    text_words = set(text.lower().split())
    return len(query_words.intersection(text_words))


def retrieve_relevant_chunks(question: str, chunks: list[dict], top_k: int = 4) -> list[dict]:
    scored = []

    for chunk in chunks:
        score = keyword_score(question, chunk["content"])
        scored.append({
            "content": chunk["content"],
            "score": score
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


def web_search(query: str, max_results: int = 5) -> list[dict]:
    """
    Search DuckDuckGo for the query and return results.
    """
    results = []
    print(f"DEBUG: Searching web for: {query}")
    try:
        with DDGS() as ddgs:
            # Using backend='auto' which is the default but more robust
            ddgs_results = ddgs.text(query, max_results=max_results)
            for r in ddgs_results:
                results.append({
                    "title": r.get("title", ""),
                    "body": r.get("body", ""),
                    "href": r.get("href", "")
                })
        print(f"DEBUG: Found {len(results)} web results")
    except Exception as e:
        print(f"Web search error: {e}")
    return results


def build_rag_prompt(question, relevant_chunks, history, web_results=None):
    context = "\n\n".join([chunk["content"] for chunk in relevant_chunks])
    
    web_context = ""
    if web_results:
        web_context = "\n\n".join([
            f"Source: {r['title']} ({r['href']})\nContent: {r['body']}"
            for r in web_results
        ])

    history_text = ""
    for msg in history[-5:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_text += f"{role}: {msg['content']}\n"

    is_summary = "summarise" in question.lower() or "summarize" in question.lower()
    is_analysis = "analyse" in question.lower() or "analyze" in question.lower() or "chart" in question.lower() or "graph" in question.lower()

    if is_analysis:
        instructions = """
- You are acting as a Data Analyst.
- Analyze the provided CSV Data Summary and Sample Data.
- If the user asks for a chart or graph, identify the appropriate chart type (bar, line, pie, etc.).
- IMPORTANT: To render a chart, you MUST include a JSON block in your response starting with '---CHART_DATA---' and ending with '---END_CHART_DATA---'.
- The JSON inside should be a Chart.js configuration object.
- Example format:
---CHART_DATA---
{
  "type": "bar",
  "data": {
    "labels": ["Jan", "Feb", "Mar"],
    "datasets": [{
      "label": "Sales",
      "data": [10, 20, 30]
    }]
  },
  "options": {
    "responsive": true,
    "plugins": { "title": { "display": true, "text": "Monthly Sales" } }
  }
}
---END_CHART_DATA---

- Provide a textual analysis first, explaining the insights, and then include the chart data block.
- Be accurate and precise with the numbers from the data.
"""
    elif is_summary:
        instructions = """
- Provide a comprehensive summary of the document based on the provided context.
- Highlight the most important points, key findings, or major themes.
- IMPORTANT: Document Context chunks contain page numbers like '[Page X] text...'. You MUST cite the source pages using the exact format '[Page X]' (e.g. '[Page 1]', '[Page 4]') next to statements or headings to indicate where the information came from.
- Organize the information logically with clear headings.
- Use the following structure:
   Executive Summary
   Key Points & Highlights
   Detailed Analysis (if applicable)
   Conclusion
"""
    else:
        instructions = """
- Prioritize information from the Document Context if it contains the answer.
- IMPORTANT: Document Context chunks contain page numbers, e.g. '[Page X] text...'. When using information from a specific page, you MUST cite it using the exact format '[Page X]' (e.g. 'As mentioned in [Page 3]...' or 'the system maintains high availability [Page 12]').
- Use Web Search Context to supplement information or if the answer is not in the Document Context.
- If Web Search Context is used, mention that the information was found on the web.
- Answer in a detailed and well-structured manner.
- Use simple and clear explanations.
- Break the answer into sections:
   Explanation
   Key Points
   Conclusion
"""

    import datetime
    current_date = datetime.date.today().strftime("%B %d, %Y")

    if not web_results and not relevant_chunks:
        instructions += "\n- IMPORTANT: Since no live Web Search Context or Document Context is active/provided, answer using your general knowledge but explicitly remind the user to toggle the 'Web Search' (globe icon) in the toolbar below the chat input box if they need real-time, live, or latest news."

    prompt = f"""
You are AnswerAI, an intelligent assistant. You answer questions based on the provided Document Context and/or Web Search Context.

Current Date Context: Today's date is {current_date}. Keep this in mind when answering questions about events, schedules, dates, or recency.

Instructions:
{instructions}

Conversation History:
{history_text}

Document Context:
{context if context else "No document context available."}

Web Search Context:
{web_context if web_context else "No web search context available."}

Question:
{question}

Answer:
"""
    return prompt


def ask_ai(prompt: str, model_provider: str = "mistral") -> str:
    if model_provider != "mistral":
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            return "Error: OPENROUTER_API_KEY not found in backend/.env. Please add it to your .env file: OPENROUTER_API_KEY=your_key"
        
        # Map provider keys to OpenRouter Model IDs
        model_mapping = {
            "qwen": "qwen/qwen3-coder:free",
            "openrouter_free": "openrouter/free",
            "gemma": "google/gemma-2-9b-it:free",
            "llama3": "meta-llama/llama-3-8b-instruct:free",
            "llama3_3": "meta-llama/llama-3.3-70b-instruct:free"
        }
        
        primary_model = model_mapping.get(model_provider, "openrouter/free")
        
        # Define the set of reliable free models to iterate through if a failure occurs
        fallback_models = [
            "openrouter/free",
            "google/gemma-2-9b-it:free",
            "meta-llama/llama-3-8b-instruct:free",
            "qwen/qwen3-coder:free",
            "meta-llama/llama-3.3-70b-instruct:free"
        ]
        
        # Order the sequence so the selected primary model is tried first
        if primary_model in fallback_models:
            fallback_models.remove(primary_model)
        fallback_models.insert(0, primary_model)
        
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": "AnswerAI"
        }
        
        last_error = ""
        for model in fallback_models:
            print(f"DEBUG: Trying OpenRouter model: {model}")
            payload = {
                "model": model,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7,
                "max_tokens": 1024
            }
            try:
                response = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=60
                )
                
                # Check for rate limit (429) or other upstream errors, proceed to fallback if not 200
                if response.status_code == 200:
                    data = response.json()
                    content = data["choices"][0]["message"]["content"]
                    # If we fell back, let the user know beautifully
                    if model != primary_model:
                        friendly_name = model.split("/")[-1].replace(":free", "").upper()
                        content = f"*(Note: The selected model '{primary_model.split('/')[-1]}' was rate-limited or unavailable. Seamlessly fell back to {friendly_name})*\n\n" + content
                    return content
                
                last_error = f"API Error ({response.status_code}): {response.text}"
                print(f"DEBUG: Model {model} failed with status {response.status_code}. Error: {response.text}")
                
            except Exception as e:
                last_error = f"API Connection Error: {str(e)}"
                print(f"DEBUG: Model {model} raised exception: {e}")
                
        return f"All OpenRouter fallback models failed. Last error: {last_error}"
        
    else:
        # Default Mistral AI Provider
        if not MISTRAL_API_KEY:
            return "Error: Mistral API key not found. Please set MISTRAL_API_KEY in the .env file."
        
        url = MISTRAL_URL
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {MISTRAL_API_KEY}"
        }
        payload = {
            "model": MISTRAL_MODEL,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 1024
        }
        
        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code != 200:
                return f"API Error ({response.status_code}): {response.text}"
                
            data = response.json()
            return data["choices"][0]["message"]["content"]
            
        except Exception as e:
            return f"API Connection Error: {str(e)}"