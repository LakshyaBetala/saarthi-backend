# modules/search.py
import requests
from urllib.parse import quote
import webbrowser
from .audio import speak

def google_search_summary(query: str):
    try:
        search_url = f"https://www.google.com/search?q={quote(query)}"
        wiki_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote(query)}"

        # Speak and open browser
        speak(f"Searching Google for {query}")
        webbrowser.open(search_url)

        # Try Wikipedia summary
        response = requests.get(wiki_url, headers={"User-Agent": "Mozilla/5.0"})
        if response.status_code == 200:
            data = response.json()
            if "extract" in data:
                summary = data["extract"]
                speak(f"According to Wikipedia: {summary[:300]}")
                return {"summary": summary}

        return {"summary": "No detailed result found. Check your browser."}

    except Exception as e:
        return {"error": str(e)}
