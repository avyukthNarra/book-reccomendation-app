# google_books_utils.py

import requests

def get_book_details(title: str, authors: str | list = "", api_key: str | None = None) -> tuple[str, str]:
    """
    Fetch description and thumbnail from Google Books API by title and optional author.
    Returns (description, thumbnail) or ("", "") if not found.
    """
    if not title:
        return "", ""

    # Normalize authors to a single first author string
    first_author = ""
    if isinstance(authors, list):
        # Take the first non-empty string element
        for a in authors:
            if isinstance(a, str) and a.strip():
                first_author = a.strip()
                break
    elif isinstance(authors, str) and authors.strip():
        # Split once on '|' to get the first segment, then once on ',' to get the first name
        part = authors.split("|", 1)[0].strip()
        first_author = part.split(",", 1)[0].strip()

    # Build query
    q_parts = [f'intitle:"{title}"']
    if first_author:
        q_parts.append(f'inauthor:"{first_author}"')
    q = " ".join(q_parts)

    params = {
        "q": q,
        "maxResults": 1,
        "printType": "books",
    }
    if api_key:
        params["key"] = api_key

    try:
        resp = requests.get("https://www.googleapis.com/books/v1/volumes", params=params, timeout=10)
        if resp.status_code != 200:
            return "", ""
        data = resp.json()
        items = data.get("items", [])
        if not items:
            return "", ""
        vi = items[0].get("volumeInfo", {})
        description = vi.get("description", "") or ""
        links = vi.get("imageLinks", {}) or {}
        thumb = (
            links.get("large")
            or links.get("medium")
            or links.get("small")
            or links.get("thumbnail")
            or ""
        )
        if isinstance(thumb, str) and thumb.startswith("http:"):
            thumb = thumb.replace("http:", "https:", 1)
        return description, thumb
    except Exception:
        return "", ""
