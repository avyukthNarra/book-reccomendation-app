# recommend.py

import joblib
import logging
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("recommend.log", encoding="utf-8"), logging.StreamHandler()]
)

logging.info("🔁 Loading processed data...")
df = joblib.load('df_cleaned.pkl')
cosine_sim = joblib.load('cosine_sim.pkl')
rating_weighted_sim = joblib.load('rating_weighted_sim.pkl')
count_weighted_sim = joblib.load('count_weighted_sim.pkl')
logging.info("✅ Data loaded.")

def _compose_similarity(idx: int, rating_weight: float, count_weight: float) -> np.ndarray:
    # Combine base + rating-weighted + count-weighted rows
    sim_weight = max(0.0, 1.0 - rating_weight - count_weight)
    row = (
        sim_weight * cosine_sim[idx]
        + rating_weight * rating_weighted_sim[idx]
        + count_weight * count_weighted_sim[idx]
    )
    # Ensure 1D numpy array even if any matrix is numpy.matrix
    if hasattr(row, "A1"):
        row = row.A1
    return np.asarray(row).ravel()

def recommend_books(book_title: str, top_n: int = 5, rating_weight: float = 0.3, count_weight: float = 0.2) -> pd.DataFrame | None:
    # Find the row index for the selected title (case-insensitive)
    idxs = df[df['book_title'].str.lower() == str(book_title).lower()].index
    if len(idxs) == 0:
        logging.warning("Book not found: %s", book_title)
        return None
    # Take the first element of the Index; int(Index) raises TypeError
    idx = int(idxs[0])

    # Rank by similarity, highest first
    final_sim = _compose_similarity(idx, rating_weight, count_weight)
    sim_scores = list(enumerate(final_sim))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Collect top-N with unique titles, skipping the same item and applying rating gates
    results = []
    seen_titles = set()
    for j, score in sim_scores:
        if j == idx:
            continue
        title_j = str(df.iloc[j]['book_title']).strip().lower()
        if title_j in seen_titles:
            continue
        r = df.iloc[j]['book_rating']
        rc = df.iloc[j]['book_rating_count']
        if r >= 2.0 and rc >= 5:
            results.append((j, score))
            seen_titles.add(title_j)
        if len(results) >= top_n:
            break

    # Fallback if too few after gates; still enforce unique titles
    if len(results) < top_n:
        for j, score in sim_scores:
            if j == idx:
                continue
            title_j = str(df.iloc[j]['book_title']).strip().lower()
            if title_j in seen_titles:
                continue
            results.append((j, score))
            seen_titles.add(title_j)
            if len(results) >= top_n:
                break

    # Build output DataFrame, include image_url so UI can show covers
    indices = [j for j, _ in results[:top_n]]
    scores = [s for _, s in results[:top_n]]
    cols = [
        'book_title', 'book_authors', 'genres', 'book_desc',
        'book_rating', 'book_rating_count'
    ]
    if 'image_url' in df.columns:
        cols.append('image_url')
    out = df.iloc[indices][cols].copy()
    out['similarity_score'] = scores
    out.index = range(1, len(out) + 1)
    out.index.name = "Rank"
    return out

def get_genre_recommendations(genre: str, top_n: int = 10) -> pd.DataFrame | None:
    if not genre:
        return None
    mask = df['genres'].str.contains(genre, case=False, na=False)
    subset = df[mask].copy()
    if subset.empty:
        return None
    subset['weighted_score'] = subset['book_rating'] * np.log(subset['book_rating_count'] + 1.0)
    cols = ['book_title', 'book_authors', 'genres', 'book_rating', 'book_rating_count']
    if 'image_url' in df.columns:
        cols.append('image_url')
    top = subset.nlargest(top_n, 'weighted_score')[cols].copy()
    top.index = range(1, len(top) + 1)
    top.index.name = "Rank"
    return top
