# preprocess.py

import os
import pandas as pd
import re
import nltk
import joblib
import logging
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("preprocess.log", encoding="utf-8"), logging.StreamHandler()]
)

logging.info("🚀 Starting book preprocessing...")

# Resolve CSV path in src
HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, "book_data.csv")
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"CSV not found at {CSV_PATH}")
if os.path.getsize(CSV_PATH) == 0:
    raise ValueError(f"CSV at {CSV_PATH} is empty")

# Skip a leading "Table ..." line if present
with open(CSV_PATH, "r", encoding="utf-8") as f:
    first_line = f.readline().strip()
skiprows = 1 if first_line.lower().startswith("table") else 0

# Read CSV; ensure book_isbn is loaded as string to avoid scientific notation/precision loss
read_kwargs = {"skiprows": skiprows}
# If the file has book_isbn, force dtype to str; safe even if column missing
try:
    # Read a small preview to check columns
    preview = pd.read_csv(CSV_PATH, nrows=1, skiprows=skiprows)
    if "book_isbn" in preview.columns:
        read_kwargs["dtype"] = {"book_isbn": "string"}  # pandas StringDtype
except Exception:
    pass

df = pd.read_csv(CSV_PATH, **read_kwargs)
required = ["book_title", "book_authors", "book_desc", "genres", "book_rating", "book_rating_count"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

# Strictly keep complete rows and enforce numeric types
before = len(df)
df = df.dropna(subset=required).copy()
df['book_rating'] = pd.to_numeric(df['book_rating'], errors='coerce')
df['book_rating_count'] = pd.to_numeric(df['book_rating_count'], errors='coerce')
df = df.dropna(subset=['book_rating', 'book_rating_count'])
logging.info("🧹 Dropped incomplete/bad rows: %d removed, %d remain", before - len(df), len(df))

# ISBN de-duplication: keep the row with highest rating_count per ISBN
if 'book_isbn' in df.columns:
    # Normalize ISBN strings and remove blanks
    df['book_isbn'] = df['book_isbn'].astype(str).str.strip()
    has_isbn = df['book_isbn'].str.len() > 0
    df_isbn = df[has_isbn].copy()
    df_no_isbn = df[~has_isbn].copy()

    before_isbn = len(df_isbn)
    # Sort so drop_duplicates keeps the most-rated edition per ISBN
    df_isbn.sort_values(['book_isbn', 'book_rating_count'], ascending=[True, False], inplace=True)
    df_isbn = df_isbn.drop_duplicates(subset=['book_isbn'], keep='first')
    after_isbn = len(df_isbn)

    # Combine back rows without ISBN unchanged
    df = pd.concat([df_isbn, df_no_isbn], ignore_index=True)


# Select top 50000 by rating count (descending)
TOP_N = 50000
keep_n = min(TOP_N, len(df))
df = df.nlargest(keep_n, 'book_rating_count').reset_index(drop=True)

# Normalize rating and count
scaler = MinMaxScaler()
df['norm_rating'] = scaler.fit_transform(df[['book_rating']])
df['norm_rating_count'] = scaler.fit_transform(df[['book_rating_count']])

# Text preprocessing
def preprocess_text(text: str) -> str:
    text = re.sub(r"[^a-zA-Z\s]", " ", str(text))
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words and len(w) > 1]
    return " ".join(tokens)

# Build combined string with extra genre weight (same logic)
df['genres_repeated'] = df['genres'].astype(str) + ' ' + df['genres'].astype(str)
df['combined_text'] = df['genres_repeated'] + ' ' + df['book_desc'].astype(str) + ' ' + df['book_authors'].astype(str)

logging.info("🧹 Cleaning text (NLTK)...")
df['cleaned_text'] = df['combined_text'].apply(preprocess_text)
df = df[df['cleaned_text'].str.len() > 0].reset_index(drop=True)
logging.info("✅ Text cleaned. Docs: %d", len(df))

# TF-IDF with original parameters
logging.info("🔠 Vectorizing with TF-IDF (max_features=8000, ngram_range=(1,2))...")
tfidf = TfidfVectorizer(
    max_features=8000,
    ngram_range=(1, 2),
    stop_words='english',
    max_df=0.95,
    min_df=2
)
tfidf_matrix = tfidf.fit_transform(df['cleaned_text'])

# Full cosine similarity on reduced set
logging.info("📐 Computing full cosine similarity...")
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
logging.info("✅ Cosine similarity computed: %s", cosine_sim.shape)

# Apply rating/count weights as before
rating_weights = np.outer(df['norm_rating'], df['norm_rating'])
count_weights = np.outer(df['norm_rating_count'], df['norm_rating_count'])
rating_weighted_sim = cosine_sim * (0.7 + 0.3 * rating_weights)
count_weighted_sim = cosine_sim * (0.8 + 0.2 * count_weights)

# Save artifacts in src
logging.info("💾 Savi ng artifacts...")
joblib.dump(df, 'df_cleaned.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
joblib.dump(tfidf_matrix, 'tfidf_matrix.pkl')
joblib.dump(cosine_sim, 'cosine_sim.pkl')
joblib.dump(rating_weighted_sim, 'rating_weighted_sim.pkl')
joblib.dump(count_weighted_sim, 'count_weighted_sim.pkl')
logging.info("✅ Preprocessing complete and saved.")
