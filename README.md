# 📚 Book Recommender

A fast, interactive web application to recommend books based on **content and ratings**. Pick any book from a large dataset and receive smart, data-powered suggestions, with covers, genres, ratings, and detailed features. Ideal for finding your next favorite read!

## Features

- **Content-Based Recommendations:** Uses TF-IDF text similarity between books (description, genres, authors), further enhanced by rating and rating count weighting.
- **Genre Recommender:** Optionally find top books within a genre, ranked by popularity and ratings.
- **Rich UI:** Sleek, user-friendly interface built with Streamlit. Covers, authors, and details displayed for each suggested book.
- **Fast Preprocessing:** Efficient handling of large datasets (up to 50,000 books), robust cleaning, and artifact saving for quick startup.
- **No API Setup Required:** All covers and metadata handled locally; Google Books API is optional.

## Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation & Quick Start

1. **Install dependencies:**
pip install -r requirements.txt
3. **Run preprocessing (first-time setup):**
cd src
python preprocess.py
- Place your `book_data.csv` in the `src` folder before running.
3. **Start the app:**
  - Visit the local Streamlit URL shown in your terminal.

*See `start_app.txt` for concise instructions.*

### File Structure

| File                    | Purpose                                              |
|-------------------------|-----------------------------------------------------|
| `main.py`               | Streamlit UI (app entrypoint, handles user inputs)  |
| `preprocess.py`         | Data cleaning, text preprocessing, TF-IDF, saves models |
| `recommend.py`          | Recommendation logic and ranking                    |
| `google_books_utils.py` | (Optional) Book cover/description fetch via API     |
| `config.json`           | Place your Google Books API key here (if needed)    |
| `requirements.txt`      | Python library dependencies                         |
| `book_data.csv`         | Main data file with books information               |

### Data Requirements

Download the data from the site:https://www.kaggle.com/datasets/meetnaren/goodreads-best-books/data

The **preprocessing automatically skips incomplete rows, deduplicates by ISBN, normalizes ratings, and vectorizes text content**.

## Configuration

- **Google Books API (Optional):** For extra metadata, add your API key to `config.json` (see template). If no key is provided, all features still work using local data.

## Customization

- Modify `main.py` to change UI, weights, or number of recommendations.
- Add fields or enrich recommendations in `recommend.py`.
- Adapt cleaning or modeling in `preprocess.py`.

## Contributing

Issues, suggestions, and PRs are welcome. For major changes, open an issue first.


*Created with Streamlit, pandas, scikit-learn, NLTK, and joblib.*
