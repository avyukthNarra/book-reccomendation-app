# main.py

import json
import streamlit as st
from recommend import df, recommend_books

# Optional Google Books API key (not used for covers anymore)
try:
    config = json.load(open("config.json"))
    GOOGLE_BOOKS_API_KEY = config.get("GOOGLE_BOOKS_API_KEY", None)
except Exception:
    GOOGLE_BOOKS_API_KEY = None

# Collapse and hide sidebar
st.set_page_config(
    page_title="Book Recommender",
    page_icon="📚",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Permanently hide the sidebar and its toggle
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {display: none;}
    [data-testid="collapsedControl"] {display: none;}
    </style>
    """,
    unsafe_allow_html=True
)

st.title("📚 Book Recommender")

# Select a book (no settings panel)
book_list = sorted(df['book_title'].dropna().unique())
selected_book = st.selectbox("📖 Select a book:", book_list)

# Fixed settings: top-N and weights (no user control)
TOP_N = 5
RATING_WEIGHT = 0.3
COUNT_WEIGHT = 0.2

if st.button("🚀 Recommend Similar Books"):
    with st.spinner("Finding similar books..."):
        recommendations = recommend_books(
            selected_book,
            top_n=TOP_N,
            rating_weight=RATING_WEIGHT,
            count_weight=COUNT_WEIGHT
        )

        if recommendations is None or recommendations.empty:
            st.warning("Sorry, no recommendations found.")
        else:
            st.success(f"Top {len(recommendations)} similar books:")
            for idx, (_, row) in enumerate(recommendations.iterrows(), 1):
                book_title = row['book_title']
                authors = row['book_authors']
                genres = row['genres']
                rating = row['book_rating']
                rating_count = row['book_rating_count']
                similarity_score = row['similarity_score']
                img_url = str(row.get('image_url', '') or '').strip()

                with st.container():
                    st.markdown("---")
                    col1, col2 = st.columns([1, 3])

                    with col1:
                        if img_url:
                            st.image(img_url, width=100)
                        else:
                            st.write("📚 No Cover")

                    with col2:
                        st.markdown(f"### {idx}. {book_title}")
                        st.markdown(f"**Authors:** {authors}")
                        st.markdown(f"**Genres:** {genres}")
                        st.markdown(f"**Rating:** ⭐ {rating:.1f} ({int(rating_count):,} ratings)")
                        st.markdown(f"**Similarity Score:** {similarity_score:.3f}")
