import streamlit as st
import pickle
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import html

st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# PATHS

BASE_DIR = Path(__file__).resolve().parent
MOVIE_DICT_FILE = BASE_DIR / "movie_dict.pkl"
SIMILARITY_FILE = BASE_DIR / "similarity.pkl"

# TMDB CONFIG

# Keep your existing TMDB key here.
# For a public GitHub repository, move this to Streamlit secrets.
TMDB_API_KEY = "8265bd1679663a7ea12ac168da84d2e"

# CUSTOM CSS

st.markdown(
    """
    <style>
        .stApp {
            background:
                radial-gradient(circle at 15% 15%, rgba(236,72,153,.12), transparent 30%),
                radial-gradient(circle at 85% 20%, rgba(59,130,246,.10), transparent 30%),
                linear-gradient(135deg,#050816 0%,#090d1d 48%,#03050d 100%);
            color: white;
        }

        .block-container {
            max-width: 1450px;
            padding-top: 1.5rem;
            padding-bottom: 3rem;
        }

        div[data-baseweb="select"] > div {
            background: #111827 !important;
            border: 1px solid #475569 !important;
            border-radius: 10px !important;
        }

        div[data-baseweb="select"] span {
            color: white !important;
        }

        .stButton > button {
            width: 100%;
            height: 52px;
            border: 0;
            border-radius: 12px;
            background: linear-gradient(90deg,#ec4899,#f97316);
            color: white;
            font-size: 16px;
            font-weight: 700;
            box-shadow: 0 8px 25px rgba(236,72,153,.25);
        }

        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 12px 30px rgba(236,72,153,.40);
        }

        header {
            background: transparent !important;
        }

        #MainMenu {
            visibility: hidden;
        }

        footer {
            visibility: hidden;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# LOAD DATA

@st.cache_resource
def load_data():
    if not MOVIE_DICT_FILE.exists():
        raise FileNotFoundError(
            f"movie_dict.pkl not found at: {MOVIE_DICT_FILE}"
        )

    if not SIMILARITY_FILE.exists():
        raise FileNotFoundError(
            f"similarity.pkl not found at: {SIMILARITY_FILE}"
        )

    with open(MOVIE_DICT_FILE, "rb") as f:
        movies_dict = pickle.load(f)

    movies_df = pd.DataFrame(movies_dict)

    with open(SIMILARITY_FILE, "rb") as f:
        similarity_matrix = pickle.load(f)

    return movies_df, similarity_matrix


try:
    movies, similarity = load_data()
except Exception as e:
    st.error("Could not load the recommendation model.")
    st.code(str(e))
    st.stop()


# TMDB POSTER

@st.cache_data(show_spinner=False)
def fetch_poster(movie_id):
    """Return a TMDB poster URL or None if unavailable."""

    if not TMDB_API_KEY or TMDB_API_KEY == "YOUR_TMDB_API_KEY":
        return None

    try:
        response = requests.get(
            f"https://api.themoviedb.org/3/movie/{int(movie_id)}",
            params={
                "api_key": TMDB_API_KEY,
                "language": "en-US",
            },
            timeout=5,
        )

        if response.status_code != 200:
            return None

        data = response.json()
        poster_path = data.get("poster_path")

        if not poster_path:
            return None

        return f"https://image.tmdb.org/t/p/w500{poster_path}"

    except (requests.RequestException, ValueError, TypeError):
        return None

# RECOMMENDATIONS

def recommend(movie_title):
    matches = movies.index[movies["title"] == movie_title].tolist()

    if not matches:
        return [], []

    movie_index = matches[0]
    distances = similarity[movie_index]

    top_movies = sorted(
        enumerate(distances),
        key=lambda x: x[1],
        reverse=True,
    )[1:6]

    names = []
    movie_ids = []

    for index, _score in top_movies:
        row = movies.iloc[index]

        names.append(str(row["title"]))

        # Your existing movie_dict.pkl uses movie_id.
        # Keep a safe fallback for datasets using "id".
        if "movie_id" in movies.columns:
            movie_ids.append(row["movie_id"])
        elif "id" in movies.columns:
            movie_ids.append(row["id"])
        else:
            movie_ids.append(None)

    # Fetch all five posters concurrently so the UI doesn't wait
    # through five sequential network requests.
    posters = [None] * len(movie_ids)

    valid_positions = [
        (position, movie_id)
        for position, movie_id in enumerate(movie_ids)
        if pd.notna(movie_id)
    ]

    if valid_positions:
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_map = {
                executor.submit(fetch_poster, movie_id): position
                for position, movie_id in valid_positions
            }

            for future in as_completed(future_map):
                position = future_map[future]
                try:
                    posters[position] = future.result()
                except Exception:
                    posters[position] = None

    return names, posters

# HERO

st.html(
    """
    <div style="
        text-align:center;
        padding:25px 10px 28px;
        color:white;
    ">
        <div style="font-size:55px;line-height:1;">🍿</div>

        <h1 style="
            margin:8px 0 5px;
            font-size:54px;
            font-weight:800;
            letter-spacing:-2px;
            color:white;
        ">
            Movie
            <span style="
                background:linear-gradient(90deg,#ec4899,#f97316);
                -webkit-background-clip:text;
                -webkit-text-fill-color:transparent;
            ">
                Recommendation
            </span>
            System
        </h1>

        <div style="
            font-size:20px;
            color:#cbd5e1;
            margin-top:8px;
        ">
            Discover Movies You'll Love
        </div>

        <div style="
            font-size:15px;
            color:#94a3b8;
            margin-top:8px;
        ">
            Get personalized movie recommendations based on content similarity
            using machine learning.
        </div>
    </div>
    """
)

# FEATURE CARDS

feature_html = """
<div style="
    display:flex;
    gap:18px;
    margin:5px 0 30px;
">
    <div style="
        flex:1;
        background:linear-gradient(145deg,rgba(20,29,55,.95),rgba(8,13,28,.95));
        border:1px solid rgba(148,163,184,.20);
        border-radius:16px;
        padding:20px;
        min-height:90px;
        box-shadow:0 10px 35px rgba(0,0,0,.25);
    ">
        <div style="font-size:30px;">✨</div>
        <div style="font-size:17px;font-weight:700;color:white;">
            Smart Recommendations
        </div>
        <div style="font-size:13px;color:#94a3b8;margin-top:5px;">
            Find similar movies you'll enjoy
        </div>
    </div>

    <div style="
        flex:1;
        background:linear-gradient(145deg,rgba(20,29,55,.95),rgba(8,13,28,.95));
        border:1px solid rgba(148,163,184,.20);
        border-radius:16px;
        padding:20px;
        min-height:90px;
        box-shadow:0 10px 35px rgba(0,0,0,.25);
    ">
        <div style="font-size:30px;">🎬</div>
        <div style="font-size:17px;font-weight:700;color:white;">
            Huge Movie Database
        </div>
        <div style="font-size:13px;color:#94a3b8;margin-top:5px;">
            Explore thousands of movies
        </div>
    </div>

    <div style="
        flex:1;
        background:linear-gradient(145deg,rgba(20,29,55,.95),rgba(8,13,28,.95));
        border:1px solid rgba(148,163,184,.20);
        border-radius:16px;
        padding:20px;
        min-height:90px;
        box-shadow:0 10px 35px rgba(0,0,0,.25);
    ">
        <div style="font-size:30px;">❤️</div>
        <div style="font-size:17px;font-weight:700;color:white;">
            Personalized for You
        </div>
        <div style="font-size:13px;color:#94a3b8;margin-top:5px;">
            Discover your next favorite
        </div>
    </div>
</div>
"""

st.html(feature_html)

# MOVIE SELECTION

st.html(
    """
    <div style="
        background:linear-gradient(145deg,rgba(15,23,42,.96),rgba(8,13,28,.96));
        border:1px solid rgba(100,116,139,.35);
        border-radius:18px 18px 0 0;
        padding:22px 25px 8px;
        margin-top:5px;
    ">
        <div style="
            color:#e2e8f0;
            font-size:17px;
            font-weight:600;
        ">
            🎞️ Select a movie
        </div>
    </div>
    """
)

selected_movie_name = st.selectbox(
    "Select a movie:",
    movies["title"].dropna().astype(str).values,
    label_visibility="collapsed",
)

# RECOMMEND BUTTON

if st.button("🚀  Recommend", use_container_width=True):

    with st.spinner("Finding movies you'll love..."):
        names, posters = recommend(selected_movie_name)

    if not names:
        st.error("Could not find this movie in the recommendation model.")
        st.stop()

    st.html(
        """
        <div style="margin:35px 0 18px;">
            <div style="
                font-size:34px;
                font-weight:800;
                color:white;
            ">
                🔥
                <span style="
                    background:linear-gradient(90deg,#ec4899,#f97316);
                    -webkit-background-clip:text;
                    -webkit-text-fill-color:transparent;
                ">
                    Top 5 Recommended
                </span>
                Movies
            </div>

            <div style="
                color:#94a3b8;
                font-size:14px;
                margin-top:5px;
            ">
                Movies similar to your selection
            </div>
        </div>
        """
    )

    cols = st.columns(5, gap="medium")

    for col, name, poster in zip(cols, names, posters):

        safe_name = html.escape(name)

        with col:

            if poster:
                st.html(
                    f"""
                    <div style="
                        background:linear-gradient(145deg,#111827,#070c19);
                        border:1px solid rgba(100,116,139,.30);
                        border-radius:15px;
                        overflow:hidden;
                        box-shadow:0 12px 35px rgba(0,0,0,.35);
                    ">
                        <img
                            src="{poster}"
                            style="
                                width:100%;
                                height:320px;
                                object-fit:cover;
                                display:block;
                            "
                        >

                        <div style="
                            color:white;
                            font-size:16px;
                            font-weight:700;
                            text-align:center;
                            padding:14px 8px 17px;
                            min-height:50px;
                        ">
                            {safe_name}
                        </div>
                    </div>
                    """
                )
            else:
                st.html(
                    f"""
                    <div style="
                        background:linear-gradient(145deg,#111827,#070c19);
                        border:1px solid rgba(100,116,139,.30);
                        border-radius:15px;
                        overflow:hidden;
                        box-shadow:0 12px 35px rgba(0,0,0,.35);
                    ">
                        <div style="
                            height:320px;
                            display:flex;
                            align-items:center;
                            justify-content:center;
                            background:linear-gradient(135deg,#111827,#1e293b);
                            font-size:55px;
                        ">
                            🎬
                        </div>

                        <div style="
                            color:white;
                            font-size:16px;
                            font-weight:700;
                            text-align:center;
                            padding:14px 8px 17px;
                            min-height:50px;
                        ">
                            {safe_name}
                        </div>
                    </div>
                    """
                )


# FOOTER

st.html(
    """
    <div style="
        text-align:center;
        margin-top:55px;
        padding:25px 10px;
        border-top:1px solid rgba(148,163,184,.15);
        color:#64748b;
        font-size:14px;
    ">
        <div style="font-size:16px;color:#cbd5e1;">
            “Good movies stay with you.”
        </div>

        <div style="margin-top:16px;">
            Built with
            <span style="color:#ec4899;">♥</span>
            using Python, Machine Learning & Streamlit
        </div>
    </div>
    """
)
