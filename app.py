import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.metrics.pairwise import linear_kernel

st.set_page_config(
    page_title="Book Recommender",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_css():
    try:
        with open('assets/style.css') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except:
        pass

load_css()

@st.cache_resource
def load_resources():
    try:
        books_df = pd.read_csv('data/books.csv')
        ratings_df = pd.read_csv('data/ratings.csv')
    except FileNotFoundError:
        st.error("CRITICAL ERROR: 'books.csv' or 'ratings.csv' not found in 'data/' folder.")
        st.stop()
    
    books_df['original_title'] = books_df['original_title'].fillna('Untitled')
    books_df['authors'] = books_df['authors'].fillna('Unknown')
    
    try:
        with open('models/svd_model.pkl', 'rb') as f:
            svd_model = pickle.load(f)
            
        with open('models/cbf_model.pkl', 'rb') as f:
            cbf_data = pickle.load(f)
    except FileNotFoundError:
        st.error("CRITICAL ERROR: Model files not found in 'models/' folder.")
        st.stop()
    
    return books_df, ratings_df, svd_model, cbf_data

books, ratings, svd, cbf_data = load_resources()

tfidf_matrix = cbf_data['tfidf_matrix']
book_id_to_idx = cbf_data['book_id_to_idx']
idx_to_book_id = cbf_data['idx_to_book_id']

def get_recommendations(user_id=None, seed_book_id=None, top_n=12):
    """
    Hybrid Recommendation System:
    - For existing users: α = 0.6 (60% SVD collaborative, 40% content-based)
    - For new users: α = 0.0 (100% content-based, cold start mode)
    """
    
    if user_id is None and seed_book_id is not None:
        if seed_book_id not in book_id_to_idx: 
            return pd.DataFrame(), "content_only", 0.0
        
        idx = book_id_to_idx[seed_book_id]
        sim_scores = linear_kernel(tfidf_matrix[idx], tfidf_matrix).flatten()
        top_indices = sim_scores.argsort()[::-1][1:top_n+1]
        rec_ids = [idx_to_book_id[i] for i in top_indices if idx_to_book_id[i] in books['id'].values]
        result_df = books[books['id'].isin(rec_ids)].head(top_n)
        return result_df, "content_only", 0.0

    user_history = ratings[ratings['user_id'] == user_id]
    is_existing = not user_history.empty
    
    if is_existing:
        alpha = 0.6 
        rated_books = user_history['book_id'].values
        rated_indices = [book_id_to_idx[bid] for bid in rated_books if bid in book_id_to_idx]
        
        if rated_indices:
            user_prof = np.asarray(np.mean(tfidf_matrix[rated_indices], axis=0)).reshape(1, -1)
            cbf_sim = linear_kernel(user_prof, tfidf_matrix).flatten()
        else:
            cbf_sim = np.zeros(tfidf_matrix.shape[0])
            
        candidates = [b for b in book_id_to_idx.keys() if b not in rated_books]
        mode = "hybrid"
    else:
        alpha = 0.0 
        if not seed_book_id:
            result_df = books.sort_values('average_rating', ascending=False).head(top_n)
            return result_df, "cold_start_popular", alpha
        
        if seed_book_id in book_id_to_idx:
            idx = book_id_to_idx[seed_book_id]
            cbf_sim = linear_kernel(tfidf_matrix[idx], tfidf_matrix).flatten()
        else:
            result_df = books.sort_values('average_rating', ascending=False).head(top_n)
            return result_df, "cold_start_popular", alpha

        candidates = [b for b in book_id_to_idx.keys() if b != seed_book_id]
        mode = "cold_start_content"

    top_cbf_indices = cbf_sim.argsort()[::-1][:500] 
    candidate_subset = [idx_to_book_id[i] for i in top_cbf_indices if idx_to_book_id[i] in candidates]
    
    results = []
    for bid in candidate_subset:
        idx = book_id_to_idx[bid]
        cbf_score = 1.0 + (cbf_sim[idx] * 4.0) 
        
        if is_existing:
            try:
                svd_score = svd.predict(user_id, bid).est
            except:
                svd_score = 3.0 
        else:
            svd_score = 0 
            
        final_score = (alpha * svd_score) + ((1 - alpha) * cbf_score)
        results.append((bid, final_score))
        
    results.sort(key=lambda x: x[1], reverse=True)
    top_ids = [x[0] for x in results[:top_n]]
    result_df = books[books['id'].isin(top_ids)]
    return result_df, mode, alpha

def render_book_card(row, button_key, button_text="Details"):
    """Renders a single book card with consistent height using custom HTML"""
    
    title = row['original_title']
    if len(title) > 60: 
        title = title[:57] + "..."
    
    card_html = f"""
    <div class="book-card">
        <div class="book-image-container">
            <img src="{row['image_url']}" alt="{title}">
        </div>
        <div class="book-title">
            {title}
        </div>
    </div>
    """
    
    st.markdown(card_html, unsafe_allow_html=True)
    
    st.markdown('<div class="book-button">', unsafe_allow_html=True)
    clicked = st.button(button_text, key=button_key, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    return clicked

if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'selected_book' not in st.session_state:
    st.session_state.selected_book = None
if 'user_id' not in st.session_state:
    st.session_state.user_id = None
if 'seed_book' not in st.session_state:
    st.session_state.seed_book = None
if 'rec_mode' not in st.session_state:
    st.session_state.rec_mode = None
if 'alpha_value' not in st.session_state:
    st.session_state.alpha_value = None

def go_to_book(book_id):
    st.session_state.selected_book = book_id
    st.session_state.page = 'details'

def go_home():
    st.session_state.page = 'home'
    st.session_state.selected_book = None

def logout():
    st.session_state.authenticated = False
    st.session_state.user_id = None
    st.session_state.seed_book = None
    st.session_state.page = 'home'

if not st.session_state.authenticated:
    st.markdown("<h1 style='text-align: center;'>🔐 Book Recommender System</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: #666;'>Please authenticate to continue</h3>", unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("### 👤 User Login")
        
        unique_users = sorted(ratings['user_id'].unique())
        
        with st.expander("🔹 Login as Existing User", expanded=True):
            st.info(f"📊 Total users in system: {len(unique_users)}")
            selected_user = st.selectbox(
                "Select your User ID:",
                options=unique_users,
                index=0,
                help="Choose from existing users in the database"
            )
            
            if st.button("🚀 Login", type="primary", use_container_width=True):
                st.session_state.user_id = selected_user
                st.session_state.authenticated = True
                st.success(f"✅ Welcome back, User {selected_user}!")
                st.balloons()
                st.rerun()
        
        st.markdown("---")
        
        with st.expander("🔹 Continue as New User"):
            st.warning("⚠️ New users will receive recommendations based on popular books or seed preferences")
            new_user_id = st.number_input(
                "Enter a new User ID:",
                min_value=max(unique_users) + 1,
                value=max(unique_users) + 1,
                help="Choose any ID above existing users"
            )
            
            if st.button("🆕 Create & Login", use_container_width=True):
                st.session_state.user_id = new_user_id
                st.session_state.authenticated = True
                st.success(f"✅ Welcome, New User {new_user_id}!")
                st.info("💡 Tip: Select a favorite book in the sidebar to get personalized recommendations!")
                st.rerun()
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.caption("🔒 Secure authentication • 📚 Powered by Hybrid Recommendation Engine")

else:
    with st.sidebar:
        st.markdown("## 👤 User Profile")
        st.markdown("---")
        
        st.success(f"**Current User:** {st.session_state.user_id}")
        
        user_exists = not ratings[ratings['user_id'] == st.session_state.user_id].empty
        
        st.markdown("### 🎯 Recommendation Mode")
        if user_exists:
            num_ratings = len(ratings[ratings['user_id'] == st.session_state.user_id])
            st.info(f"""
            **Mode:** Hybrid\n
            **Algorithm:** Weighted Ensemble\n
            **Alpha (α):** 0.6\n
            - 60% Collaborative (SVD)\n
            - 40% Content-Based\n\n
            **Your History:** {num_ratings} books rated
            """)
        else:
            st.warning(f"""
            **Mode:** Cold Start\n
            **Algorithm:** Content-Based\n
            **Alpha (α):** 0.0\n
            - 0% Collaborative\n
            - 100% Content-Based\n\n
            **Status:** New User (No history)
            """)
        
        st.markdown("---")
        
        if not user_exists:
            st.markdown("### 📖 Personalize Your Experience")
            st.write("Pick a favorite book to get better recommendations:")
            
            search_query = st.text_input("Search book title...", key="sidebar_search")
            if search_query:
                matches = books[books['original_title'].str.contains(search_query, case=False, na=False)]
                if not matches.empty:
                    option = st.selectbox("Select a book:", matches['original_title'].values)
                    if st.button("❤️ Set as Favorite", use_container_width=True):
                        selected_id = matches[matches['original_title'] == option]['id'].values[0]
                        st.session_state.seed_book = selected_id
                        st.toast(f"✅ Preference saved: {option}")
                        st.rerun()
            
            if st.session_state.seed_book:
                seed_book_info = books[books['id'] == st.session_state.seed_book].iloc[0]
                st.success(f"Current favorite: **{seed_book_info['original_title']}**")
        
        st.markdown("---")
        
        if st.button("🚪 Logout", use_container_width=True):
            logout()
            st.rerun()

    if st.session_state.page == 'home':
        st.markdown("<h1>📚 Intelligent Book Recommender</h1>", unsafe_allow_html=True)
        
        user_exists = not ratings[ratings['user_id'] == st.session_state.user_id].empty
        col_info1, col_info2, col_info3 = st.columns(3)
        
        with col_info1:
            st.metric("User ID", st.session_state.user_id)
        with col_info2:
            mode_display = "Hybrid (α=0.6)" if user_exists else "Cold Start (α=0.0)"
            st.metric("Mode", mode_display)
        with col_info3:
            if user_exists:
                num_ratings = len(ratings[ratings['user_id'] == st.session_state.user_id])
                st.metric("Books Rated", num_ratings)
            else:
                st.metric("Status", "New User")
        
        st.markdown("---")
        
        c1, c2, c3 = st.columns([1, 6, 1])
        with c2:
            search = st.text_input("", placeholder="🔍 Search for a book title (e.g. Harry Potter)...")

        st.markdown("<br>", unsafe_allow_html=True)

        if search:
            st.subheader("🔎 Search Results")
            results = books[books['original_title'].str.contains(search, case=False, na=False)].head(8)
            
            if results.empty:
                st.warning("No books found.")
            else:
                cols = st.columns(4)
                for i, (idx, row) in enumerate(results.iterrows()):
                    with cols[i % 4]:
                        if render_book_card(row, f"btn_{row['id']}", "View Details"):
                            go_to_book(row['id'])
                            st.rerun()
        
        else:
            if user_exists:
                st.subheader(f"✨ Personalized Recommendations (Hybrid α=0.6)")
                st.caption("60% based on your rating history + 40% based on book content similarity")
            elif st.session_state.seed_book:
                st.subheader("✨ Based on Your Favorite Book (Content-Based)")
                st.caption("100% based on content similarity to your selected book")
            else:
                st.subheader("🔥 Top Rated Books for You")
                st.caption("Popular books to help you get started")
                
            recs, mode, alpha = get_recommendations(
                user_id=st.session_state.user_id, 
                seed_book_id=st.session_state.seed_book
            )
            
            st.session_state.rec_mode = mode
            st.session_state.alpha_value = alpha
            
            if not recs.empty:
                cols = st.columns(6)
                for i, (idx, row) in enumerate(recs.iterrows()):
                    with cols[i % 6]:
                        if render_book_card(row, f"rec_{row['id']}", "Details"):
                            go_to_book(row['id'])
                            st.rerun()
            else:
                st.warning("No recommendations available.")

    elif st.session_state.page == 'details':
        b_id = st.session_state.selected_book
        book = books[books['id'] == b_id].iloc[0]
        
        if st.button("⬅️ Back to Home"):
            go_home()
            st.rerun()
            
        st.markdown("---")
        
        c1, c2 = st.columns([1, 2], gap="large")
        
        with c1:
            st.image(book['image_url'], width=350)
            
        with c2:
            st.markdown(f"<h1 style='text-align: left; color: #4b6cb7;'>{book['original_title']}</h1>", unsafe_allow_html=True)
            st.markdown(f"### by {book['authors']}")
            
            c_a, c_b = st.columns(2)
            with c_a:
                st.metric("Year", int(book['original_publication_year']) if pd.notnull(book['original_publication_year']) else 'N/A')
            with c_b:
                st.metric("Rating", f"{book['average_rating']} ⭐")
                
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("### 📚 Similar Books You Might Like")
        
        similar_books, _, _ = get_recommendations(user_id=None, seed_book_id=b_id, top_n=6)
        
        if not similar_books.empty:
            cols = st.columns(6)
            for i, (idx, row) in enumerate(similar_books.iterrows()):
                with cols[i % 6]:
                    if render_book_card(row, f"sim_{row['id']}", "View"):
                        go_to_book(row['id'])
                        st.rerun()