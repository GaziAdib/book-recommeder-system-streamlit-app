import streamlit as st
import pickle
import numpy as np

# Load pickled files
model = pickle.load(open("./artifacts/model.pkl", "rb"))
book_names = pickle.load(open("./artifacts/book_names.pkl", "rb"))
final_rating = pickle.load(open("./artifacts/final_rating.pkl", "rb"))
book_pivot = pickle.load(open("./artifacts/book_pivot.pkl", "rb"))

def fetch_book_info(book_title):
    """Fetch image URLs based on book title"""
    ids = np.where(final_rating['title'] == book_title)[0][0]
    url = final_rating.iloc[ids]['image_url']
    return url

def recommend_book(book_name):
    """Recommend similar books"""
    index = np.where(book_pivot.index == book_name)[0][0]
    distance, suggestion = model.kneighbors(book_pivot.iloc[index, :].values.reshape(1, -1), n_neighbors=6)

    recommended_books = []
    for i in suggestion[0]:
        title = book_pivot.index[i]
        if title != book_name:
            recommended_books.append({
                'title': title,
                'image_url': fetch_book_info(title)
            })
    return recommended_books

# Streamlit UI
st.set_page_config(page_title="📚 Book Recommender", layout="wide")
st.title("📖 Book Recommendation System")
st.write("Select a book you like, and we’ll recommend similar ones.")

# Dropdown
selected_book = st.selectbox("Choose a book", book_names)

# Button to recommend
if st.button("Show Recommendations"):
    with st.spinner("Finding similar books..."):
        recommendations = recommend_book(selected_book)
        
        cols = st.columns(5)
        for idx, col in enumerate(cols[:len(recommendations)]):
            col.image(recommendations[idx]['image_url'], use_container_width=True)
            col.markdown(f"**{recommendations[idx]['title']}**")
