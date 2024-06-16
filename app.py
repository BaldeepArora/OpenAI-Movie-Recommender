import streamlit as st
import numpy as np
import openai
import requests
import pickle
import time
from PIL import Image, ImageDraw
from io import BytesIO
from dotenv import load_dotenv
import os

load_dotenv()
tmdb_api_key= os.getenv('tmdb_api_key')

# Measuring the time it takes to load the dataset
start_time = time.time()
# Loading the preprocessed data and embeddings
@st.cache_data
def load_data():
    with open('movies_with_embeddings.pkl', 'rb') as file:
        movies_df, embeddings = pickle.load(file)
    return movies_df.copy(), embeddings.copy()

movies_df, embeddings = load_data()

end_time = time.time()
time_taken = end_time - start_time
st.write(f"Time taken to load the dataset: {time_taken:.2f} seconds")


# Function to get movie posters
def get_movie_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={tmdb_api_key}&language=en-US"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        poster_path = data.get('poster_path')
        if poster_path:
            full_path = f"https://image.tmdb.org/t/p/w500{poster_path}"
            return full_path
    return None  # Will return None if the poster is not found

# Function to create an outlined placeholder
def create_placeholder_image(width, height):
    placeholder = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(placeholder)
    draw.rectangle([(0, 0), (width-1, height-1)], outline="black", width=5)
    return placeholder

# Function to get recommendations
def get_recommendations(prompt, n=10):
    response = openai.Embedding.create(
        input=prompt,
        model="text-embedding-ada-002"
    )
    prompt_embedding = np.array(response['data'][0]['embedding'])

    similarities = [np.dot(embedding, prompt_embedding) for embedding in embeddings]
    movies_df['similarity'] = similarities
    recommended_movies = movies_df.sort_values(by='similarity', ascending=False).head(n)
    return recommended_movies

# Streamlit app
st.title('Movie Recommendation System with OpenAI Embeddings')

user_input = st.text_input('Hello there! What would you like to watch?:')


if user_input:
    recommendations = get_recommendations(user_input)
    st.write('Top 10 recommended movies:')

    # Split recommendations into two rows
    first_row_recommendations = recommendations[:5]
    second_row_recommendations = recommendations[5:]

    # First row of recommendations
    first_row_cols = st.columns(5)
    for index, (row_index, row) in enumerate(first_row_recommendations.iterrows()):
        with first_row_cols[index]:
            poster_url = get_movie_poster(row['id'])
            if poster_url:
                response = requests.get(poster_url)
                try:
                    img = Image.open(BytesIO(response.content))
                    st.image(img, width=150)
                except Exception as e:
                    st.image(create_placeholder_image(150, 225), width=150)  # Placeholder size is 150x225
                    st.write("Poster not available.")
            else:
                st.image(create_placeholder_image(150, 225), width=150)  # Placeholder size is 150x225
                st.write("Poster not available.")
            st.markdown(f"<div style='text-align: center; font-size: large;'>{row['original_title']}</div>", unsafe_allow_html=True)

    # Second row of recommendations
    second_row_cols = st.columns(5)
    for index, (row_index, row) in enumerate(second_row_recommendations.iterrows()):
        with second_row_cols[index]:
            poster_url = get_movie_poster(row['id'])
            if poster_url:
                response = requests.get(poster_url)
                try:
                    img = Image.open(BytesIO(response.content))
                    st.image(img, width=150)
                except Exception as e:
                    st.image(create_placeholder_image(150, 225), width=150)  # Placeholder size is 150x225
                    st.write("Poster not available.")
            else:
                st.image(create_placeholder_image(150, 225), width=150)  # Placeholder size is 150x225
                st.write("Poster not available.")
            st.markdown(f"<div style='text-align: center; font-size: large;'>{row['original_title']}</div>", unsafe_allow_html=True)