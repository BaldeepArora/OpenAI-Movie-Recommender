# OpenAI Movie Recommender

OpenAI Movie Recommender is an intelligent movie recommendation application that utilizes OpenAI's text embeddings and The Movie Database (TMDB) API. The app uses Streamlit for its frontend interface, providing an interactive and user-friendly experience.

## Features

- **Personalized Recommendations**: Get tailored movie recommendations based on your input using advanced machine learning models.
- **Rich Media Content**: Display movie posters alongside the recommendations, enhancing the user experience.
- **High-Quality Embeddings**: Utilize OpenAI's text-embedding-ada-002 model to generate high-quality embeddings for accurate recommendations.
- **Interactive Frontend**: Powered by Streamlit, the app offers a responsive and interactive user interface.
- **API Integration**: Seamlessly fetch movie data and posters using the TMDB API.

## Demo

You can see the app in action in the video below:
<video width="600" controls>
  <source src="demo.mp4" type="video/webm">
</video>

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/your-username/openai-movie-recommender.git
    ```

2. Navigate to the project directory:

    ```bash
    cd openai-movie-recommender
    ```

3. Create a virtual environment:

    ```bash
    python -m venv venv
    ```

4. Activate the virtual environment:

    - On Windows:

      ```bash
      venv\Scripts\activate
      ```

    - On macOS and Linux:

      ```bash
      source venv/bin/activate
      ```

5. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

6. Add your OpenAI and TMDB API keys to a `.env` file:

    ```env
    OPENAI_API_KEY=your-openai-api-key
    TMDB_API_KEY=your-tmdb-api-key
    ```

## Usage

1. Run the Streamlit app:

    ```bash
    streamlit run app.py
    ```

2. Open your web browser and go to `http://localhost:8501` to use the app.

## Deployed Version

You can also access the deployed version of this app on Streamlit Cloud:

[Deployed App](https://share.streamlit.io/your-username/movie-recommendation-app/main/app.py)

## Files

- `app.py`: The main Streamlit app script.
- `main.py`: Script to preprocess data and generate embeddings.
- `main.ipynb`: Jupyter notebook for data preprocessing and embedding generation.
- `.env`: File containing API keys (not included in the repository).
- `.gitignore`: File specifying files to be ignored by Git.
- `tmdb_5000_credits.csv` and `tmdb_5000_movies.csv`: Datasets used for the project.
- `movies_with_embeddings.pkl`: Preprocessed data with embeddings.

<video width="600" controls>
  <source src="demo.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>
