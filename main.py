import openai
from dotenv import load_dotenv
import pandas as pd
import ast
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import pickle
import time


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
# load the API Key
load_dotenv()
# Load the datasets
credits_df = pd.read_csv('tmdb_5000_credits.csv')
movies_df = pd.read_csv('tmdb_5000_movies.csv')



movies_df['runtime'].fillna(movies_df['runtime'].median(), inplace=True)
movies_df.dropna(subset=['overview','genres'],inplace=True)
movies_df = movies_df.merge(credits_df, left_on='id',right_on='movie_id',how='left')


# Extract Director//Writer
def extract_crew(crew, job):
    for member in crew:
        if member['job'] == job:
            return member['name']
    return None


movies_df['director'] = movies_df['crew'].apply(lambda x: extract_crew(ast.literal_eval(x),'Director'))
movies_df['director'].fillna('', inplace=True)



def preprocess_text(text):
    if not isinstance(text,str):
        return ""
    
    tokens = word_tokenize(text) #Tokenize the text

    stop_words = set(stopwords.words('english')) # Setting the stop words
    filtered_tokens = [w for w in tokens if not w.lower() in stop_words] # Filtering out the stop words

    lemmatizer = WordNetLemmatizer()
    lemmatized_text = ' '.join([lemmatizer.lemmatize(w) for w in filtered_tokens])

    return lemmatized_text


def flatten_list(data_str, limit=5):
    try:
        # Evaluate string as a list
        items = ast.literal_eval(data_str)
        names = [item['name'] for item in items[:limit]]
        return ' '.join(names)
    except:
        return data_str



movies_df.drop(['movie_id','crew'], axis=1, inplace=True)
movies_df['cast'] = movies_df['cast'].apply(lambda x: flatten_list(x,5))
print("Cast Pre-processing completed")

def load_and_preprocess(df):
    list_columns = ['genres','keywords']
    text_columns = ['original_title','overview','release_date','director','runtime','budget','cast']

    for col in list_columns:
        df[col] = df[col].apply(lambda x: flatten_list(x)).apply(lambda x: preprocess_text(x))
    
    for col in text_columns:
        df[col] = df[col].apply(lambda x: preprocess_text(x))

    df['combined_features'] = df[list_columns + text_columns].apply(lambda row: ' '.join(row.values.astype(str)),axis=1)

    return df


df_movies = load_and_preprocess(movies_df)
print("Pre-processing completed")



def generate_embeddings(text):
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response['data'][0]['embedding']

# Generate embeddings and store them in an array
embeddings = []
start_time = time.time()

for text in movies_df['combined_features']:
    embedding = generate_embeddings(text)
    embeddings.append(embedding)

print(f"Generated embeddings for {len(embeddings)} movies in {(time.time() - start_time) / 60:.2f} minutes")

# Keeping only the required columns
movies_df = movies_df[['id', 'original_title', 'combined_features']]

# Saving the embeddings and movies_df to a .pkl file
with open('movies_with_embeddings.pkl', 'wb') as file:
    pickle.dump((movies_df, embeddings), file)

print("Data has been preprocessed and saved to movies_with_embeddings.pkl")



