import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import linear_kernel
import joblib
import faiss
from sklearn.feature_extraction.text import TfidfTransformer
import re


index = faiss.read_index("movie_faiss.index")
indices_map = joblib.load('indices_map.pkl')
metadata = joblib.load('metadata.pkl')

def professional_title_case(text):
    lowercase_exceptions = {'a', 'an', 'and', 'as', 'at', 'but', 'by', 'for', 
                            'if', 'in', 'is', 'nor', 'of', 'on', 'or', 'so', 
                            'the', 'to', 'up', 'yet', 'it', 'with'}
    
    def replace_func(match):
        word = match.group(0)
        start_index = match.start()
        
        # 1. Handle Possessives (e.g., 's in Ocean's)
        if word.lower() == "'s":
            return word.lower()

        # 2. Check if this word follows a colon (e.g., Hachi: A Dog's Tale)
        # We look at the substring before the current word
        preceding_text = text[:start_index].rstrip()
        follows_colon = preceding_text.endswith(':')

        # 3. Protect Initials with periods
        if "." in word:
            if word.lower().rstrip('.') == 'vs':
                return "vs." if (start_index != 0 and not follows_colon) else "Vs."
            return word.upper()

        # 4. Protect All-Caps Acronyms
        if word.isupper() and len(word) > 1:
            return word
        
        # 5. Always capitalize the FIRST word OR word AFTER A COLON
        if start_index == 0 or follows_colon:
            return word.capitalize()
        
        # 6. Apply lowercase exceptions
        if word.lower() in lowercase_exceptions:
            return word.lower()

        # 7. Default to standard Title Case
        return word.capitalize()

    return re.sub(r"[\w\.\']+", replace_func, text)

def get_recommendations_faiss(title, top_n=10):
    # 1. Clean and Verify Title
    title = title.strip()
    title = professional_title_case(title)
    if title not in indices_map:
        return f"Error: '{title}' not found in index."

    # 2. Retrieve the Index Position
    movie_idx = indices_map[title]

    # FIX: Handle duplicates correctly
    if isinstance(movie_idx, (pd.Series, np.ndarray)):
        movie_idx = movie_idx.iloc[0]
    # If it's just a single integer/number, movie_idx is already what we need!
    
    # 3. Reconstruct the Vector
    query_vector = index.reconstruct(int(movie_idx)).reshape(1, -1)

    # 4. Search
    D, I = index.search(query_vector, top_n + 1)

    # 5. Process
    recommended_indices = I[0][1:]
    similarity_scores = D[0][1:]

    # 6. Build and Sort Results
    results = metadata.iloc[recommended_indices].copy()
    results['match_score_raw'] = similarity_scores # Keep numeric for sorting
    results['match_score'] = [f"{round(s * 100, 2)}%" for s in similarity_scores]

    # Use sort_values instead of orderBy
    return results.sort_values(by=['averageRating', 'startYear', 'numVotes', 'titleType'], ascending=False)[
        ['primaryTitle', 'startYear', 'numVotes', 'genres', 'titleType', 'averageRating', 'match_score']
    ]


print(get_recommendations_faiss("hell or high water", 20))
