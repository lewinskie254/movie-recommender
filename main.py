import pandas as pd
import numpy as np
import joblib
import faiss
import re
import sys
import os

# --- PART 1: ASSET LOADING ---
base_path = os.path.dirname(os.path.abspath(__file__))

def load_assets():
    try:
        idx = faiss.read_index(os.path.join(base_path, "movie_faiss.index"))
        m_map = joblib.load(os.path.join(base_path, 'indices_map.pkl'))
        meta = joblib.load(os.path.join(base_path, 'metadata.pkl'))
        return idx, m_map, meta
    except Exception as e:
        print(f"❌ Error: Could not find model files. {e}")
        sys.exit(1)

# Load assets globally so they are ready for the function
index, indices_map, metadata = load_assets()

# --- PART 2: TITLE CASING LOGIC ---
def professional_title_case(text):
    lowercase_exceptions = {'a', 'an', 'and', 'as', 'at', 'but', 'by', 'for', 
                            'if', 'in', 'is', 'nor', 'of', 'on', 'or', 'so', 
                            'the', 'to', 'up', 'yet', 'it', 'with'}
    
    def replace_func(match):
        word = match.group(0)
        start_index = match.start()
        
        if word.lower() == "'s": return word.lower()
        
        preceding_text = text[:start_index].rstrip()
        follows_colon = preceding_text.endswith(':')

        if "." in word:
            if word.lower().rstrip('.') == 'vs':
                return "vs." if (start_index != 0 and not follows_colon) else "Vs."
            return word.upper()

        if word.isupper() and len(word) > 1: return word
        
        if start_index == 0 or follows_colon: return word.capitalize()
        
        if word.lower() in lowercase_exceptions: return word.lower()

        return word.capitalize()

    return re.sub(r"[\w\.\']+", replace_func, text)

# --- PART 3: RECOMMENDATION LOGIC ---
def get_recommendations_faiss(title, top_n=10):
    title = title.strip()
    title = professional_title_case(title)
    
    # 1. Check if it exists
    if title not in indices_map:
        return f"Error: '{title}' not found in index."
    
    # 2. ACTUALLY GRAB THE VALUE (This was missing!)
    movie_idx = indices_map[title]

    # 3. Handle duplicates (If movie_idx is a Series/List)
    if isinstance(movie_idx, (pd.Series, pd.Index, np.ndarray)):
        movie_idx = movie_idx.iloc[0] if hasattr(movie_idx, 'iloc') else movie_idx[0]
    
    # 4. Math and Search
    query_vector = index.reconstruct(int(movie_idx)).reshape(1, -1)
    D, I = index.search(query_vector, top_n + 1)

    # 5. Build Results
    recommended_indices = I[0][1:]
    similarity_scores = D[0][1:]

    results = metadata.iloc[recommended_indices].copy()
    results['match_score'] = [f"{round(s * 100, 2)}%" for s in similarity_scores]

    return results.sort_values(by=['averageRating', 'numVotes'], ascending=False)[
        ['primaryTitle', 'startYear', 'titleType', 'numVotes', 'genres', 'averageRating', 'match_score']
    ]



# --- PART 4: EXECUTION ---
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\nUsage: python main.py \"Movie Name\"")
        sys.exit(1)

    input_title = " ".join(sys.argv[1:])
    print(f"\n🔍 Searching for: '{input_title}'...")
    
    output = get_recommendations_faiss(input_title)

    if isinstance(output, str):
        print(output)
    else:
        print("\n--- Top Recommendations ---")
        # Replace the print line in your main block with this:
        print(output.to_json(orient='records', indent=4))        
        print("\n")