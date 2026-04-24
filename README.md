# Genre Similarity movie recommendation system 


## Overview 
The movie recommender system works very quickly. You just add the title of a film to the function `get_recommendations_faiss("`**Name of movie**`")` and it will list out the top 10 most similar movies in terms of genre, rating and the number of votes. It works like a charm really. 


## How to Use 
1. Clone this repo
2. Create a virtual environment
3. Install all packages - `pip install requirements.txt`
4. Run all cells 
5. Afterwards, on the final cell, you can just use the `get_recommendation_faiss` function to inference the system to show you the movies most similar to what you are looking for. Best damn project I have ever made so far. 

## Contacts 
Simply message me on `+254 708 445 839` on whatsapp in case you run into any problems. 


## Technical Specs 
The model uses imdb's public datasets available on this link `https://developer.imdb.com/non-commercial-datasets/` and is trained using Facebook AI Similarity Search algorithm (`FAISS`), using the movies genres, average rating, and the number of votes scaled on a logarithmic scale to prevent skewness in the data. Vectorization was done using sci-kit learn's `TfidfTransformer` and a Minimax scaler was used in the data preparation phase to normalize the data, and Yadi, Yadi, Yada... Other than that, all you need to know is that it works like a charm... 

## Future plans
You just need to wait and see... 