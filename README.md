# Genre Similarity movie recommendation system 


## Overview 
The movie recommender system works very quickly. You just add the title of a film as an argument to `python main.py "Name of Movie"` and it will list out the top 10 most similar movies in terms of genre, rating and the number of votes. It works like a charm really. 


## How to Use 
1. Clone this repo
2. Create a virtual environment
3. Install all packages - `pip install requirements.txt`
4. Run this command `python main.py "name of movie"`. The second argument should be the name of the movie or television show to get similar recommendations. 
5. If you don't find the movie in the index, google the movie's or series' title and paste it. Typos are still not handled in this version but will handled in the next. 

## Contacts 
Simply message me on `+254 708 445 839` on whatsapp in case you run into any problems. 


## Technical Specs 
The model uses imdb's public datasets available on this link `https://developer.imdb.com/non-commercial-datasets/` and is trained using Facebook AI Similarity Search algorithm (`FAISS`), using the movies genres, average rating, and the number of votes scaled on a logarithmic scale to prevent skewness in the data. Vectorization was done using sci-kit learn's `TfidfTransformer` and a Minimax scaler was used in the data preparation phase to normalize the data, and Yadi, Yadi, Yada... Other than that, all you need to know is that it works like a charm... 

## Future plans
You just need to wait and see... 
