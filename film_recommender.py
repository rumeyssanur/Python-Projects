import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

film = pd.read_csv("film.csv")

count = CountVectorizer(tokenizer=lambda x : x.split(','))
count_matrix = count.fit_transform(film['genres'])

cosine_sim = cosine_similarity(count_matrix,count_matrix)

def recommend(film_title):
    idx = film[film['title']== film_title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key =lambda x :x[1], reverse=True)
    sim_scores = sim_scores[1:6]
    film_indices = [i[0] for i in sim_scores]
    print("Önerilen Filmler:")
    print(film['title'].iloc[film_indices])

recommend("The Godfather")







