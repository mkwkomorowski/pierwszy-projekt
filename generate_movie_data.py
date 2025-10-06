import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Ustawiamy seed dla powtarzalności wyników
np.random.seed(42)
random.seed(42)

# Lista przykładowych filmów z gatunkami
movies_data = [
    {"title": "Avengers: Endgame", "genre": "Action", "year": 2019, "director": "Russo Brothers"},
    {"title": "The Dark Knight", "genre": "Action", "year": 2008, "director": "Christopher Nolan"},
    {"title": "Pulp Fiction", "genre": "Crime", "year": 1994, "director": "Quentin Tarantino"},
    {"title": "The Shawshank Redemption", "genre": "Drama", "year": 1994, "director": "Frank Darabont"},
    {"title": "Forrest Gump", "genre": "Drama", "year": 1994, "director": "Robert Zemeckis"},
    {"title": "The Matrix", "genre": "Sci-Fi", "year": 1999, "director": "Wachowski Sisters"},
    {"title": "Inception", "genre": "Sci-Fi", "year": 2010, "director": "Christopher Nolan"},
    {"title": "Interstellar", "genre": "Sci-Fi", "year": 2014, "director": "Christopher Nolan"},
    {"title": "The Godfather", "genre": "Crime", "year": 1972, "director": "Francis Ford Coppola"},
    {"title": "Goodfellas", "genre": "Crime", "year": 1990, "director": "Martin Scorsese"},
    {"title": "Titanic", "genre": "Romance", "year": 1997, "director": "James Cameron"},
    {"title": "The Notebook", "genre": "Romance", "year": 2004, "director": "Nick Cassavetes"},
    {"title": "Casablanca", "genre": "Romance", "year": 1942, "director": "Michael Curtiz"},
    {"title": "Superbad", "genre": "Comedy", "year": 2007, "director": "Greg Mottola"},
    {"title": "The Hangover", "genre": "Comedy", "year": 2009, "director": "Todd Phillips"},
    {"title": "Anchorman", "genre": "Comedy", "year": 2004, "director": "Adam McKay"},
    {"title": "The Conjuring", "genre": "Horror", "year": 2013, "director": "James Wan"},
    {"title": "A Quiet Place", "genre": "Horror", "year": 2018, "director": "John Krasinski"},
    {"title": "Hereditary", "genre": "Horror", "year": 2018, "director": "Ari Aster"},
    {"title": "Spirited Away", "genre": "Animation", "year": 2001, "director": "Hayao Miyazaki"},
    {"title": "Toy Story", "genre": "Animation", "year": 1995, "director": "John Lasseter"},
    {"title": "Finding Nemo", "genre": "Animation", "year": 2003, "director": "Andrew Stanton"},
    {"title": "Mad Max: Fury Road", "genre": "Action", "year": 2015, "director": "George Miller"},
    {"title": "John Wick", "genre": "Action", "year": 2014, "director": "Chad Stahelski"},
    {"title": "La La Land", "genre": "Musical", "year": 2016, "director": "Damien Chazelle"},
    {"title": "The Greatest Showman", "genre": "Musical", "year": 2017, "director": "Michael Gracey"},
    {"title": "Parasite", "genre": "Thriller", "year": 2019, "director": "Bong Joon-ho"},
    {"title": "Gone Girl", "genre": "Thriller", "year": 2014, "director": "David Fincher"},
    {"title": "Dune", "genre": "Sci-Fi", "year": 2021, "director": "Denis Villeneuve"},
    {"title": "Spider-Man: Into the Spider-Verse", "genre": "Animation", "year": 2018, "director": "Peter Ramsey"}
]

# Tworzenie DataFrame z filmami
movies_df = pd.DataFrame(movies_data)
movies_df['movie_id'] = range(1, len(movies_df) + 1)

# Generowanie użytkowników
users_data = []
for i in range(1, 101):  # 100 użytkowników
    age = np.random.randint(18, 65)
    gender = np.random.choice(['M', 'F', 'Other'])
    users_data.append({
        'user_id': i,
        'age': age,
        'gender': gender
    })

users_df = pd.DataFrame(users_data)

# Generowanie ocen (ratings)
# Każdy użytkownik ocenia losową liczbę filmów (5-20)
ratings_data = []

for user_id in range(1, 101):
    # Losowa liczba filmów które użytkownik obejrzał
    num_movies = np.random.randint(8, 25)
    
    # Losowe filmy które użytkownik obejrzał
    watched_movies = np.random.choice(movies_df['movie_id'].values, num_movies, replace=False)
    
    # Preferencje użytkownika - niektórzy wolą określone gatunki
    user_preferences = {}
    preferred_genres = np.random.choice(movies_df['genre'].unique(), np.random.randint(1, 4), replace=False)
    
    for movie_id in watched_movies:
        movie_genre = movies_df[movies_df['movie_id'] == movie_id]['genre'].iloc[0]
        
        # Jeśli film z preferowanego gatunku, wyższa szansa na dobrą ocenę
        if movie_genre in preferred_genres:
            rating = np.random.choice([3, 4, 5], p=[0.2, 0.4, 0.4])
        else:
            rating = np.random.choice([1, 2, 3, 4, 5], p=[0.1, 0.2, 0.4, 0.2, 0.1])
        
        # Timestamp - losowa data z ostatnich 2 lat
        timestamp = datetime.now() - timedelta(days=np.random.randint(1, 730))
        
        ratings_data.append({
            'user_id': user_id,
            'movie_id': movie_id,
            'rating': rating,
            'timestamp': timestamp
        })

ratings_df = pd.DataFrame(ratings_data)

# Zapisywanie do plików CSV
movies_df.to_csv('movies.csv', index=False)
users_df.to_csv('users.csv', index=False)
ratings_df.to_csv('ratings.csv', index=False)

print("Wygenerowano przykladowe dane:")
print(f"Filmy: {len(movies_df)} pozycji")
print(f"Uzytkownicy: {len(users_df)} osob")
print(f"Oceny: {len(ratings_df)} ocen")
print("\nPliki zapisane:")
print("- movies.csv")
print("- users.csv") 
print("- ratings.csv")

# Podstawowe statystyki
print(f"\nStatystyki:")
print(f"Srednia ocena: {ratings_df['rating'].mean():.2f}")
print(f"Najczesciej oceniany gatunek: {movies_df.merge(ratings_df, on='movie_id')['genre'].mode().iloc[0]}")
print(f"Rozklad ocen:")
print(ratings_df['rating'].value_counts().sort_index())
