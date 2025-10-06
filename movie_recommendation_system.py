import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

class MovieRecommendationSystem:
    def __init__(self):
        self.movies_df = None
        self.users_df = None
        self.ratings_df = None
        self.user_movie_matrix = None
        self.svd_model = None
        self.user_similarity = None
        self.movie_similarity = None
        
    def load_data(self):
        """Ładuje dane z plików CSV"""
        try:
            self.movies_df = pd.read_csv('movies.csv')
            self.users_df = pd.read_csv('users.csv')
            self.ratings_df = pd.read_csv('ratings.csv')
            print("Dane zaladowane pomyslnie!")
            return True
        except FileNotFoundError as e:
            print(f"Blad: Nie znaleziono pliku {e.filename}")
            print("Najpierw uruchom generate_movie_data.py")
            return False
    
    def create_user_movie_matrix(self):
        """Tworzy macierz użytkownik-film"""
        self.user_movie_matrix = self.ratings_df.pivot(
            index='user_id', 
            columns='movie_id', 
            values='rating'
        ).fillna(0)
        print(f"Macierz uzytkownik-film: {self.user_movie_matrix.shape}")
        
    def train_svd_model(self, n_components=20):
        """Trenuje model SVD (Singular Value Decomposition)"""
        print("Trenowanie modelu SVD...")
        
        # Przygotowanie danych
        X = self.user_movie_matrix.values
        
        # Podział na zbiór treningowy i testowy
        train_data = []
        test_data = []
        
        for i in range(len(self.ratings_df)):
            if np.random.random() < 0.8:  # 80% dla treningu
                train_data.append(self.ratings_df.iloc[i])
            else:
                test_data.append(self.ratings_df.iloc[i])
        
        # Trenowanie modelu SVD
        self.svd_model = TruncatedSVD(n_components=n_components, random_state=42)
        self.svd_model.fit(X)
        
        # Transformacja danych
        user_factors = self.svd_model.transform(X)
        movie_factors = self.svd_model.components_.T
        
        # Obliczanie predykcji dla zbioru testowego
        if test_data:
            predictions = []
            actuals = []
            
            for rating in test_data:
                user_idx = rating['user_id'] - 1
                movie_idx = rating['movie_id'] - 1
                
                if user_idx < len(user_factors) and movie_idx < len(movie_factors):
                    pred = np.dot(user_factors[user_idx], movie_factors[movie_idx])
                    predictions.append(pred)
                    actuals.append(rating['rating'])
            
            if predictions:
                rmse = np.sqrt(mean_squared_error(actuals, predictions))
                print(f"RMSE na zbiorze testowym: {rmse:.3f}")
        
        print("Model SVD wytrenowany!")
        
    def calculate_user_similarity(self):
        """Oblicza podobieństwo między użytkownikami"""
        print("Obliczanie podobienstwa uzytkownikow...")
        self.user_similarity = cosine_similarity(self.user_movie_matrix)
        print("Podobienstwo uzytkownikow obliczone!")
        
    def calculate_movie_similarity(self):
        """Oblicza podobieństwo między filmami"""
        print("Obliczanie podobienstwa filmow...")
        self.movie_similarity = cosine_similarity(self.user_movie_matrix.T)
        print("Podobienstwo filmow obliczone!")
        
    def get_user_recommendations_collaborative(self, user_id, n_recommendations=5):
        """Rekomendacje oparte na collaborative filtering (użytkownicy)"""
        if user_id not in self.user_movie_matrix.index:
            return f"Uzytkownik {user_id} nie istnieje w bazie danych"
        
        user_idx = user_id - 1
        user_ratings = self.user_movie_matrix.loc[user_id]
        
        # Znajdź podobnych użytkowników
        similar_users = self.user_similarity[user_idx]
        
        # Przewidywane oceny dla filmów, których użytkownik nie ocenił
        recommendations = {}
        
        for movie_id in self.user_movie_matrix.columns:
            if user_ratings[movie_id] == 0:  # Film nie został oceniony
                weighted_sum = 0
                similarity_sum = 0
                
                for other_user_id in self.user_movie_matrix.index:
                    if other_user_id != user_id:
                        other_user_idx = other_user_id - 1
                        other_user_rating = self.user_movie_matrix.loc[other_user_id, movie_id]
                        
                        if other_user_rating > 0:
                            similarity = similar_users[other_user_idx]
                            weighted_sum += similarity * other_user_rating
                            similarity_sum += abs(similarity)
                
                if similarity_sum > 0:
                    predicted_rating = weighted_sum / similarity_sum
                    recommendations[movie_id] = predicted_rating
        
        # Sortuj i zwróć top N
        top_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:n_recommendations]
        
        return self._format_recommendations(top_recommendations, "Collaborative Filtering (Users)")
        
    def get_movie_recommendations_content(self, user_id, n_recommendations=5):
        """Rekomendacje oparte na podobieństwie filmów"""
        if user_id not in self.user_movie_matrix.index:
            return f"Uzytkownik {user_id} nie istnieje w bazie danych"
        
        user_ratings = self.user_movie_matrix.loc[user_id]
        watched_movies = user_ratings[user_ratings > 0]
        
        # Znajdź filmy podobne do tych, które użytkownik dobrze ocenił (4-5 gwiazdek)
        high_rated_movies = watched_movies[watched_movies >= 4]
        
        if len(high_rated_movies) == 0:
            return "Uzytkownik nie ma wysoko ocenionych filmow"
        
        recommendations = {}
        
        for movie_id in high_rated_movies.index:
            movie_idx = movie_id - 1
            similar_movies = self.movie_similarity[movie_idx]
            
            for other_movie_id in self.user_movie_matrix.columns:
                if user_ratings[other_movie_id] == 0:  # Film nie został oceniony
                    other_movie_idx = other_movie_id - 1
                    similarity = similar_movies[other_movie_idx]
                    
                    if other_movie_id not in recommendations:
                        recommendations[other_movie_id] = 0
                    
                    recommendations[other_movie_id] += similarity * high_rated_movies[movie_id]
        
        # Sortuj i zwróć top N
        top_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:n_recommendations]
        
        return self._format_recommendations(top_recommendations, "Content-based (Movie Similarity)")
        
    def get_svd_recommendations(self, user_id, n_recommendations=5):
        """Rekomendacje oparte na modelu SVD"""
        if user_id not in self.user_movie_matrix.index:
            return f"Uzytkownik {user_id} nie istnieje w bazie danych"
        
        user_idx = user_id - 1
        user_ratings = self.user_movie_matrix.loc[user_id]
        
        # Przekształć dane użytkownika przez model SVD
        user_vector = self.user_movie_matrix.iloc[user_idx:user_idx+1].values
        user_factors = self.svd_model.transform(user_vector)[0]
        movie_factors = self.svd_model.components_.T
        
        # Oblicz przewidywane oceny
        predictions = np.dot(user_factors, movie_factors.T)
        
        # Znajdź filmy, które użytkownik nie ocenił
        recommendations = {}
        for i, movie_id in enumerate(self.user_movie_matrix.columns):
            if user_ratings[movie_id] == 0:
                recommendations[movie_id] = predictions[i]
        
        # Sortuj i zwróć top N
        top_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:n_recommendations]
        
        return self._format_recommendations(top_recommendations, "SVD Matrix Factorization")
        
    def _format_recommendations(self, recommendations, method_name):
        """Formatuje rekomendacje z tytułami filmów"""
        result = f"\nRekomendacje ({method_name}):\n"
        result += "=" * 50 + "\n"
        
        for i, (movie_id, score) in enumerate(recommendations, 1):
            movie_info = self.movies_df[self.movies_df['movie_id'] == movie_id].iloc[0]
            result += f"{i}. {movie_info['title']} ({movie_info['year']})\n"
            result += f"   Gatunek: {movie_info['genre']} | Reżyser: {movie_info['director']}\n"
            result += f"   Przewidywana ocena: {score:.2f}\n\n"
        
        return result
        
    def get_user_profile(self, user_id):
        """Pokazuje profil użytkownika i jego oceny"""
        if user_id not in self.user_movie_matrix.index:
            return f"Uzytkownik {user_id} nie istnieje w bazie danych"
        
        user_info = self.users_df[self.users_df['user_id'] == user_id].iloc[0]
        user_ratings = self.ratings_df[self.ratings_df['user_id'] == user_id]
        
        result = f"\nProfil uzytkownika {user_id}:\n"
        result += "=" * 30 + "\n"
        result += f"Wiek: {user_info['age']}\n"
        result += f"Płeć: {user_info['gender']}\n"
        result += f"Liczba ocenionych filmow: {len(user_ratings)}\n"
        result += f"Srednia ocena: {user_ratings['rating'].mean():.2f}\n\n"
        
        result += "Ostatnie oceny:\n"
        recent_ratings = user_ratings.merge(self.movies_df, on='movie_id').sort_values('timestamp', ascending=False).head(5)
        
        for _, rating in recent_ratings.iterrows():
            result += f"* {rating['title']} ({rating['year']}) - {rating['rating']} gwiazdek [{rating['genre']}]\n"
        
        return result
        
    def train_model(self):
        """Trenuje wszystkie modele"""
        print("Rozpoczynanie trenowania modeli rekomendacji...\n")
        
        if not self.load_data():
            return False
            
        self.create_user_movie_matrix()
        self.train_svd_model()
        self.calculate_user_similarity()
        self.calculate_movie_similarity()
        
        print("\nWszystkie modele zostaly wytrenowane!")
        return True

# Funkcja pomocnicza do demonstracji
def demo_recommendations():
    """Demonstracja systemu rekomendacji"""
    print("System Rekomendacji Filmow")
    print("=" * 40)
    
    # Inicjalizacja systemu
    recommender = MovieRecommendationSystem()
    
    if not recommender.train_model():
        return
    
    # Testuj rekomendacje dla kilku użytkowników
    test_users = [1, 15, 30, 50]
    
    for user_id in test_users:
        print(f"\n{'='*60}")
        print(recommender.get_user_profile(user_id))
        print(recommender.get_user_recommendations_collaborative(user_id, 3))
        print(recommender.get_movie_recommendations_content(user_id, 3))
        print(recommender.get_svd_recommendations(user_id, 3))

if __name__ == "__main__":
    demo_recommendations()
