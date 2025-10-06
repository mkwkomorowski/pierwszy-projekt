from flask import Flask, render_template, request, jsonify
from movie_recommendation_system import MovieRecommendationSystem
import pandas as pd

app = Flask(__name__)

# Inicjalizacja systemu rekomendacji (globalnie)
recommender = None

def init_recommender():
    """Inicjalizuje system rekomendacji"""
    global recommender
    if recommender is None:
        recommender = MovieRecommendationSystem()
        if recommender.train_model():
            print("✅ System rekomendacji załadowany!")
        else:
            print("❌ Błąd podczas ładowania systemu")
            recommender = None
    return recommender is not None

@app.route('/')
def index():
    """Strona główna"""
    if not init_recommender():
        return render_template('error.html', message="Błąd ładowania systemu rekomendacji")
    
    # Pobierz listę użytkowników i filmów
    users = list(range(1, 101))  # 100 użytkowników
    movies = recommender.movies_df.to_dict('records')
    
    return render_template('index.html', users=users, movies=movies)

@app.route('/user/<int:user_id>')
def user_profile(user_id):
    """Profil użytkownika"""
    if not init_recommender():
        return jsonify({'error': 'System nie jest dostępny'})
    
    if user_id < 1 or user_id > 100:
        return jsonify({'error': 'Nieprawidłowy ID użytkownika'})
    
    profile = recommender.get_user_profile(user_id)
    return render_template('user_profile.html', user_id=user_id, profile=profile)

@app.route('/api/recommendations/<int:user_id>')
def get_recommendations_api(user_id):
    """API dla rekomendacji"""
    if not init_recommender():
        return jsonify({'error': 'System nie jest dostępny'})
    
    if user_id < 1 or user_id > 100:
        return jsonify({'error': 'Nieprawidłowy ID użytkownika'})
    
    method = request.args.get('method', 'svd')
    n_recs = int(request.args.get('n', 5))
    
    try:
        if method == 'collaborative':
            recs = recommender.get_user_recommendations_collaborative(user_id, n_recs)
        elif method == 'content':
            recs = recommender.get_movie_recommendations_content(user_id, n_recs)
        else:  # svd
            recs = recommender.get_svd_recommendations(user_id, n_recs)
        
        return jsonify({'recommendations': recs, 'user_id': user_id, 'method': method})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/recommendations/<int:user_id>')
def recommendations(user_id):
    """Strona z rekomendacjami"""
    if not init_recommender():
        return render_template('error.html', message="System nie jest dostępny")
    
    if user_id < 1 or user_id > 100:
        return render_template('error.html', message="Nieprawidłowy ID użytkownika")
    
    # Pobierz profil użytkownika
    profile = recommender.get_user_profile(user_id)
    
    # Pobierz rekomendacje ze wszystkich metod
    collaborative_recs = recommender.get_user_recommendations_collaborative(user_id, 5)
    content_recs = recommender.get_movie_recommendations_content(user_id, 5)
    svd_recs = recommender.get_svd_recommendations(user_id, 5)
    
    return render_template('recommendations.html', 
                         user_id=user_id, 
                         profile=profile,
                         collaborative_recs=collaborative_recs,
                         content_recs=content_recs,
                         svd_recs=svd_recs)

@app.route('/movies')
def movies_list():
    """Lista wszystkich filmów"""
    if not init_recommender():
        return render_template('error.html', message="System nie jest dostępny")
    
    movies = recommender.movies_df.to_dict('records')
    return render_template('movies.html', movies=movies)

@app.route('/stats')
def statistics():
    """Statystyki systemu"""
    if not init_recommender():
        return render_template('error.html', message="System nie jest dostępny")
    
    stats = {
        'total_movies': len(recommender.movies_df),
        'total_users': len(recommender.users_df),
        'total_ratings': len(recommender.ratings_df),
        'avg_rating': round(recommender.ratings_df['rating'].mean(), 2),
        'genre_counts': recommender.movies_df['genre'].value_counts().to_dict(),
        'rating_counts': recommender.ratings_df['rating'].value_counts().sort_index().to_dict()
    }
    
    return render_template('stats.html', stats=stats)

if __name__ == '__main__':
    print("🚀 Uruchamianie aplikacji webowej...")
    app.run(debug=True, host='127.0.0.1', port=5000)
