#!/usr/bin/env python3
"""
🎬 Demonstracja Systemu Rekomendacji Filmów
==========================================

Ten skrypt pokazuje jak używać systemu rekomendacji filmów.
"""

from movie_recommendation_system import MovieRecommendationSystem
import sys

def interactive_demo():
    """Interaktywna demonstracja systemu"""
    print("Witaj w Systemie Rekomendacji Filmow!")
    print("=" * 50)
    
    # Inicjalizacja systemu
    recommender = MovieRecommendationSystem()
    
    print("Ladowanie i trenowanie modeli...")
    if not recommender.train_model():
        print("Blad podczas ladowania danych. Sprawdz czy pliki CSV istnieja.")
        return
    
    print("\nSystem gotowy do uzycia!")
    
    while True:
        print("\n" + "="*50)
        print("Co chcesz zrobic?")
        print("1. Zobacz profil użytkownika")
        print("2. Otrzymaj rekomendacje (Collaborative Filtering)")
        print("3. Otrzymaj rekomendacje (Content-based)")
        print("4. Otrzymaj rekomendacje (SVD)")
        print("5. Wszystkie rekomendacje dla użytkownika")
        print("6. Zobacz listę wszystkich filmów")
        print("7. Statystyki systemu")
        print("0. Wyjście")
        
        choice = input("\nWybierz opcję (0-7): ").strip()
        
        if choice == "0":
            print("👋 Dziękujemy za korzystanie z systemu!")
            break
        elif choice == "1":
            user_id = get_user_id()
            if user_id:
                print(recommender.get_user_profile(user_id))
        elif choice == "2":
            user_id = get_user_id()
            if user_id:
                n = get_num_recommendations()
                print(recommender.get_user_recommendations_collaborative(user_id, n))
        elif choice == "3":
            user_id = get_user_id()
            if user_id:
                n = get_num_recommendations()
                print(recommender.get_movie_recommendations_content(user_id, n))
        elif choice == "4":
            user_id = get_user_id()
            if user_id:
                n = get_num_recommendations()
                print(recommender.get_svd_recommendations(user_id, n))
        elif choice == "5":
            user_id = get_user_id()
            if user_id:
                n = get_num_recommendations()
                print("\n🎬 WSZYSTKIE REKOMENDACJE DLA UŻYTKOWNIKA", user_id)
                print("=" * 60)
                print(recommender.get_user_profile(user_id))
                print(recommender.get_user_recommendations_collaborative(user_id, n))
                print(recommender.get_movie_recommendations_content(user_id, n))
                print(recommender.get_svd_recommendations(user_id, n))
        elif choice == "6":
            show_movie_list(recommender)
        elif choice == "7":
            show_statistics(recommender)
        else:
            print("❌ Nieprawidłowa opcja. Spróbuj ponownie.")

def get_user_id():
    """Pobiera ID użytkownika od użytkownika"""
    try:
        user_id = int(input("Podaj ID użytkownika (1-100): "))
        if 1 <= user_id <= 100:
            return user_id
        else:
            print("❌ ID użytkownika musi być między 1 a 100")
            return None
    except ValueError:
        print("❌ Podaj prawidłowy numer")
        return None

def get_num_recommendations():
    """Pobiera liczbę rekomendacji"""
    try:
        n = int(input("Ile rekomendacji chcesz zobaczyć? (1-10): "))
        if 1 <= n <= 10:
            return n
        else:
            print("❌ Liczba rekomendacji musi być między 1 a 10. Używam domyślnej wartości 5.")
            return 5
    except ValueError:
        print("❌ Podaj prawidłowy numer. Używam domyślnej wartości 5.")
        return 5

def show_movie_list(recommender):
    """Pokazuje listę wszystkich filmów"""
    print("\n🎬 Lista wszystkich filmów w bazie:")
    print("=" * 50)
    
    movies = recommender.movies_df.sort_values('year')
    for _, movie in movies.iterrows():
        print(f"{movie['movie_id']:2d}. {movie['title']} ({movie['year']}) - {movie['genre']}")
        print(f"    Reżyser: {movie['director']}")

def show_statistics(recommender):
    """Pokazuje statystyki systemu"""
    print("\n📊 Statystyki Systemu Rekomendacji:")
    print("=" * 40)
    
    print(f"🎬 Liczba filmów: {len(recommender.movies_df)}")
    print(f"👥 Liczba użytkowników: {len(recommender.users_df)}")
    print(f"⭐ Liczba ocen: {len(recommender.ratings_df)}")
    
    print(f"\n📈 Statystyki ocen:")
    print(f"Średnia ocena: {recommender.ratings_df['rating'].mean():.2f}")
    print(f"Odchylenie standardowe: {recommender.ratings_df['rating'].std():.2f}")
    
    print(f"\n🎭 Rozkład gatunków:")
    genre_counts = recommender.movies_df['genre'].value_counts()
    for genre, count in genre_counts.items():
        print(f"  {genre}: {count} filmów")
    
    print(f"\n⭐ Rozkład ocen:")
    rating_counts = recommender.ratings_df['rating'].value_counts().sort_index()
    for rating, count in rating_counts.items():
        print(f"  {rating} gwiazdek: {count} ocen ({count/len(recommender.ratings_df)*100:.1f}%)")

def quick_demo():
    """Szybka demonstracja dla przykładowych użytkowników"""
    print("Szybka Demonstracja Systemu")
    print("=" * 40)
    
    recommender = MovieRecommendationSystem()
    
    if not recommender.train_model():
        print("Blad podczas ladowania danych.")
        return
    
    # Testuj dla kilku użytkowników
    test_users = [1, 15, 30]
    
    for user_id in test_users:
        print(f"\n{'='*60}")
        print(f"REKOMENDACJE DLA UZYTKOWNIKA {user_id}")
        print("="*60)
        print(recommender.get_user_profile(user_id))
        print(recommender.get_user_recommendations_collaborative(user_id, 3))
        print(recommender.get_svd_recommendations(user_id, 3))

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        quick_demo()
    else:
        interactive_demo()
