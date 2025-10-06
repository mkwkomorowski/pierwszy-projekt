# 🎬 System Rekomendacji Filmów

Kompletny system rekomendacji filmów wykorzystujący różne algorytmy machine learning do przewidywania preferencji użytkowników.

## 🚀 Funkcjonalności

- **Collaborative Filtering** - rekomendacje oparte na podobieństwie użytkowników
- **Content-based Filtering** - rekomendacje oparte na podobieństwie filmów  
- **Matrix Factorization (SVD)** - zaawansowany model redukcji wymiarowości
- **Interaktywny interfejs** - łatwy w użyciu system menu
- **Analiza profilu użytkownika** - szczegółowe informacje o preferencjach

## 📋 Wymagania

```bash
pip install -r requirements.txt
```

## 🎯 Jak używać

### 1. Generowanie danych przykładowych
```bash
python generate_movie_data.py
```
To utworzy 3 pliki CSV:
- `movies.csv` - baza 30 filmów z różnych gatunków
- `users.csv` - 100 przykładowych użytkowników
- `ratings.csv` - ~1500 ocen filmów

### 2. Uruchomienie systemu rekomendacji

#### Tryb interaktywny:
```bash
python demo.py
```

#### Szybka demonstracja:
```bash
python demo.py --quick
```

#### Bezpośrednie użycie w kodzie:
```python
from movie_recommendation_system import MovieRecommendationSystem

# Inicjalizacja i trenowanie
recommender = MovieRecommendationSystem()
recommender.train_model()

# Rekomendacje dla użytkownika
recommendations = recommender.get_svd_recommendations(user_id=1, n_recommendations=5)
print(recommendations)
```

## 🧠 Algorytmy

### 1. Collaborative Filtering
- Znajduje użytkowników o podobnych gustach
- Rekomenduje filmy które podobali się innym użytkownikom
- Wykorzystuje cosine similarity

### 2. Content-based Filtering  
- Analizuje podobieństwo między filmami
- Rekomenduje filmy podobne do tych, które użytkownik ocenił wysoko
- Opiera się na gatunkach i cechach filmów

### 3. SVD Matrix Factorization
- Zaawansowana technika redukcji wymiarowości
- Znajduje ukryte wzorce w danych
- Najlepsze wyniki predykcyjne

## 📊 Przykładowe wyniki

```
🎬 Rekomendacje (SVD Matrix Factorization):
==================================================
1. Inception (2010)
   Gatunek: Sci-Fi | Reżyser: Christopher Nolan
   Przewidywana ocena: 4.23

2. The Dark Knight (2008)
   Gatunek: Action | Reżyser: Christopher Nolan
   Przewidywana ocena: 4.18

3. Pulp Fiction (1994)
   Gatunek: Crime | Reżyser: Quentin Tarantino
   Przewidywana ocena: 4.05
```

## 🔧 Struktura projektu

```
📁 moj-projekt/
├── 🐍 generate_movie_data.py          # Generator danych przykładowych
├── 🤖 movie_recommendation_system.py  # Główny system rekomendacji
├── 🎮 demo.py                        # Interaktywny interfejs
├── 📋 requirements.txt               # Zależności Python
├── 📖 README.md                      # Dokumentacja
├── 📊 movies.csv                     # Baza filmów (generowana)
├── 👥 users.csv                      # Baza użytkowników (generowana)
└── ⭐ ratings.csv                    # Oceny użytkowników (generowana)
```

## 🎓 Zastosowania edukacyjne

Ten projekt jest idealny do nauki:
- **Machine Learning** - różne algorytmy rekomendacji
- **Data Science** - analiza i przetwarzanie danych
- **Python** - zaawansowane programowanie
- **Collaborative Filtering** - systemy rekomendacyjne
- **Matrix Factorization** - techniki redukcji wymiarowości

## 🚀 Możliwe rozszerzenia

1. **Deep Learning** - implementacja neural collaborative filtering
2. **Hybrid Models** - kombinacja różnych algorytmów
3. **Web Interface** - aplikacja webowa Flask/Django
4. **Real-time** - system rekomendacji w czasie rzeczywistym
5. **A/B Testing** - testowanie różnych algorytmów
6. **Cold Start** - rozwiązanie problemu nowych użytkowników

## 📈 Metryki jakości

System automatycznie oblicza:
- **RMSE** (Root Mean Square Error) - błąd predykcji
- **Pokrycie** - procent filmów które system może rekomendować
- **Różnorodność** - zróżnicowanie rekomendacji

---

*Stworzony z ❤️ dla nauki Data Science i Machine Learning*
