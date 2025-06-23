#!/usr/bin/env python3
"""
Advanced Bookbot System
A comprehensive book management, analysis, and recommendation system
implementing NLP, machine learning, and bibliographic data processing.
"""

import os
import json
import sqlite3
import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
from pathlib import Path
import hashlib
import re
from collections import defaultdict, Counter
import pickle

# NLP and ML imports
try:
    import nltk
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    from nltk.sentiment import SentimentIntensityAnalyzer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("NLTK not available. Some features will be limited.")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.cluster import KMeans
    from sklearn.decomposition import LatentDirichletAllocation
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Scikit-learn not available. ML features will be limited.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bookbot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class Book:
    """Data structure representing a book with comprehensive metadata."""
    isbn: Optional[str] = None
    title: str = ""
    authors: List[str] = None
    publication_year: Optional[int] = None
    publisher: str = ""
    pages: Optional[int] = None
    language: str = "en"
    genres: List[str] = None
    description: str = ""
    rating: Optional[float] = None
    tags: List[str] = None
    file_path: Optional[str] = None
    added_date: Optional[datetime] = None
    last_read: Optional[datetime] = None
    reading_progress: float = 0.0
    
    def __post_init__(self):
        if self.authors is None:
            self.authors = []
        if self.genres is None:
            self.genres = []
        if self.tags is None:
            self.tags = []
        if self.added_date is None:
            self.added_date = datetime.now()

class DatabaseManager:
    """Handles all database operations for the Bookbot system."""
    
    def __init__(self, db_path: str = "bookbot.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the SQLite database with required tables."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Books table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS books (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    isbn TEXT UNIQUE,
                    title TEXT NOT NULL,
                    authors TEXT,
                    publication_year INTEGER,
                    publisher TEXT,
                    pages INTEGER,
                    language TEXT DEFAULT 'en',
                    genres TEXT,
                    description TEXT,
                    rating REAL,
                    tags TEXT,
                    file_path TEXT,
                    added_date TEXT,
                    last_read TEXT,
                    reading_progress REAL DEFAULT 0.0
                )
            ''')
            
            # Reading sessions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS reading_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    book_id INTEGER,
                    start_time TEXT,
                    end_time TEXT,
                    pages_read INTEGER,
                    notes TEXT,
                    FOREIGN KEY (book_id) REFERENCES books (id)
                )
            ''')
            
            # Book analysis table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS book_analysis (
                    book_id INTEGER PRIMARY KEY,
                    word_count INTEGER,
                    readability_score REAL,
                    sentiment_score REAL,
                    topics TEXT,
                    complexity_score REAL,
                    FOREIGN KEY (book_id) REFERENCES books (id)
                )
            ''')
            
            conn.commit()
    
    def add_book(self, book: Book) -> int:
        """Add a book to the database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO books (
                    isbn, title, authors, publication_year, publisher,
                    pages, language, genres, description, rating, tags,
                    file_path, added_date, last_read, reading_progress
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                book.isbn, book.title, json.dumps(book.authors),
                book.publication_year, book.publisher, book.pages,
                book.language, json.dumps(book.genres), book.description,
                book.rating, json.dumps(book.tags), book.file_path,
                book.added_date.isoformat() if book.added_date else None,
                book.last_read.isoformat() if book.last_read else None,
                book.reading_progress
            ))
            
            return cursor.lastrowid
    
    def get_book(self, book_id: int) -> Optional[Book]:
        """Retrieve a book by ID."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM books WHERE id = ?', (book_id,))
            row = cursor.fetchone()
            
            if row:
                return self._row_to_book(row)
            return None
    
    def search_books(self, query: str, field: str = "title") -> List[Tuple[int, Book]]:
        """Search books by various fields."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            if field == "title":
                cursor.execute('SELECT * FROM books WHERE title LIKE ?', (f'%{query}%',))
            elif field == "author":
                cursor.execute('SELECT * FROM books WHERE authors LIKE ?', (f'%{query}%',))
            elif field == "genre":
                cursor.execute('SELECT * FROM books WHERE genres LIKE ?', (f'%{query}%',))
            else:
                cursor.execute('''
                    SELECT * FROM books WHERE 
                    title LIKE ? OR authors LIKE ? OR description LIKE ?
                ''', (f'%{query}%', f'%{query}%', f'%{query}%'))
            
            results = []
            for row in cursor.fetchall():
                book = self._row_to_book(row)
                results.append((row[0], book))
            
            return results
    
    def _row_to_book(self, row) -> Book:
        """Convert database row to Book object."""
        return Book(
            isbn=row[1],
            title=row[2],
            authors=json.loads(row[3]) if row[3] else [],
            publication_year=row[4],
            publisher=row[5],
            pages=row[6],
            language=row[7],
            genres=json.loads(row[8]) if row[8] else [],
            description=row[9],
            rating=row[10],
            tags=json.loads(row[11]) if row[11] else [],
            file_path=row[12],
            added_date=datetime.fromisoformat(row[13]) if row[13] else None,
            last_read=datetime.fromisoformat(row[14]) if row[14] else None,
            reading_progress=row[15]
        )

class BookAnalyzer:
    """Advanced text analysis and NLP processing for books."""
    
    def __init__(self):
        self.stemmer = PorterStemmer() if NLTK_AVAILABLE else None
        self.sia = SentimentIntensityAnalyzer() if NLTK_AVAILABLE else None
        self.stop_words = set(stopwords.words('english')) if NLTK_AVAILABLE else set()
        
        # Download required NLTK data
        if NLTK_AVAILABLE:
            try:
                nltk.download('punkt', quiet=True)
                nltk.download('stopwords', quiet=True)
                nltk.download('vader_lexicon', quiet=True)
            except:
                logger.warning("Failed to download NLTK data")
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """Comprehensive text analysis including readability, sentiment, and complexity."""
        if not NLTK_AVAILABLE:
            return {"error": "NLTK not available for text analysis"}
        
        analysis = {}
        
        # Basic statistics
        sentences = sent_tokenize(text)
        words = word_tokenize(text.lower())
        words = [w for w in words if w.isalpha()]
        
        analysis['word_count'] = len(words)
        analysis['sentence_count'] = len(sentences)
        analysis['avg_words_per_sentence'] = len(words) / len(sentences) if sentences else 0
        
        # Readability metrics (Flesch Reading Ease approximation)
        syllable_count = sum(self._count_syllables(word) for word in words)
        analysis['avg_syllables_per_word'] = syllable_count / len(words) if words else 0
        
        flesch_score = (206.835 - 1.015 * analysis['avg_words_per_sentence'] - 
                       84.6 * analysis['avg_syllables_per_word'])
        analysis['readability_score'] = max(0, min(100, flesch_score))
        
        # Sentiment analysis
        if self.sia:
            sentiment = self.sia.polarity_scores(text)
            analysis['sentiment'] = sentiment
            analysis['sentiment_score'] = sentiment['compound']
        
        # Vocabulary diversity
        unique_words = set(words)
        analysis['vocabulary_diversity'] = len(unique_words) / len(words) if words else 0
        
        # Most frequent words (excluding stop words)
        filtered_words = [w for w in words if w not in self.stop_words]
        word_freq = Counter(filtered_words)
        analysis['top_words'] = word_freq.most_common(10)
        
        # Complexity score (composite metric)
        complexity = (
            (analysis['avg_words_per_sentence'] / 20) * 0.3 +
            (analysis['avg_syllables_per_word'] / 2) * 0.3 +
            (1 - analysis['vocabulary_diversity']) * 0.2 +
            (len(unique_words) / 1000) * 0.2
        )
        analysis['complexity_score'] = min(10, complexity * 10)
        
        return analysis
    
    def _count_syllables(self, word: str) -> int:
        """Estimate syllable count in a word."""
        word = word.lower()
        vowels = "aeiouy"
        syllable_count = 0
        prev_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_was_vowel:
                syllable_count += 1
            prev_was_vowel = is_vowel
        
        # Handle silent 'e'
        if word.endswith('e') and syllable_count > 1:
            syllable_count -= 1
        
        return max(1, syllable_count)
    
    def extract_topics(self, texts: List[str], n_topics: int = 5) -> List[List[Tuple[str, float]]]:
        """Extract topics from a collection of texts using LDA."""
        if not SKLEARN_AVAILABLE or not texts:
            return []
        
        # Preprocess texts
        processed_texts = []
        for text in texts:
            if NLTK_AVAILABLE:
                words = word_tokenize(text.lower())
                words = [w for w in words if w.isalpha() and w not in self.stop_words]
                processed_texts.append(' '.join(words))
            else:
                # Simple preprocessing without NLTK
                words = re.findall(r'\b[a-z]+\b', text.lower())
                processed_texts.append(' '.join(words))
        
        # TF-IDF vectorization
        vectorizer = TfidfVectorizer(max_features=100, min_df=2, max_df=0.8)
        tfidf_matrix = vectorizer.fit_transform(processed_texts)
        
        # LDA topic modeling
        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        lda.fit(tfidf_matrix)
        
        # Extract top words for each topic
        feature_names = vectorizer.get_feature_names_out()
        topics = []
        
        for topic_idx, topic in enumerate(lda.components_):
            top_words_idx = topic.argsort()[-10:][::-1]
            top_words = [(feature_names[i], topic[i]) for i in top_words_idx]
            topics.append(top_words)
        
        return topics

class RecommendationEngine:
    """Machine learning-based book recommendation system."""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.vectorizer = TfidfVectorizer(max_features=1000) if SKLEARN_AVAILABLE else None
        self.book_vectors = None
        self.book_ids = []
    
    def build_recommendation_model(self):
        """Build the recommendation model based on book descriptions and metadata."""
        if not SKLEARN_AVAILABLE:
            logger.warning("Scikit-learn not available. Recommendation features limited.")
            return
        
        # Get all books from database
        with sqlite3.connect(self.db_manager.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT id, title, authors, genres, description FROM books')
            books = cursor.fetchall()
        
        if not books:
            logger.warning("No books in database for building recommendation model")
            return
        
        # Create feature vectors from book metadata
        book_features = []
        self.book_ids = []
        
        for book_id, title, authors, genres, description in books:
            # Combine text features
            authors_str = ' '.join(json.loads(authors) if authors else [])
            genres_str = ' '.join(json.loads(genres) if genres else [])
            feature_text = f"{title} {authors_str} {genres_str} {description or ''}"
            
            book_features.append(feature_text)
            self.book_ids.append(book_id)
        
        # Create TF-IDF vectors
        self.book_vectors = self.vectorizer.fit_transform(book_features)
        logger.info(f"Built recommendation model with {len(books)} books")
    
    def get_recommendations(self, book_id: int, n_recommendations: int = 5) -> List[Tuple[int, float]]:
        """Get book recommendations based on similarity to a given book."""
        if not SKLEARN_AVAILABLE or self.book_vectors is None:
            return []
        
        try:
            book_idx = self.book_ids.index(book_id)
        except ValueError:
            logger.error(f"Book ID {book_id} not found in recommendation model")
            return []
        
        # Calculate cosine similarity
        book_vector = self.book_vectors[book_idx]
        similarities = cosine_similarity(book_vector, self.book_vectors).flatten()
        
        # Get top similar books (excluding the input book itself)
        similar_indices = similarities.argsort()[::-1][1:n_recommendations+1]
        
        recommendations = []
        for idx in similar_indices:
            recommended_book_id = self.book_ids[idx]
            similarity_score = similarities[idx]
            recommendations.append((recommended_book_id, similarity_score))
        
        return recommendations
    
    def get_personalized_recommendations(self, user_ratings: Dict[int, float], 
                                       n_recommendations: int = 5) -> List[Tuple[int, float]]:
        """Get personalized recommendations based on user ratings."""
        if not SKLEARN_AVAILABLE or self.book_vectors is None:
            return []
        
        # Create user profile based on rated books
        user_profile = np.zeros(self.book_vectors.shape[1])
        total_weight = 0
        
        for book_id, rating in user_ratings.items():
            if book_id in self.book_ids:
                book_idx = self.book_ids.index(book_id)
                weight = rating / 5.0  # Normalize to 0-1
                user_profile += self.book_vectors[book_idx].toarray().flatten() * weight
                total_weight += weight
        
        if total_weight > 0:
            user_profile /= total_weight
        
        # Calculate similarities to user profile
        similarities = cosine_similarity([user_profile], self.book_vectors).flatten()
        
        # Filter out already rated books
        for book_id in user_ratings.keys():
            if book_id in self.book_ids:
                book_idx = self.book_ids.index(book_id)
                similarities[book_idx] = -1
        
        # Get top recommendations
        top_indices = similarities.argsort()[::-1][:n_recommendations]
        
        recommendations = []
        for idx in top_indices:
            if similarities[idx] > 0:  # Only positive similarities
                recommended_book_id = self.book_ids[idx]
                similarity_score = similarities[idx]
                recommendations.append((recommended_book_id, similarity_score))
        
        return recommendations

class BookMetadataFetcher:
    """Fetch book metadata from external APIs."""
    
    def __init__(self):
        self.google_books_api = "https://www.googleapis.com/books/v1/volumes"
    
    def fetch_by_isbn(self, isbn: str) -> Optional[Dict]:
        """Fetch book metadata using ISBN from Google Books API."""
        try:
            params = {'q': f'isbn:{isbn}'}
            response = requests.get(self.google_books_api, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            if data.get('totalItems', 0) > 0:
                return self._parse_google_books_response(data['items'][0])
                
        except requests.RequestException as e:
            logger.error(f"Error fetching metadata for ISBN {isbn}: {e}")
        
        return None
    
    def search_books(self, title: str, author: str = "") -> List[Dict]:
        """Search for books by title and author."""
        try:
            query = title
            if author:
                query += f" inauthor:{author}"
            
            params = {'q': query, 'maxResults': 10}
            response = requests.get(self.google_books_api, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            for item in data.get('items', []):
                book_data = self._parse_google_books_response(item)
                if book_data:
                    results.append(book_data)
            
            return results
            
        except requests.RequestException as e:
            logger.error(f"Error searching books: {e}")
            return []
    
    def _parse_google_books_response(self, item: Dict) -> Optional[Dict]:
        """Parse Google Books API response to extract book metadata."""
        try:
            volume_info = item.get('volumeInfo', {})
            
            # Extract ISBN
            isbn = None
            for identifier in volume_info.get('industryIdentifiers', []):
                if identifier.get('type') in ['ISBN_13', 'ISBN_10']:
                    isbn = identifier.get('identifier')
                    break
            
            return {
                'isbn': isbn,
                'title': volume_info.get('title', ''),
                'authors': volume_info.get('authors', []),
                'publication_year': self._extract_year(volume_info.get('publishedDate', '')),
                'publisher': volume_info.get('publisher', ''),
                'pages': volume_info.get('pageCount'),
                'language': volume_info.get('language', 'en'),
                'genres': volume_info.get('categories', []),
                'description': volume_info.get('description', ''),
                'rating': volume_info.get('averageRating'),
                'thumbnail': volume_info.get('imageLinks', {}).get('thumbnail')
            }
            
        except Exception as e:
            logger.error(f"Error parsing Google Books response: {e}")
            return None
    
    def _extract_year(self, date_string: str) -> Optional[int]:
        """Extract year from date string."""
        if not date_string:
            return None
        
        try:
            # Try to extract year from various date formats
            if '-' in date_string:
                return int(date_string.split('-')[0])
            elif len(date_string) == 4 and date_string.isdigit():
                return int(date_string)
        except ValueError:
            pass
        
        return None

class Bookbot:
    """Main Bookbot class integrating all components."""
    
    def __init__(self, db_path: str = "bookbot.db"):
        self.db_manager = DatabaseManager(db_path)
        self.analyzer = BookAnalyzer()
        self.recommender = RecommendationEngine(self.db_manager)
        self.metadata_fetcher = BookMetadataFetcher()
        
        # Build recommendation model on initialization
        self.recommender.build_recommendation_model()
        
        logger.info("Bookbot system initialized successfully")
    
    def add_book_by_isbn(self, isbn: str) -> Optional[int]:
        """Add a book to the library using its ISBN."""
        metadata = self.metadata_fetcher.fetch_by_isbn(isbn)
        if not metadata:
            logger.error(f"Could not fetch metadata for ISBN: {isbn}")
            return None
        
        book = Book(
            isbn=metadata['isbn'],
            title=metadata['title'],
            authors=metadata['authors'],
            publication_year=metadata['publication_year'],
            publisher=metadata['publisher'],
            pages=metadata['pages'],
            language=metadata['language'],
            genres=metadata['genres'],
            description=metadata['description'],
            rating=metadata['rating']
        )
        
        book_id = self.db_manager.add_book(book)
        logger.info(f"Added book: {book.title} (ID: {book_id})")
        
        # Rebuild recommendation model
        self.recommender.build_recommendation_model()
        
        return book_id
    
    def add_book_manual(self, title: str, authors: List[str], **kwargs) -> int:
        """Manually add a book with provided metadata."""
        book = Book(title=title, authors=authors, **kwargs)
        book_id = self.db_manager.add_book(book)
        
        logger.info(f"Manually added book: {title} (ID: {book_id})")
        
        # Rebuild recommendation model
        self.recommender.build_recommendation_model()
        
        return book_id
    
    def analyze_book_text(self, book_id: int, text: str) -> Dict[str, Any]:
        """Analyze the text content of a book."""
        analysis = self.analyzer.analyze_text(text)
        
        # Store analysis in database
        with sqlite3.connect(self.db_manager.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO book_analysis 
                (book_id, word_count, readability_score, sentiment_score, complexity_score)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                book_id,
                analysis.get('word_count', 0),
                analysis.get('readability_score', 0),
                analysis.get('sentiment_score', 0),
                analysis.get('complexity_score', 0)
            ))
            conn.commit()
        
        logger.info(f"Analyzed text for book ID: {book_id}")
        return analysis
    
    def get_recommendations(self, book_id: int, n: int = 5) -> List[Dict]:
        """Get book recommendations based on a specific book."""
        recommendations = self.recommender.get_recommendations(book_id, n)
        
        result = []
        for rec_id, similarity in recommendations:
            book = self.db_manager.get_book(rec_id)
            if book:
                result.append({
                    'book_id': rec_id,
                    'book': book,
                    'similarity_score': similarity
                })
        
        return result
    
    def search_library(self, query: str, field: str = "all") -> List[Dict]:
        """Search the book library."""
        results = self.db_manager.search_books(query, field)
        
        return [{'book_id': book_id, 'book': book} for book_id, book in results]
    
    def get_reading_statistics(self) -> Dict[str, Any]:
        """Generate comprehensive reading statistics."""
        with sqlite3.connect(self.db_manager.db_path) as conn:
            cursor = conn.cursor()
            
            # Total books
            cursor.execute('SELECT COUNT(*) FROM books')
            total_books = cursor.fetchone()[0]
            
            # Books by year
            cursor.execute('''
                SELECT publication_year, COUNT(*) 
                FROM books 
                WHERE publication_year IS NOT NULL 
                GROUP BY publication_year 
                ORDER BY publication_year
            ''')
            books_by_year = dict(cursor.fetchall())
            
            # Average rating
            cursor.execute('SELECT AVG(rating) FROM books WHERE rating IS NOT NULL')
            avg_rating = cursor.fetchone()[0] or 0
            
            # Most common genres
            cursor.execute('SELECT genres FROM books WHERE genres IS NOT NULL')
            all_genres = []
            for (genres_json,) in cursor.fetchall():
                genres = json.loads(genres_json)
                all_genres.extend(genres)
            
            genre_counts = Counter(all_genres)
            
            # Reading progress statistics
            cursor.execute('''
                SELECT AVG(reading_progress), 
                       COUNT(CASE WHEN reading_progress > 0 THEN 1 END),
                       COUNT(CASE WHEN reading_progress = 1.0 THEN 1 END)
                FROM books
            ''')
            avg_progress, books_started, books_completed = cursor.fetchone()
            
            return {
                'total_books': total_books,
                'books_by_year': books_by_year,
                'average_rating': round(avg_rating, 2) if avg_rating else 0,
                'top_genres': genre_counts.most_common(10),
                'average_reading_progress': round(avg_progress or 0, 2),
                'books_started': books_started or 0,
                'books_completed': books_completed or 0,
                'completion_rate': round((books_completed or 0) / max(1, books_started or 1) * 100, 1)
            }
    
    def export_library(self, file_path: str, format: str = "json"):
        """Export library to various formats."""
        with sqlite3.connect(self.db_manager.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM books')
            books = cursor.fetchall()
            
            # Convert to list of dictionaries
            columns = [desc[0] for desc in cursor.description]
            books_data = []
            
            for book in books:
                book_dict = dict(zip(columns, book))
                # Parse JSON fields
                for field in ['authors', 'genres', 'tags']:
                    if book_dict[field]:
                        book_dict[field] = json.loads(book_dict[field])
                books_data.append(book_dict)
        
        if format.lower() == "json":
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(books_data, f, indent=2, default=str)
        elif format.lower() == "csv" and pd:
            df = pd.DataFrame(books_data)
            df.to_csv(file_path, index=False)
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        logger.info(f"Library exported to {file_path} in {format} format")

def main():
    """Demonstration of Bookbot capabilities."""
    print("🤖 Advanced Bookbot System")
    print("=" * 50)
    
    # Initialize Bookbot
    bot = Bookbot()
    
    # Example usage
    print("\n1. Adding books by ISBN...")
    example_isbns = ["9780134685991", "9781449355739"]  # Example ISBNs
    
    for isbn in example_isbns:
        book_id = bot.add_book_by_isbn(isbn)
        if book_id:
            print(f"   ✓ Added book with ID: {book_id}")
    
    print("\n2. Manual book addition...")
    manual_book_id = bot.add_book_manual(
        title="The Art of Computer Programming",
        authors=["Donald E. Knuth"],
        publication_year=1968,
        genres=["Computer Science", "Mathematics"],
        description="A comprehensive monograph written by Donald Knuth on algorithms."
    )
    print(f"   ✓ Manually added book with ID: {manual_book_id}")
    
    print("\n3. Searching library...")
    search_results = bot.search_library("programming")
    for result in search_results[:3]:  # Show first 3 results
        book = result['book']
        print(f"   • {book.title} by {', '.join(book.authors)}")
    
    print("\n4. Getting recommendations...")
    if search_results:
        book_id = search_results[0]['book_id']
        recommendations = bot.get_recommendations(book_id, 3)
        print(f"   Recommendations based on '{search_results[0]['book'].title}':")
        for rec in recommendations:
            book = rec['book']
            score = rec['similarity_score']
            print(f"   • {book.title} (similarity: {score:.3f})")
    
    print("\n5. Reading statistics...")
    stats = bot.get_reading_statistics()
    print(f"   • Total books: {stats['total_books']}")
    print(f"   • Average rating: {stats['average_rating']}")
    print(f"   • Completion rate: {stats['completion_rate']}%")