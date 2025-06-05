import pandas as pd
import numpy as np
import re
import json
from collections import Counter
from typing import Dict, List, Tuple, Optional
import logging

# NLP libraries
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.chunk import ne_chunk
from nltk.tag import pos_tag

# Scikit-learn for feature extraction
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import StandardScaler
# TextBlob for sentiment analysis
from textblob import TextBlob

# Download 2.required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
    nltk.data.find('taggers/averaged_perceptron_tagger')
    nltk.data.find('taggers/averaged_perceptron_tagger_eng')
    nltk.data.find('chunkers/maxent_ne_chunker')
    nltk.data.find('chunkers/maxent_ne_chunker_tab')  # Add this line
    nltk.data.find('corpora/words')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('averaged_perceptron_tagger_eng')
    nltk.download('maxent_ne_chunker')
    nltk.download('maxent_ne_chunker_tab')  # Add this line
    nltk.download('words')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FinancialNLPProcessor:
    """
    Advanced NLP processor for financial KAP reports
    """
    
    def __init__(self):
        """Initialize the NLP processor with financial domain knowledge"""
        
        # Initialize NLTK components
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Financial keywords and sentiment lexicons
        self.positive_financial_terms = {
            'growth', 'increase', 'profit', 'revenue', 'gain', 'expansion', 'improvement',
            'strong', 'positive', 'success', 'achievement', 'opportunity', 'benefit',
            'excellent', 'outstanding', 'favorable', 'optimistic', 'bullish', 'upward',
            'rising', 'boost', 'enhance', 'strengthen', 'upgrade', 'advance'
        }
        
        self.negative_financial_terms = {
            'loss', 'decrease', 'decline', 'fall', 'drop', 'negative', 'poor', 'weak',
            'disappointing', 'concern', 'risk', 'threat', 'challenge', 'difficulty',
            'problem', 'issue', 'bearish', 'downward', 'falling', 'reduce', 'lower',
            'cut', 'downgrade', 'deteriorate', 'worsen', 'crisis'
        }
        
        self.financial_entities = {
            'revenue', 'profit', 'earnings', 'sales', 'income', 'costs', 'expenses',
            'margin', 'ebitda', 'dividend', 'share', 'stock', 'equity', 'debt',
            'assets', 'liabilities', 'cash', 'investment', 'acquisition', 'merger',
            'board', 'ceo', 'management', 'director', 'chairman', 'governance'
        }
        
        # Report type indicators
        self.report_type_keywords = {
            'financial_statement': ['financial', 'statement', 'earnings', 'quarterly', 'annual', 'results'],
            'board_changes': ['board', 'director', 'appointment', 'resignation', 'election', 'management'],
            'corporate_action': ['dividend', 'split', 'merger', 'acquisition', 'restructuring'],
            'material_event': ['material', 'event', 'disclosure', 'announcement', 'agreement'],
            'governance': ['governance', 'committee', 'compliance', 'audit', 'risk']
        }
        
        # Initialize vectorizers
        self.tfidf_vectorizer = None
        self.count_vectorizer = None
        self.lda_model = None
        
    def preprocess_text(self, text: str) -> str:
        """
        Clean and preprocess text for analysis
        
        Args:
            text: Raw text content
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace and newlines
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep financial symbols
        text = re.sub(r'[^\w\s\.\,\%\$\-]', ' ', text)
        
        # Remove numbers that are not percentages or financial figures
        # Keep patterns like "5%", "$100", "2.5%", etc.
        text = re.sub(r'\b\d+(?!\s*[%$]|\.\d)', ' ', text)
        
        return text.strip()
    
    def tokenize_and_lemmatize(self, text: str) -> List[str]:
        """
        Tokenize text and apply lemmatization
        
        Args:
            text: Preprocessed text
            
        Returns:
            List of lemmatized tokens
        """
        tokens = word_tokenize(text)
        
        # Remove stopwords and apply lemmatization
        lemmatized_tokens = [
            self.lemmatizer.lemmatize(token.lower())
            for token in tokens
            if token.lower() not in self.stop_words and len(token) > 2
        ]
        
        return lemmatized_tokens
    
    def extract_financial_sentiment(self, text: str) -> Dict:
        """
        Extract financial sentiment using domain-specific approach
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment scores
        """
        # TextBlob sentiment
        blob = TextBlob(text)
        textblob_sentiment = blob.sentiment
        
        # Domain-specific sentiment
        tokens = self.tokenize_and_lemmatize(text)
        
        positive_count = sum(1 for token in tokens if token in self.positive_financial_terms)
        negative_count = sum(1 for token in tokens if token in self.negative_financial_terms)
        
        # Calculate financial sentiment score
        total_financial_terms = positive_count + negative_count
        if total_financial_terms > 0:
            financial_sentiment = (positive_count - negative_count) / total_financial_terms
        else:
            financial_sentiment = 0
        
        return {
            'textblob_polarity': textblob_sentiment.polarity,
            'textblob_subjectivity': textblob_sentiment.subjectivity,
            'financial_sentiment': financial_sentiment,
            'positive_terms_count': positive_count,
            'negative_terms_count': negative_count,
            'total_financial_terms': total_financial_terms
        }
    
    def extract_named_entities(self, text: str) -> Dict:
        """
        Extract named entities and financial entities
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with entity information
        """
        # NLTK Named Entity Recognition
        tokens = word_tokenize(text)
        pos_tags = pos_tag(tokens)
        chunks = ne_chunk(pos_tags)
        
        # Extract named entities
        entities = []
        for chunk in chunks:
            if hasattr(chunk, 'label'):
                entity = ' '.join([token for token, pos in chunk.leaves()])
                entities.append((entity, chunk.label()))
        
        # Extract financial entities
        tokens_lower = [token.lower() for token in tokens]
        financial_entities_found = [
            token for token in tokens_lower 
            if token in self.financial_entities
        ]
        
        return {
            'named_entities': entities,
            'named_entities_count': len(entities),
            'financial_entities': financial_entities_found,
            'financial_entities_count': len(financial_entities_found)
        }
    
    def classify_report_type(self, text: str) -> Dict:
        """
        Classify the type of report based on content
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with report type probabilities
        """
        text_lower = text.lower()
        type_scores = {}
        
        for report_type, keywords in self.report_type_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            type_scores[f'{report_type}_score'] = score
        
        # Determine primary type
        if type_scores:
            primary_type = max(type_scores.keys(), key=lambda k: type_scores[k])
            type_scores['primary_report_type'] = primary_type.replace('_score', '')
        else:
            type_scores['primary_report_type'] = 'unknown'
        
        return type_scores
    
    def extract_numerical_features(self, text: str) -> Dict:
        """
        Extract numerical information from text
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with numerical features
        """
        # Extract percentages
        percentages = re.findall(r'(\d+(?:\.\d+)?)\s*%', text)
        percentages = [float(p) for p in percentages]
        
        # Extract monetary amounts (simplified)
        amounts = re.findall(r'\$(\d+(?:,\d{3})*(?:\.\d{2})?)', text)
        amounts = [float(amt.replace(',', '')) for amt in amounts]
        
        # Extract years
        years = re.findall(r'\b(20\d{2})\b', text)
        years = [int(year) for year in years]
        
        return {
            'percentages_found': percentages,
            'percentages_count': len(percentages),
            'avg_percentage': np.mean(percentages) if percentages else 0,
            'max_percentage': max(percentages) if percentages else 0,
            'amounts_found': amounts,
            'amounts_count': len(amounts),
            'total_amount': sum(amounts) if amounts else 0,
            'years_mentioned': years,
            'years_count': len(set(years)),
            'most_recent_year': max(years) if years else 0
        }
    
    def extract_text_statistics(self, text: str) -> Dict:
        """
        Extract basic text statistics
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with text statistics
        """
        sentences = sent_tokenize(text)
        words = word_tokenize(text)
        
        # Calculate readability metrics (simplified)
        avg_sentence_length = len(words) / len(sentences) if sentences else 0
        
        return {
            'text_length': len(text),
            'word_count': len(words),
            'sentence_count': len(sentences),
            'avg_sentence_length': avg_sentence_length,
            'unique_words': len(set(word.lower() for word in words)),
            'lexical_diversity': len(set(word.lower() for word in words)) / len(words) if words else 0
        }
    
    def fit_vectorizers(self, texts: List[str]):
        """
        Fit TF-IDF and other vectorizers on the corpus
        
        Args:
            texts: List of preprocessed texts
        """
        logger.info("Fitting TF-IDF vectorizer...")
        
        # TF-IDF with financial domain focus
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            stop_words='english'
        )
        
        try:
            self.tfidf_vectorizer.fit(texts)
            logger.info(f"TF-IDF fitted with {len(self.tfidf_vectorizer.vocabulary_)} features")
        except Exception as e:
            logger.error(f"Error fitting TF-IDF: {e}")
            self.tfidf_vectorizer = None
        
        # Count vectorizer for topic modeling
        self.count_vectorizer = CountVectorizer(
            max_features=100,
            ngram_range=(1, 1),
            min_df=2,
            max_df=0.95,
            stop_words='english'
        )
        
        try:
            count_matrix = self.count_vectorizer.fit_transform(texts)
            
            # LDA topic modeling
            self.lda_model = LatentDirichletAllocation(
                n_components=5,
                random_state=42,
                max_iter=100
            )
            self.lda_model.fit(count_matrix)
            logger.info("Topic modeling (LDA) fitted successfully")
        except Exception as e:
            logger.error(f"Error fitting topic models: {e}")
            self.count_vectorizer = None
            self.lda_model = None
    
    def extract_tfidf_features(self, text: str) -> np.ndarray:
        """
        Extract TF-IDF features for a single text
        
        Args:
            text: Preprocessed text
            
        Returns:
            TF-IDF feature vector
        """
        if not self.tfidf_vectorizer:
            return np.array([])
        
        try:
            return self.tfidf_vectorizer.transform([text]).toarray()[0]
        except Exception as e:
            logger.error(f"Error extracting TF-IDF features: {e}")
            return np.array([])
    
    def extract_topic_features(self, text: str) -> np.ndarray:
        """
        Extract topic modeling features for a single text
        
        Args:
            text: Preprocessed text
            
        Returns:
            Topic distribution vector
        """
        if not self.count_vectorizer or not self.lda_model:
            return np.array([])
        
        try:
            count_vector = self.count_vectorizer.transform([text])
            topic_dist = self.lda_model.transform(count_vector)[0]
            return topic_dist
        except Exception as e:
            logger.error(f"Error extracting topic features: {e}")
            return np.array([])
    
    def process_single_text(self, text: str, fit_vectorizers: bool = False) -> Dict:
        """
        Process a single text and extract all NLP features
        
        Args:
            text: Raw text content
            fit_vectorizers: Whether to fit vectorizers (use for training data)
            
        Returns:
            Dictionary with all extracted features
        """
        # Preprocess text
        cleaned_text = self.preprocess_text(text)
        
        # Extract all features
        features = {}
        
        # Basic text statistics
        features.update(self.extract_text_statistics(text))
        
        # Sentiment analysis
        features.update(self.extract_financial_sentiment(cleaned_text))
        
        # Named entity recognition
        entity_features = self.extract_named_entities(text)
        features.update({
            'named_entities_count': entity_features['named_entities_count'],
            'financial_entities_count': entity_features['financial_entities_count']
        })
        
        # Report type classification
        features.update(self.classify_report_type(cleaned_text))
        
        # Numerical features
        features.update(self.extract_numerical_features(text))
        
        # Store cleaned text and entities for later use
        features['cleaned_text'] = cleaned_text
        features['named_entities'] = entity_features['named_entities']
        features['financial_entities'] = entity_features['financial_entities']
        
        return features
    
    def process_corpus(self, df: pd.DataFrame, text_column: str = 'content') -> pd.DataFrame:
        """
        Process entire corpus and extract features for all documents
        
        Args:
            df: DataFrame with text data
            text_column: Name of column containing text
            
        Returns:
            DataFrame with added NLP features
        """
        logger.info(f"Processing corpus of {len(df)} documents...")
        
        # Process all texts
        all_features = []
        cleaned_texts = []
        
        for idx, text in enumerate(df[text_column]):
            logger.info(f"Processing document {idx + 1}/{len(df)}")
            features = self.process_single_text(text)
            all_features.append(features)
            cleaned_texts.append(features['cleaned_text'])
        
        # Fit vectorizers on entire corpus
        logger.info("Fitting vectorizers on corpus...")
        self.fit_vectorizers(cleaned_texts)
        
        # Extract TF-IDF and topic features
        logger.info("Extracting TF-IDF and topic features...")
        for idx, features in enumerate(all_features):
            tfidf_features = self.extract_tfidf_features(features['cleaned_text'])
            topic_features = self.extract_topic_features(features['cleaned_text'])
            
            # Add TF-IDF features
            if len(tfidf_features) > 0:
                for i, val in enumerate(tfidf_features[:50]):  # Top 50 TF-IDF features
                    features[f'tfidf_{i}'] = val
            
            # Add topic features
            if len(topic_features) > 0:
                for i, val in enumerate(topic_features):
                    features[f'topic_{i}'] = val
        
        # Convert to DataFrame
        features_df = pd.DataFrame(all_features)
        
        # Combine with original DataFrame
        result_df = pd.concat([df.reset_index(drop=True), features_df], axis=1)
        
        logger.info(f"NLP processing completed. Added {len(features_df.columns)} new features.")
        
        return result_df
    
    def get_feature_importance(self, df: pd.DataFrame, target_column: str = 'price_change_percentage') -> pd.DataFrame:
        """
        Calculate correlation between NLP features and target variable
        
        Args:
            df: DataFrame with features and target
            target_column: Name of target column
            
        Returns:
            DataFrame with feature importance scores
        """
        # Select only numerical features for correlation analysis
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        numerical_cols = [col for col in numerical_cols if col != target_column]
        
        if target_column not in df.columns:
            logger.warning(f"Target column '{target_column}' not found")
            return pd.DataFrame()
        
        # Calculate correlations
        correlations = []
        for col in numerical_cols:
            try:
                corr = df[col].corr(df[target_column])
                if not np.isnan(corr):
                    correlations.append({
                        'feature': col,
                        'correlation': corr,
                        'abs_correlation': abs(corr)
                    })
            except Exception as e:
                logger.warning(f"Could not calculate correlation for {col}: {e}")
        
        # Sort by absolute correlation
        importance_df = pd.DataFrame(correlations)
        if not importance_df.empty:
            importance_df = importance_df.sort_values('abs_correlation', ascending=False)
        
        return importance_df


# Main function to integrate with Stage 1
def main():
    """
    Main function to demonstrate NLP pipeline integration
    """
    # Load processed data from Stage 1
    try:
        df = pd.read_csv("training_dataset_thyao.csv")
        logger.info(f"Loaded {len(df)} records from training dataset")
    except FileNotFoundError:
        logger.error("training_dataset_thyao.csv not found. Please run Stage 1 first.")
        return None
    
    # Initialize NLP processor
    nlp_processor = FinancialNLPProcessor()
    
    # Process the corpus
    enhanced_df = nlp_processor.process_corpus(df, text_column='content')
    
    # Calculate feature importance
    importance_df = nlp_processor.get_feature_importance(enhanced_df)
    
    # Save results
    enhanced_df.to_csv("nlp_enhanced_dataset.csv", index=False)
    if not importance_df.empty:
        importance_df.to_csv("feature_importance.csv", index=False)
    
    # Print summary
    print("\n=== NLP PROCESSING SUMMARY ===")
    print(f"Original features: {len(df.columns)}")
    print(f"Enhanced features: {len(enhanced_df.columns)}")
    print(f"New NLP features added: {len(enhanced_df.columns) - len(df.columns)}")
    
    if not importance_df.empty:
        print("\n=== TOP 10 MOST IMPORTANT FEATURES ===")
        print(importance_df.head(10)[['feature', 'correlation']].to_string(index=False))
    
    print(f"\nEnhanced dataset saved to 'nlp_enhanced_dataset.csv'")
    print(f"Feature importance saved to 'feature_importance.csv'")
    
    return enhanced_df, nlp_processor


if __name__ == "__main__":
    enhanced_df, nlp_processor = main()