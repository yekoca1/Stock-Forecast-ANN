# nlp_features.py
# Module: NLP Feature Extraction for KAP Disclosures
# 
# Environment:
#   - Python 3.9+
#   - Dependencies (install with pip):
#       pandas
#       scikit-learn
#       nltk
#       langdetect
#
# Usage Example:
#   from kap_scraper import KAPScraper
#   from nlp_features import NLPFeatures
#   scraper = KAPScraper('THYAO')
#   df_news = scraper.fetch_announcements(start_date='2024-01-01', end_date='2024-12-31')
#   nlp = NLPFeatures()
#   df_features = nlp.transform(df_news)
#   # df_features contains TF-IDF, sentiment, keyword counts, metadata

import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from langdetect import detect

# Ensure NLTK Turkish stopwords are downloaded
import nltk
nltk.download('punkt')
nltk.download('stopwords')


class NLPFeatures:
    """
    Extracts NLP-based features from a DataFrame of disclosures.

    Methods:
        fit(corpus): learn TF-IDF vocabulary
        transform(df, **kwargs): return feature-augmented DataFrame
    """
    def __init__(self, max_features: int = 500):
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=(1, 2),
            stop_words=stopwords.words('turkish'),
            token_pattern=r"(?u)\b\w+\b"
        )
        # Example keyword list for Turkish economy
        self.keywords = ['faiz', 'enflasyon', 'kur', 'bilanço', 'temettü']

    def clean_text(self, text: str) -> str:
        """Lowercase, remove non-letters, extra whitespace."""
        text = text.lower()
        text = re.sub(r'[^a-zçğıöşü\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def fit(self, corpus: pd.Series):
        """Learn TF-IDF vocabulary from corpus of raw texts."""
        cleaned = corpus.apply(self.clean_text)
        self.vectorizer.fit(cleaned)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform disclosures DataFrame into feature DataFrame.
        Expects df.columns: ['date', 'doc_type', 'title', 'url', 'text']
        Returns: df_features with original date + extracted features
        """
        df = df.copy()
        # Clean text
        df['clean_text'] = df['text'].apply(self.clean_text)

        # TF-IDF features
        tfidf_matrix = self.vectorizer.transform(df['clean_text'])
        tfidf_df = pd.DataFrame(
            tfidf_matrix.toarray(),
            columns=[f"tfidf_{w}" for w in self.vectorizer.get_feature_names_out()],
            index=df.index
        )

        # Keyword counts
        for kw in self.keywords:
            df[f'kw_count_{kw}'] = df['clean_text'].apply(lambda x: x.split().count(kw))

        # Sentiment proxy: simple polarity by counting positive vs negative words
        # (Placeholder: user to integrate Turkish sentiment lexicon)
        df['sentiment_score'] = df['clean_text'].apply(lambda x: 0)

        # Metadata: doc length (#words)
        df['doc_length'] = df['clean_text'].apply(lambda x: len(x.split()))
        df['is_pdf'] = (df['doc_type'] == 'pdf').astype(int)

        # Combine all features
        df_features = pd.concat(
            [df[['date', 'title', 'url', 'doc_length', 'is_pdf']].reset_index(drop=True),
             tfidf_df.reset_index(drop=True),
             df[[f'kw_count_{kw}' for kw in self.keywords]].reset_index(drop=True),
             df[['sentiment_score']].reset_index(drop=True)
            ],
            axis=1
        )
        return df_features

# Example test to ensure transform produces expected columns
if __name__ == '__main__':
    # Dummy test
    data = {
        'date': [pd.to_datetime('2024-01-01')],
        'doc_type': ['html'],
        'title': ['Test'],
        'url': ['http://example.com'],
        'text': ['Faiz ve enflasyon oranı arttı.']
    }
    df_news = pd.DataFrame(data)
    nlp = NLPFeatures(max_features=10)
    nlp.fit(df_news['text'])
    df_feat = nlp.transform(df_news)
    print(df_feat.head())
