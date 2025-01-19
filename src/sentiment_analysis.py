from transformers import pipeline
import pandas as pd
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the FILE_PATH variable
FILE_PATH = r"E:\AI-Powered Customer Sentiment Analytics and Forecasting\data\cleaned_amazon_reviews.csv"

class SentimentAnalyzer:
    def __init__(self):
        try:
            # Update to use top_k instead of return_all_scores
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                top_k=None  # This replaces return_all_scores=True
            )
        except Exception as e:
            logger.error(f"Error initializing sentiment analyzer: {e}")
            raise

    def analyze_sentiment(self, text):
        try:
            if not isinstance(text, str) or not text.strip():
                return 'neutral'
            
            # Get sentiment scores
            result = self.sentiment_analyzer(text[:512])[0]
            
            # Convert scores to dictionary for easier access
            scores = {item['label']: item['score'] for item in result}
            
            # Define thresholds for more accurate neutral classification
            positive_score = scores.get('POSITIVE', 0)
            negative_score = scores.get('NEGATIVE', 0)
            
            # Adjust thresholds for better neutral detection
            CONFIDENCE_THRESHOLD = 0.7  # Increased threshold
            NEUTRAL_MARGIN = 0.2  # Adjusted margin for neutral classification
            
            # More sophisticated neutral detection
            if (positive_score < CONFIDENCE_THRESHOLD and negative_score < CONFIDENCE_THRESHOLD) or \
               abs(positive_score - negative_score) < NEUTRAL_MARGIN or \
               "okay" in text.lower():  # Special case for "okay"
                return 'neutral'
            elif positive_score > negative_score:
                return 'positive'
            else:
                return 'negative'
            
        except Exception as e:
            logger.error(f"Error analyzing text: {text}\nError: {e}")
            return 'neutral'

    def process_sentiments(self, filepath, force_reanalysis=True, batch_size=32):
        try:
            data = pd.read_csv(filepath)
            if 'sentiment' in data.columns and not force_reanalysis:
                logger.info("Sentiment analysis already exists. Use force_reanalysis=True to overwrite.")
                return data
            
            logger.info("Starting sentiment analysis...")
            
            # Process in batches with progress bar
            sentiments = []
            for i in tqdm(range(0, len(data), batch_size)):
                batch = data['cleaned_text'].iloc[i:i+batch_size]
                batch_sentiments = [self.analyze_sentiment(text) for text in batch]
                sentiments.extend(batch_sentiments)
            
            data['sentiment'] = sentiments
            data.to_csv(filepath, index=False)
            logger.info("Sentiment analysis complete!")
            return data
            
        except Exception as e:
            logger.error(f"Error processing sentiments: {e}")
            raise

if __name__ == "__main__":
    analyzer = SentimentAnalyzer()
    analyzer.process_sentiments(FILE_PATH, force_reanalysis=True)