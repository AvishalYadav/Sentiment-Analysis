import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Dataset path
FILE_PATH = r"E:\AI-Powered Customer Sentiment Analytics and Forecasting\data\cleaned_amazon_reviews.csv"

class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
    
    def clean_text(self, text):
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove stopwords
        words = word_tokenize(text)
        text = ' '.join([word for word in words if word not in self.stop_words])
        
        return text

    def preprocess_data(self, filepath):
        try:
            logger.info("Loading data...")
            data = pd.read_csv(filepath)
            
            logger.info("Cleaning text data...")
            data.dropna(subset=['reviewText'], inplace=True)
            data['cleaned_text'] = data['reviewText'].apply(self.clean_text)
            
            # Remove empty reviews after cleaning
            data = data[data['cleaned_text'].str.strip().astype(bool)]
            
            logger.info("Saving preprocessed data...")
            data.to_csv(filepath, index=False)
            logger.info("Data preprocessing complete!")
            
            return data
            
        except Exception as e:
            logger.error(f"Error preprocessing data: {e}")
            raise

if __name__ == "__main__":
    preprocessor = TextPreprocessor()
    preprocessor.preprocess_data(FILE_PATH)
