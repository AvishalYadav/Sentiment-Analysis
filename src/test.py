import unittest
from sentiment_analysis import SentimentAnalyzer
import pandas as pd
import tempfile
import os

class TestSentimentAnalysis(unittest.TestCase):
    def setUp(self):
        self.analyzer = SentimentAnalyzer()
        
        # Create temporary test data
        self.test_data = pd.DataFrame({
            'cleaned_text': [
                "This product is amazing! I absolutely love it.",
                "I hate this, it's terrible.",
                "The weather is okay today.",
            ]
        })
        
        # Create temporary file
        self.temp_dir = tempfile.mkdtemp()
        self.temp_file = os.path.join(self.temp_dir, 'test_data.csv')
        self.test_data.to_csv(self.temp_file, index=False)

    def test_sentiment_analysis(self):
        # Test individual sentiment analysis
        self.assertEqual(self.analyzer.analyze_sentiment("This is great!"), "positive")
        self.assertEqual(self.analyzer.analyze_sentiment("This is terrible!"), "negative")
        self.assertEqual(self.analyzer.analyze_sentiment("This is okay."), "neutral")

    def test_batch_processing(self):
        # Test batch processing
        result_data = self.analyzer.process_sentiments(self.temp_file, force_reanalysis=True)
        self.assertIn('sentiment', result_data.columns)
        self.assertEqual(len(result_data), len(self.test_data))

    def test_empty_input(self):
        # Test empty input handling
        self.assertEqual(self.analyzer.analyze_sentiment(""), "neutral")
        self.assertEqual(self.analyzer.analyze_sentiment(" "), "neutral")

    def tearDown(self):
        # Clean up temporary files
        os.remove(self.temp_file)
        os.rmdir(self.temp_dir)

if __name__ == "__main__":
    unittest.main(verbosity=2)
