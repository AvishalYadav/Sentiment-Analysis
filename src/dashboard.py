import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Use relative path for data
FILE_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'sample_reviews.csv')

class SentimentDashboard:
    def __init__(self):
        self.data = None
        
    def load_data(self, filepath):
        try:
            self.data = pd.read_csv(filepath)
            return True
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False

    def create_sentiment_plot(self):
        sentiment_counts = self.data['sentiment'].value_counts()
        fig = px.pie(values=sentiment_counts.values, 
                    names=sentiment_counts.index,
                    title='Sentiment Distribution')
        return fig

    def run_dashboard(self):
        st.set_page_config(page_title="Sentiment Analysis Dashboard", layout="wide")
        st.title("Customer Sentiment Analysis Dashboard")

        if not self.load_data(FILE_PATH):
            st.error("Failed to load data. Please check the file path and try again.")
            return

        # Sentiment Distribution
        st.subheader("Sentiment Distribution")
        if 'sentiment' not in self.data.columns:
            st.error("The dataset does not contain sentiment analysis. Please run the sentiment analysis first.")
            return

        st.plotly_chart(self.create_sentiment_plot(), use_container_width=True)

        # Sample Reviews
        st.subheader("Sample Reviews")
        sentiment_filter = st.multiselect("Filter by Sentiment", ['positive', 'neutral', 'negative'], default=['positive', 'neutral', 'negative'])
        filtered_data = self.data[self.data['sentiment'].isin(sentiment_filter)]
        st.dataframe(filtered_data[['cleaned_text', 'sentiment']].head(10))

if __name__ == "__main__":
    dashboard = SentimentDashboard()
    dashboard.run_dashboard()
