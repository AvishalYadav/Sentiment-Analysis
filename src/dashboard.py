import streamlit as st
import pandas as pd
import plotly.express as px
import os

# Configure page
st.set_page_config(page_title="Sentiment Analysis Dashboard", layout="wide")
st.title("Customer Sentiment Analysis Dashboard")

# Load data
try:
    # Try multiple file paths
    file_paths = [
        'data/sample_reviews.csv',
        os.path.join(os.path.dirname(__file__), '..', 'data', 'sample_reviews.csv'),
    ]
    
    data = None
    for path in file_paths:
        if os.path.exists(path):
            data = pd.read_csv(path)
            break
    
    if data is None:
        st.error("Could not find the data file.")
    else:
        # Sentiment Distribution
        st.subheader("Sentiment Distribution")
        sentiment_counts = data['sentiment'].value_counts()
        fig = px.pie(values=sentiment_counts.values, 
                    names=sentiment_counts.index,
                    title='Sentiment Distribution')
        st.plotly_chart(fig, use_container_width=True)
        
        # Sample Reviews
        st.subheader("Sample Reviews")
        sentiment_filter = st.multiselect(
            "Filter by Sentiment", 
            ['positive', 'neutral', 'negative'],
            default=['positive', 'neutral', 'negative']
        )
        
        filtered_data = data[data['sentiment'].isin(sentiment_filter)]
        st.dataframe(filtered_data[['cleaned_text', 'sentiment']].head(10))

except Exception as e:
    st.error(f"Error: {e}")
