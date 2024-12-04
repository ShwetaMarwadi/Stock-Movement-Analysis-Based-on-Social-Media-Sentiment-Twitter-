# Stock-Movement-Analysis-Based-on-Social-Media-Sentiment-
Twitter Sentiment Analysis using Machine Learning with Python

This project predicts stock movements using sentiment analysis from the Sentiment140 dataset, comprising 1.6 million tweets. It includes data scraping, preprocessing, sentiment analysis, and machine learning modeling.

1. Features
Scraper: Extracts and preprocesses sentiment data.
Prediction Model: Predicts stock price movements using historical and sentiment features.
Visualization: Displays insights and trends.

2. Importing Dependencies
import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


3. Run the Code
Data Scraping: Run the notebooks/data_scraping.ipynb notebook to scrape and preprocess data.
Preprocessing: Clean and analyze data with notebooks/preprocessing.ipynb.
Model Training: Train the prediction model using notebooks/model_training.ipynb.
Evaluation: Analyze performance in notebooks/results.ipynb.


