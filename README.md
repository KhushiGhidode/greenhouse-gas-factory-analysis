# greenhouse-gas-factory-analysis
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv('tripadvisor_hotel_reviews.csv')

# Data Preprocessing
def clean_data(review):
    no_punc = re.sub(r'[^\w\s]', '', review)
    no_digits = ''.join([i for i in no_punc if not i.isdigit()])
    return no_digits

df['Review'] = df['Review'].apply(clean_data)

# Feature Extraction using TF-IDF
tfidf = TfidfVectorizer(strip_accents=None, lowercase=False, preprocessor=None)
X = tfidf.fit_transform(df['Review'])
y = df['Sentiment']  # Assuming 'Sentiment' is the target variable

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Model Training
lr = LogisticRegression(solver='liblinear')
lr.fit(X_train, y_train)

# Predictions
preds = lr.predict(X_test)

# Model Evaluation
accuracy = accuracy_score(preds, y_test)
print("Accuracy Score: ", accuracy)
