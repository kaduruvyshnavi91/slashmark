import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import nltk
import re

# Download NLTK resources (if not already downloaded)
nltk.download('punkt')

# Load the dataset
data = pd.read_csv('black_friday_sales.csv')  # Replace with your dataset path

# Preprocess the text data
def preprocess_text(text):
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Tokenize the text
    tokens = nltk.word_tokenize(text)
    return ' '.join(tokens)

# Apply preprocessing
data['cleaned_description'] = data['product_description'].apply(preprocess_text)

# Convert text data to numerical features using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['cleaned_description'])
y = data['sales']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse:.2f}')

# Display predictions
predictions_df = pd.DataFrame({'Actual Sales': y_test, 'Predicted Sales': y_pred})
print(predictions_df)