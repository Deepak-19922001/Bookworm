import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# Load the CSV file
data = pd.read_csv('wings_of_fire_1 (1).csv')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data['queries'], data['answers'], test_size=0.2, random_state=42)

# Create a pipeline for the TfidfVectorizer and LinearSVC
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('classifier', LinearSVC())
])

# Train the model
pipeline.fit(X_train, y_train)

# Test the model
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

# Function to get the chatbot's response
def get_response(user_input):
    return pipeline.predict([user_input])[0]

# Example usage
user_input = "What was the profession of APJ Abdul Kalam's father?"
print(get_response(user_input))
