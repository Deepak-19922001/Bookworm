import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from main3 import get_response

# Load data from CSV file
data = pd.read_csv('wings_of_fire_1 (1).csv')

# Split data into training and testing sets
train_data = data.sample(frac=0.8, random_state=42)
test_data = data.drop(train_data.index)

# Define the pipeline for the classifier
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('nb', MultinomialNB())
])

# Train the classifier
pipeline.fit(train_data['queries'], train_data['answers'])

# Evaluate the classifier on the testing set
accuracy = pipeline.score(test_data['queries'], test_data['answers'])
print("Accuracy:", accuracy)

while True:
    user_input = input('User: ')
    response = get_response(user_input)
    print(f'Bot: {response}')
