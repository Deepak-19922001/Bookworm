import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

from main3 import get_response

# Load the data
data = pd.read_csv('wings_of_fire_1 (1).csv')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data['queries'], data['answers'], test_size=0.2, random_state=42)

# Vectorize the input data
vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

# Train the Naive Bayes classifier
nb_clf = MultinomialNB().fit(X_train_tfidf, y_train)

# Train the Support Vector Machines classifier
svm_clf = SVC(kernel='linear', C=1, gamma='auto').fit(X_train_tfidf, y_train)

# Train the Decision Trees classifier
dt_clf = DecisionTreeClassifier().fit(X_train_tfidf, y_train)

# Evaluate the performance of the classifiers on the test set
X_test_counts = vectorizer.transform(X_test)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)

nb_pred = nb_clf.predict(X_test_tfidf)
nb_accuracy = accuracy_score(y_test, nb_pred)

svm_pred = svm_clf.predict(X_test_tfidf)
svm_accuracy = accuracy_score(y_test, svm_pred)

dt_pred = dt_clf.predict(X_test_tfidf)
dt_accuracy = accuracy_score(y_test, dt_pred)

print('Naive Bayes classifier accuracy:', nb_accuracy)
print('Support Vector Machines classifier accuracy:', svm_accuracy)
print('Decision Trees classifier accuracy:', dt_accuracy)

# Print classification report for each classifier
print('\nNaive Bayes classifier report:\n', classification_report(y_test, nb_pred))
print('Support Vector Machines classifier report:\n', classification_report(y_test, svm_pred))
print('Decision Trees classifier report:\n', classification_report(y_test, dt_pred))

while True:
    user_input = input('User: ')
    response = get_response(user_input)
    print(f'Bot: {response}')
