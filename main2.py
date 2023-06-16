import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Load the CSV file
df = pd.read_csv('Book1.csv')

# Clean the data
df = df.dropna()
df['queries'] = df['queries'].astype('str')
df['answers'] = df['answers'].astype('str')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['queries'], df['answers'], test_size=0.10)

# Train the ML models
gnb = GaussianNB()
gnb.fit(X_train, y_train)

lr = LogisticRegression()
lr.fit(X_train, y_train)

rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)

# Evaluate the ML models
print('GaussianNB accuracy:', gnb.score(X_test, y_test))
print('LogisticRegression accuracy:', lr.score(X_test, y_test))
print('RandomForestClassifier accuracy:', rfc.score(X_test, y_test))

# Create a function to get the answer to a query
def get_answer(query):
    # Check if the query is in the training set
    if query in df['queries']:
        return df['answers'][df['queries'] == query].values[0]

    # If the query is not in the training set, use the ML models to get the answer
    else:
        answers = []
        for model in [gnb, lr, rfc]:
            answers.append(model.predict([query])[0])

        return max(answers, key=answers.count)

# Start a loop to interact with the user
while True:
    # Get the user's query
    query = input('What can I help you with? ')

    # Get the answer to the query
    answer = get_answer(query)

    # Print the answer
    print(answer)

    # Check if the user wants to continue
    continue_query = input('Do you have any other questions? (Y/N) ')
    if continue_query == 'N':
        break
