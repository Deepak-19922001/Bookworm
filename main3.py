import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# load the CSV file into a pandas DataFrame
df = pd.read_csv('wings_of_fire_1 (1).csv')

# split the data into training and testing sets
train_df = df.sample(frac=0.8, random_state=42)
test_df = df.drop(train_df.index)

# create a CountVectorizer to convert the text into a matrix of word counts
vectorizer = CountVectorizer()

# fit the vectorizer to the training data and transform both the training and testing data
train_X = vectorizer.fit_transform(train_df['queries'])
train_y = train_df['answers']
test_X = vectorizer.transform(test_df['queries'])
test_y = test_df['answers']

# create a Multinomial Naive Bayes model and train it on the training data
model = MultinomialNB()
model.fit(train_X, train_y)

# test the model on the testing data
accuracy = model.score(test_X, test_y)
print(f'Test accuracy: {accuracy:.2f}')

# define a function to generate responses based on user input
def get_response(user_input):
    input_X = vectorizer.transform([user_input])
    prediction = model.predict(input_X)
    return prediction[0]

# run the chatbot
while True:
    user_input = input('User: ')
    response = get_response(user_input)
    print(f'Bot: {response}')
