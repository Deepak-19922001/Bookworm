import pandas as pd
import tkinter as tk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neural_network import MLPClassifier
from textblob import TextBlob
import openai

openai.api_key = "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"

def load_data(file_path):
    data = pd.read_csv(file_path)
    data['queries'] = data['queries'].apply(lambda x: x.lower())
    return data

csv_file_path = 'wings_of_fire_1.csv'
data = load_data(csv_file_path)

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(data['queries'])

clf = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, alpha=0.0001,
                    solver='adam', verbose=10,  random_state=21,tol=0.000000001)
clf.fit(tfidf_matrix, data['answers'])

context = {}

def find_best_answer(query, data, clf, context):
    query_vector = vectorizer.transform([query.lower()])
    predicted_answer = clf.predict(query_vector)[0]
    if predicted_answer == 'I don\'t know':
        if query.lower() in context:
            predicted_answer = context[query.lower()]
        else:
            predicted_answer = data['answers'][0]
    context[query.lower()] = predicted_answer
    return predicted_answer

def get_sentiment(query):
    blob = TextBlob(query)
    return blob.sentiment.polarity

def adjust_response(response, sentiment):
    if sentiment < -0.5:
        return "I'm sorry to hear that. Is there anything I can do to help?"
    elif sentiment < 0:
        return "I understand how you feel. Let me know if there's anything I can do to help."
    elif sentiment < 0.5:
        return "I'm glad to hear that. Is there anything else I can help you with?"
    else:
        return "That's great to hear! Let me know if there's anything else I can help you with."

def generate_response(query):
    response = openai.Completion.create(
        engine="davinci",
        prompt=f"User: {query}\nBot:",
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.5,
    )
    return response.choices[0].text.strip().split('Bot: ')[1]

def on_send_click():
    query = user_input.get()
    if query:
        response = find_best_answer(query, data, clf, context)
        chat_history.insert(tk.END, f'You: {query}\n')
        chat_history.insert(tk.END, f'Bot: {response}\n')
        sentiment = get_sentiment(query)
        adjusted_response = adjust_response(response, sentiment)
        chat_history.insert(tk.END, f'Bot: {adjusted_response}\n')
        reward = int(input('Was the response helpful? (1 for yes, 0 for no): '))
        if reward:
            clf.partial_fit(vectorizer.transform([query.lower()]), [response])
        user_input.delete(0, tk.END)

root = tk.Tk()
root.title('Chatbot')

chat_history = tk.Text(root, wrap=tk.WORD)
chat_history.pack(padx=10, pady=10)

user_input = tk.Entry(root, width=50)
user_input.pack(padx=10, pady=10)

send_button = tk.Button(root, text='Send', command=on_send_click)
send_button.pack(padx=10, pady=10)

root.mainloop()
