import pandas as pd
import tkinter as tk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
def load_data(file_path):
    data = pd.read_csv(file_path)
    data['queries'] = data['queries'].apply(lambda x: x.lower())
    return data

csv_file_path = 'wings_of_fire_1 (1).csv'
data = load_data(csv_file_path)

def find_best_answer(query, data, q_values, learning_rate, discount_factor, exploration_rate):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(data['queries'])
    query_vector = vectorizer.transform([query.lower()])
    similarity_scores = cosine_similarity(query_vector, tfidf_matrix)
    best_match_index = similarity_scores.argmax()
    best_match_answer = data['answers'][best_match_index]
    if np.random.uniform(0, 1) < exploration_rate:
        return best_match_answer
    else:
        q_values[query.lower()][best_match_answer] += learning_rate * (1 - q_values[query.lower()][best_match_answer])
        return np.random.choice(list(q_values[query.lower()].keys()), p=list(q_values[query.lower()].values()))

def initialize_q_values(data):
    q_values = {}
    for query in data['queries']:
        q_values[query.lower()] = {}
        for answer in data['answers']:
            q_values[query.lower()][answer] = 0
    return q_values

q_values = initialize_q_values(data)
learning_rate = 0.1
discount_factor = 0.9
exploration_rate = 0.1

def on_send_click():
    query = user_input.get()
    if query:
        response = find_best_answer(query, data, q_values, learning_rate, discount_factor, exploration_rate)
        chat_history.insert(tk.END, f'You: {query}\n')
        chat_history.insert(tk.END, f'Bot: {response}\n')
        reward = int(input('Was the response helpful? (1 for yes, 0 for no): '))
        if reward:
            q_values[query.lower()][response] += learning_rate * (reward + discount_factor * max(q_values[query.lower()].values()) - q_values[query.lower()][response])
        else:
            q_values[query.lower()][response] += learning_rate * (-1 + discount_factor * max(q_values[query.lower()].values()) - q_values[query.lower()][response])
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
