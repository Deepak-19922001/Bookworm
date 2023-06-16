import pandas as pd
import tkinter as tk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neural_network import MLPClassifier
from textblob import TextBlob
import openai
import tkinter.messagebox as messagebox
from tkinter import simpledialog
import tkinter.ttk as ttk
import time

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


def add_data_to_csv(query, answer):
    global data, csv_file_path, clf, vectorizer
    new_data = pd.DataFrame({'queries': [query.lower()], 'answers': [answer]})
    data = pd.concat([data, new_data], ignore_index=True)
    data.to_csv(csv_file_path, index=False)
    messagebox.showinfo("Retraining", "Model is retraining...")
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(data['queries'])
    clf.fit(tfidf_matrix, data['answers'])

def find_best_answer(query, data, clf, context):
    query_vector = vectorizer.transform([query.lower()])
    predicted_answer = clf.predict(query_vector)[0]
    if predicted_answer == 'I don\'t know':
        if query.lower() in context:
            predicted_answer = context[query.lower()]
        else:
            predicted_answer = data['answers'][0]
    context[query.lower()] = predicted_answer
    accuracy = round(cosine_similarity(query_vector, vectorizer.transform([predicted_answer]))[0][0], 2)
    return predicted_answer, accuracy

def handle_greetings(query):
    greetings = ['hi', 'hello', 'hey', 'greetings']
    farewells = ['bye', 'goodbye', 'see you later', 'farewell']

    if query.lower() in greetings:
        return "Hello! How can I help you today?"
    elif query.lower() in farewells:
        return "Goodbye! Have a great day!"
    else:
        return None

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

def on_add_data_click():
    new_query = simpledialog.askstring("Add Data", "Enter the query:")
    new_answer = simpledialog.askstring("Add Data", "Enter the answer:")
    if new_query and new_answer:
        add_data_to_csv(new_query, new_answer)
        messagebox.showinfo("Add Data", "Data added successfully!")


import openai
import requests
import json

import openai
import requests
import json

def generate_response(query):
    url = 'https://api.openai.com/v1/engines/davinci-codex/completions'
    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer YOUR_API_KEY'
    }
    data = {
        'prompt': f'User: {query}\nBot:',
        'max_tokens': 1024,
        'temperature': 0.5,
        'n': 1,
        'stop': None
    }
    response = requests.post(url, headers=headers, data=json.dumps(data))
    response_text = json.loads(response.text)
    if 'choices' in response_text and len(response_text['choices']) > 0:
        return response_text['choices'][0]['text'].strip().split('Bot: ')[1]
    else:
        return "I'm sorry, I couldn't understand your query. Please try again."



def on_send_click():
    global data
    query = user_input.get()
    if query:
        greeting_response = handle_greetings(query)
        if greeting_response:
            response = greeting_response
            accuracy = "N/A"
        else:
            response, accuracy = find_best_answer(query, data, clf, context)
        chat_history.insert(tk.END, f'You: {query}\n')
        chat_history.insert(tk.END, f'Bot: {response} (Accuracy: {accuracy})\n')
        sentiment = get_sentiment(query)
        adjusted_response = adjust_response(response, sentiment)
        chat_history.insert(tk.END, f'Bot: {adjusted_response}\n')
        feedback = messagebox.askyesno('Feedback', 'Was the response helpful?')
        if feedback:
            clf.partial_fit(vectorizer.transform([query.lower()]), [response])
        else:
            chatgpt_response = generate_response(query)
            chat_history.insert(tk.END, f'Bot: {chatgpt_response} (ChatGPT)\n')
            clf.partial_fit(vectorizer.transform([query.lower()]), [chatgpt_response])
            data = data.append({'queries': query.lower(), 'answers': chatgpt_response}, ignore_index=True)
            data.to_csv(csv_file_path, index=False)
        user_input.delete(0, tk.END)

def clear_chat_history():
    global chat_history
    chat_history.delete('1.0', tk.END)

def on_exit_click():
    root.destroy()

root = tk.Tk()
root.title('Book Buddy: Your Personal Chatbot for All Things related to "Wings of Fire: An Autobiography"')

chat_history = tk.Text(root, wrap=tk.WORD)
chat_history.pack(padx=10, pady=10)
chat_history.insert(tk.END, f'Bot: Hello! How can I help you today?\n')


user_input = tk.Entry(root, width=50)
user_input.pack(padx=10, pady=10)

send_button = tk.Button(root, text='Send', command=on_send_click)
send_button.pack(padx=10, pady=10)

add_data_button = tk.Button(root, text='Add Data', command=on_add_data_click)
add_data_button.pack(padx=10, pady=10)

clear_button = tk.Button(root, text='Clear', command=clear_chat_history)
clear_button.pack(padx=10, pady=10)

exit_button = tk.Button(root, text='Exit', command=on_exit_click)
exit_button.pack(padx=10, pady=10)

root.mainloop()
