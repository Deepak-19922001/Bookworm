import pandas as pd
import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import ssl
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Load dataset
df = pd.read_csv('wings_of_fire_1 (1).csv')

# Preprocess dataset
lemmatizer = WordNetLemmatizer()
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import stopwords

def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def preprocess_text(text):
    tokens = nltk.word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    lemmatized = [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in filtered_tokens]
    return ' '.join(lemmatized)

df['processed_queries'] = df['queries'].apply(preprocess_text)

# Build model
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
tfidf_matrix = vectorizer.fit_transform(df['processed_queries'])
cosine_sim = cosine_similarity(tfidf_matrix)

def get_top_n_similar(query, n=2):
    query_tfidf = vectorizer.transform([preprocess_text(query)])
    cosine_similarities = cosine_similarity(query_tfidf, tfidf_matrix).flatten()
    related_docs_indices = [i for i in cosine_similarities.argsort()[::-1]]
    return [(df['answers'].iloc[i], cosine_similarities[i]) for i in related_docs_indices][:n]

# GUI
def on_submit():
    user_input = user_entry.get()
    if user_input.lower() == 'quit':
        window.destroy()
    elif user_input.lower() == 'more':
        user_input = prev_ans

    response, score = get_top_n_similar(user_input)[0]
    prev_ans = response
    if score < 0.5:
        res = messagebox.askyesno("Low Confidence", "I'm not sure of the answer here, the answer I have might be wrong. you still want to see?")
        if res:
            response_text.set(f"Bot ({score:.2f}): {response}")
    else:
        response_text.set(f"Bot ({score:.2f}): {response}")

def on_clear():
    user_entry.delete(0, tk.END)
    response_text.set("")

def on_quit():
    window.destroy()

window = tk.Tk()
window.title("Chatbot")

frame = ttk.Frame(window, padding="10")
frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

user_entry = ttk.Entry(frame, width=50)
user_entry.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

submit_button = ttk.Button(frame, text="Submit", command=on_submit)
submit_button.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))

response_text = tk.StringVar()
response_label = ttk.Label(frame, textvariable=response_text, wraplength=300)
response_label.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))

clear_button = ttk.Button(frame, text="Clear", command=on_clear)
clear_button.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

quit_button = ttk.Button(frame, text="Quit", command=on_quit)
quit_button.grid(row=2, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))

window.mainloop()