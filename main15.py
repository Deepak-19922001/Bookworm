import tkinter as tk
import tkinter.messagebox as messagebox
import tkinter.simpledialog as simpledialog
import tkinter.ttk as ttk
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
import time

# Global variables
data = pd.read_csv('wings_of_fire_1.csv')
csv_file_path = 'wings_of_fire_1.csv'
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(data['queries'])
clf = MLPClassifier(hidden_layer_sizes=(8, 8, 8), activation='relu', solver='adam', max_iter=500)

# GUI functions
def on_send_click():
    global data, clf, vectorizer
    # Code to handle user input and generate response

def on_add_data_click():
    global data, clf, vectorizer
    new_query = simpledialog.askstring("Add Data", "Enter the query:")
    new_answer = simpledialog.askstring("Add Data", "Enter the answer:")
    if new_query and new_answer:
        add_data_to_csv(new_query, new_answer)
        messagebox.showinfo("Add Data", "Data added successfully!")

def add_data_to_csv(query, answer):
    global data, csv_file_path, clf, vectorizer
    new_data = pd.DataFrame({'queries': [query.lower()], 'answers': [answer]})
    data = pd.concat([data, new_data], ignore_index=True)
    data.to_csv(csv_file_path, index=False)
    progress_bar = ttk.Progressbar(root, orient='horizontal', length=200, mode='indeterminate')
    progress_bar.pack(padx=10, pady=10)
    progress_bar.start()
    for i in range(100):
        root.update()
        time.sleep(0.01)
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(data['queries'])
    clf.fit(tfidf_matrix, data['answers'])
    progress_bar.stop()
    progress_bar.destroy()

def clear_chat_history():
    global chat_history
    chat_history.delete('1.0', tk.END)

# GUI setup
root = tk.Tk()
root.title('Bookworm Chatbot')
root.geometry('400x500')

# Code to create GUI widgets

root.mainloop()
