import pandas as pd
from transformers import pipeline
import tkinter as tk

def load_data(file_path):
    data = pd.read_csv(file_path)
    data['queries'] = data['queries'].apply(lambda x: x.lower())
    return data

csv_file_path = 'wings_of_fire_1.csv'
data = load_data(csv_file_path)

models = [
    pipeline('text-generation', model='EleutherAI/gpt-neo-1.3B'),
    pipeline('text-generation', model='EleutherAI/gpt-neo-2.7B')
]

def find_best_answer(query, data, models):
    best_answer = None
    best_similarity = 0
    for model in models:
        response = model(query, max_length=1024, do_sample=True, temperature=0.5)[0]['generated_text']
        similarity = 1
        for q in data['queries']:
            sim = model(q, response, max_length=1024, do_sample=True, temperature=0.5)[0]['generated_text']
            if sim > similarity:
                similarity = sim
        if similarity > best_similarity:
            best_similarity = similarity
            best_answer = response
    return best_answer, best_similarity

def on_send_click():
    global data
    query = user_input.get()
    if query:
        response, similarity = find_best_answer(query, data, models)
        chat_history.insert(tk.END, f'You: {query}\n')
        chat_history.insert(tk.END, f'Bot: {response} (Similarity: {similarity})\n')
        user_input.delete(0, tk.END)

def on_exit_click():
    root.destroy()

root = tk.Tk()
root.title('Chatbot')

chat_history = tk.Text(root, wrap=tk.WORD)
chat_history.pack(padx=10, pady=10)

user_input = tk.Entry(root, width=50)
user_input.pack(padx=10, pady=10)

send_button = tk.Button(root, text='Send', command=on_send_click)
send_button.pack(padx=10, pady=10)

exit_button = tk.Button(root, text='Exit', command=on_exit_click)
exit_button.pack(padx=10, pady=10)

root.mainloop()
