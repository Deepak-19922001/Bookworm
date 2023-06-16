import numpy as np
import pandas as pd
import sklearn
from sklearn import linear_model, neighbors
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier
import tkinter as tk


# Load the data
#f = pd.read_csv("wings_of_fire.csv")
df = pd.read_csv("Book1.csv", names=["queries", "answers", "y"])

# Create the models
models = []
# Linear and Logistic Regression
for algorithm in ["LinearRegression", "LogisticRegression"]:
    model = getattr(sklearn.linear_model, algorithm)()
    models.append(model)

# K Nearest Neighbour
model = getattr(sklearn.neighbors, "KNeighborsClassifier")()
models.append(model)

# Decision Tree
model = DecisionTreeClassifier()
models.append(model)
# "random_forests", , , "artificial_neural_networks", "deep_learning",

# Random Forest
model = RandomForestClassifier()
models.append(model)

# Support Vector Machine
model = SVC()
models.append(model)

# Naive Bayer's
model = GaussianNB()
models.append(model)

# Artificial Neural Network
model = MLPClassifier()
models.append(model)

# Deep Learning Model
model = MLPClassifier(hidden_layer_sizes=(25, 25, 25, 25, 25))
models.append(model)

# Ensemble Learning Model using Bagging
model = BaggingClassifier(base_estimator=LogisticRegression(), n_estimators=100)
models.append(model)


# Fit the models
for model in models:
    model.fit(df[["queries", "answers"]], df["y"])


# Create the chatbot
class Chatbot:
    def __init__(self):
        self.models = models

    def add_model(self, model):
        self.models.append(model)

    def start(self):
        while True:
            query = input("What can I help you with? ")
            for model in self.models:
                try:
                    answer = model.predict(query)
                    print(answer)
                    break
                except:
                    continue

# Create the UI
root = tk.Tk()

# Create the label
label = tk.Label(root, text="Wings of Fire Chatbot")
label.pack()

# Create the entry box
entry_box = tk.Entry(root)
entry_box.pack()

# Create the button
button = tk.Button(root, text="Submit", command=lambda: chatbot.start())
button.pack()

# Start the chatbot
chatbot = Chatbot()

# Start the UI
root.mainloop()
