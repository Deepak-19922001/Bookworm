import pandas as pd
import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# nltk.download()
# load dataset
df = pd.read_csv('wings_of_fire_1.csv')
# preprocess dataset
lemmatizer = WordNetLemmatizer()
nltk.download('punkt')
nltk.download('wordnet')
def get_wordnet_pos(word):
    """Map POS tag to first character used by WordNetLemmatizer"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)
def preprocess_text(text):
    tokens = nltk.word_tokenize(text.lower())
    lemmatized = [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in tokens]
    return ' '.join(lemmatized)
df['processed_queries'] = df['queries'].apply(preprocess_text)
# build model
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['processed_queries'])
cosine_sim = cosine_similarity(tfidf_matrix)
def get_top_n_similar(query, n=2):
    query_tfidf = vectorizer.transform([preprocess_text(query)])
    cosine_similarities = cosine_similarity(query_tfidf, tfidf_matrix).flatten()
    related_docs_indices = [i for i in cosine_similarities.argsort()[::-1]]
    return [(df['answers'].iloc[i], cosine_similarities[i]) for i in related_docs_indices][:n]
# deploy chatbot
prev_ans=""
while True:
    user_input = input("User: ")
    if user_input.lower() == 'quit':
        break
    elif user_input.lower()=='more' :
        user_input=prev_ans

    response, score = get_top_n_similar(user_input)[0]
    prev_ans=response
    if score<0.5:
        res=input("I'm not sure of the answer here, the answer i have might be wrong. You still want to see? Y/N  : ")
        if res == 'N':
            continue
    else:
        print(f"Bot ({score:.2f}): {response}")