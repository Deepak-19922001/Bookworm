import pandas as pd
import spacy
import re

# Load the book text data
with open('wings_of_fire.txt', 'r') as f:
    book_text = f.read()

# Split the book text into chapters
chapters = book_text.split('\n\n\n\n\n')

# Preprocess the chapter text data
def preprocess(text):
    # Remove newlines and extra whitespaces
    text = re.sub('\n', ' ', text)
    text = re.sub('\s+', ' ', text)
    # Remove chapter headings
    text = re.sub('^CHAPTER [IVXLCDM]+', '', text)
    text = re.sub('^[IVXLCDM]+', '', text)
    # Remove page numbers and other metadata
    text = re.sub('[0-9]+', '', text)
    text = re.sub('^[A-Z]+\s?[A-Z]*\s?[A-Z]*\s?[A-Z]*\s?[A-Z]*\s?[A-Z]*\s?[A-Z]*\s?[A-Z]*\s?[A-Z]*', '', text)
    # Remove non-alphanumeric characters
    text = re.sub('[^0-9a-zA-Z\s]+', '', text)
    return text

# Apply the preprocessing function to each chapter
chapters = [preprocess(chapter) for chapter in chapters]

# Load the stop words
nlp = spacy.load('en_core_web_sm')
stop_words = nlp.Defaults.stop_words

# Define a function to tokenize and filter the text
def tokenize(text):
    doc = nlp(text)
    tokens = [token.lemma_.lower() for token in doc if token.is_alpha and not token.is_stop and token.lemma_.lower() not in stop_words]
    return tokens

# Apply the tokenization function to each chapter
chapters = [tokenize(chapter) for chapter in chapters]

# Create a Pandas DataFrame to store the chapter data
df = pd.DataFrame({'chapter': chapters})
