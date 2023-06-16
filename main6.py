import pandas as pd
import tensorflow as tf
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras.saving.hdf5_format import save_attributes_to_hdf5_group

# Load the CSV file
data = pd.read_csv('wings_of_fire_1 (1).csv')

# Encode the labels
label_encoder = LabelEncoder()
data['encoded_answers'] = label_encoder.fit_transform(data['answers'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data['queries'], data['encoded_answers'], test_size=0.2, random_state=42)

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = TFAutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_encoder.classes_))

# Tokenize the data
train_encodings = tokenizer(X_train.tolist(), truncation=True, padding=True)
test_encodings = tokenizer(X_test.tolist(), truncation=True, padding=True)

# Convert the data to TensorFlow format
train_dataset = tf.data.Dataset.from_tensor_slices((
    dict(train_encodings),
    y_train
)).shuffle(1000).batch(8)

test_dataset = tf.data.Dataset.from_tensor_slices((
    dict(test_encodings),
    y_test
)).batch(8)

# Train the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
model.fit(train_dataset, epochs=10, validation_data=test_dataset)

# Function to get the chatbot's response
def get_response(user_input):
    input_encoding = tokenizer(user_input, truncation=True, padding=True, return_tensors='tf')
    logits = model(input_encoding)[0]
    predicted_class = tf.argmax(logits, axis=-1).numpy()[0]
    return label_encoder.inverse_transform([predicted_class])[0]

# Example usage
user_input = "What was the profession of Kalam's father?"
print(get_response(user_input))
