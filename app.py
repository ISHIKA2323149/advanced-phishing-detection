from flask import Flask, request, render_template, jsonify
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import nltk

# Initialize Flask app
app = Flask(__name__)

# Load pre-trained tokenizer and model
model_path = './distilbert_phishing_model'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(nltk.corpus.stopwords.words('english'))

# Preprocessing function
def preprocess_text(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    return ' '.join(tokens)

# Define the route for the index page
@app.route('/')
def index():
    return render_template('index.html')

# Define the route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the request
    data = request.get_json()
    email = data.get('email')
    url = data.get('url')

    # Preprocess input and make prediction
    input_text = f"Email: {preprocess_text(email)} URL: {url}"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding="max_length", max_length=256).to(model.device)
    outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()

    # Return result
    result = "Phishing" if prediction == 1 else "Legitimate"
    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)
