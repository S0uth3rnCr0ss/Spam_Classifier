import pickle 
from flask import Flask, render_template, request
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
import os

# Creating Flask App
app = Flask(__name__)

# Ensure custom nltk data path is used
nltk_data_path = os.path.join(os.path.dirname(__file__), 'nltk_data')
nltk.data.path.append(nltk_data_path)

# Download resources if not present
if not os.path.exists(os.path.join(nltk_data_path, 'tokenizers/punkt')):
    nltk.download('punkt', download_dir=nltk_data_path)
if not os.path.exists(os.path.join(nltk_data_path, 'corpora/stopwords')):
    nltk.download('stopwords', download_dir=nltk_data_path)

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))  # cache stopwords for efficiency

# Text Transformation
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    y = []
    
    for i in text:
        if i.isalnum():
            y.append(i)
    
    text = y[:]
    y.clear()
    
    for i in text:
        if i not in stop_words and i not in string.punctuation:
            y.append(ps.stem(i))
            
    return " ".join(y)

# Predict the spam or ham
def predict_spam(message):
    transform_sms = transform_text(message)
    vector_input = tfidf.transform([transform_sms])
    result = model.predict(vector_input)[0]
    return result

# Routes
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        input_sms = request.form['message']
        result = predict_spam(input_sms)
        return render_template('index.html', result=result)

# Load model and run app
if __name__ == "__main__":
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
    app.run(host='0.0.0.0', debug=True)
