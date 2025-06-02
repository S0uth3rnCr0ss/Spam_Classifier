
import pickle 
from flask import Flask,render_template,request
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string

# Creating Flask App

app = Flask(__name__)

nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

# Text Transformation
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    y = []
    
    for i in text:
        if i.isalnum(): # checks if alphanumeric
            y.append(i)
            
    text = y[:]
    y.clear()
    
    for i in text: #Remove stopwords and punctiations
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    
    text = y[:]
    y.clear()
    
    for i in text: # Stemming
        y.append(ps.stem(i))
            
    return " ".join(y)# Preprocessed text
    
# Predict the spam or ham
def predict_spam(message):
    
    #preprocessing
    transform_sms = transform_text(message)
    
    #Vectorize 
    vector_input = tfidf.transform([transform_sms])
    
    #predict the model
    result = model.predict(vector_input)[0]
    
    return result

# Routes
@ app.route('/',methods=['GET']) #homepage
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST']) #   Prediction page
def predict():
    if request.method == 'POST':
        input_sms = request.form['message']
        result = predict_spam(input_sms)
        return render_template('index.html',result = result)



if __name__ == "__main__":
    tfidf = pickle.load(open('vectorizer.pkl','rb'))
    model = pickle.load(open('model.pkl','rb'))
    app.run(host='0.0.0.0',debug=True)
    
    
    
    
    
    
# from flask import Flask, render_template, request
# import nltk
# from nltk.corpus import stopwords
# from nltk.stem.porter import PorterStemmer
# import pickle
# import string

# app = Flask(__name__)

# nltk.download('punkt')
# nltk.download('stopwords')

# ps = PorterStemmer()

# def transform_text(text):
#     text = text.lower()
#     text = nltk.word_tokenize(text)

#     y = []
#     for i in text:
#         if i.isalnum():
#             y.append(i)

#     text = y[:]
#     y.clear()

#     for i in text:
#         if i not in stopwords.words('english') and i not in string.punctuation:
#             y.append(i)

#     text = y[:]
#     y.clear()

#     for i in text:
#         y.append(ps.stem(i))

#     return " ".join(y)

# def predict_spam(message):
#     # Preprocess
#     transformed_sms = transform_text(message)
#     # Vectorize
#     vector_input = tfidf.transform([transformed_sms])
#     # Predict
#     result = model.predict(vector_input)[0]
#     return result

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     if request.method == 'POST':
#         input_sms = request.form['message']
#         result = predict_spam(input_sms)
#         return render_template('index.html', result=result)  # Pass 'result' to the template


# if __name__ == '__main__':
#     tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
#     model = pickle.load(open('model.pkl', 'rb'))
#     app.run(host='0.0.0.0',debug=True)


