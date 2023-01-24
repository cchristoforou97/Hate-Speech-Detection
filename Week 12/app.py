from flask import Flask, request, render_template

#from clean_data import * 
import numpy as np

import torch
from simpletransformers.classification import ClassificationModel, ClassificationArgs

import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet 
from nltk.corpus import stopwords


def download_nltk_corpus():
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('punkt')


def clean_text(text):
    
    # tokenize
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)
    

    ## lemmatize + lowercase
    lemmatizer = WordNetLemmatizer()
    for word in text.split():
          token = lemmatizer.lemmatize(word.lower(), pos='v')
             
    
    ## remove stopwords
    keep_words = [token for token in tokens if token not in stopwords.words('english')]
    row_text = ' '.join(keep_words)
    row_text = ' '.join([word for word in row_text.split() if len(word)>1])  ## remove one letter words
    row_text = re.sub(r'\w*\d\w*', '', row_text).strip()


    return row_text


app = Flask(__name__)

## load the model
# Load the weights and biases from a file
weights_and_biases = torch.load('pytorch_model.bin')

# Create a new ClassificationModel object using the loaded weights and biases
roberta = ClassificationModel('roberta', 'roberta-base', state_dict=weights_and_biases, use_cuda=False)

@app.route('/')
def home():
	return render_template('Index.html')

@app.route('/predict', methods=['POST'])
def predict():
	
	input = [clean_text(x) for x in request.form.values()]
	prediction = roberta.predict(input)
	
	if prediction[0].item() == 0:
		final_pred = 'Free Speech'
	else:
		final_pred = 'Hate Speech'

	return render_template('Index.html', prediction_text = 'The tweet is classified as {}'.format(final_pred))

if __name__ == '__main__':
	download_nltk_corpus()
	app.run(debug=True)