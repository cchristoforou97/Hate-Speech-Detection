# -*- coding: utf-8 -*-
"""clean_data.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/147zBWua0d07SVtr_Eh67PTT2PqGs90El
"""

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