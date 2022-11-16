import re
from autocorrect import Speller
import string
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer, PorterStemmer
import nltk
from nltk.tokenize import TweetTokenizer
import numpy as np
from tqdm.notebook import tqdm
tqdm.pandas()
from textblob import TextBlob
# customregexes.py contains all the regex functions used for all parts
from customregexes import *

def tokenization(text):
    twt_tk = TweetTokenizer()
    return twt_tk.tokenize(text)

wordnet_words = []

for word in wn.words():
    wordnet_words.append(word)

words = set(wordnet_words)

# Spelling correction using autocorrect
def spelling_correction_autocorrect_full_sentence(text):
    spell = Speller()
    res = []
    text = " ".join(text)
    text = spell(text)
    return tokenization(text)

# Spelling correction using autocorrect
def spelling_correction_autocorrect_per_token(text):
    spell = Speller()
    res = []
    for token in text:
        res.append(spell(token))
    return res

# Spelling correction using textblob
def spelling_correction_textblob(text):
    text = " ".join(text)
    text = str(TextBlob(text))
    return tokenization(text)

def spelling_correction(text, method ='autocorrect_full'):
    if method == 'autocorrect_full':
        return spelling_correction_autocorrect_full_sentence(text)
    elif method == 'autocorrect_token':
        return spelling_correction_autocorrect_per_token(text)
    elif method == 'textblob':
        return spelling_correction_textblob(text)
    else:
        raise ValueError('Please use a valid method')

#Here we can either do lemmatization or stemming, doing both at the same time will not be useful

def lemmatization_text(text):
    wn = WordNetLemmatizer()
    lemmatized_text = []
    for each in text:
        lemmatized_word = wn.lemmatize(each)
        lemmatized_text.append(lemmatized_word)
    return ' '.join(lemmatized_text)

def stemming_text(text):
    stemmed_text = []
    ps = PorterStemmer()
    for each in text:
        stemmed_word = ps.stem(each)
        stemmed_text.append(stemmed_word)
    return ' '.join(stemmed_text)

def remove_punctuations(text):
    puncts = string.punctuation
    s = ""
    for i in text:
        if i not in puncts:
            s += i
    return s

stop_words = stopwords.words("english")
stop_words_without_punc = []
for word in stop_words:
    stop_words_without_punc.append(remove_punctuations(word))
stop_words = stop_words_without_punc
def remove_stopwords(text):
    new_words = []
    pattern = re.compile(r'\b(' + (r'|'.join(stop_words)) + r')\b\s*')
    for word in text:
        isMatch = pattern.match(word)
        if not isMatch:
            new_words.append(word)
    return new_words

def remove_whitespaces(text):
    return re.sub(r'\s*\s', ' ', text)

def remove_url_html(text):
    urls = findURLsandHTML(text)
    for i in urls:
        text = re.sub(f"{i}", "", text)
    return text

def remove_users(text):
    username = findUsernames(text)
    for i in username:
        text = re.sub(f"{i}", "", text)
    return text

def lowercase_text(text):
    return lowercase(text)

def remove_alphanum(list_of_tokens):
    new_toks = []
    for token in list_of_tokens:
        countNum = 0
        countAlpha = 0
        for char in token:
            if char.isalpha():
                countAlpha += 1
            elif char.isdigit():
                countNum += 1
        if countAlpha > 0 and countNum == 0:
            new_toks.append(token)
    return new_toks
