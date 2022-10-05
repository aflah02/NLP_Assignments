import pandas as pd
from preprocess_text import *
import pickle
from tqdm import tqdm
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import pipeline

df = pd.read_csv('A1_dataset.csv')

text = df['TEXT'].to_list()
def preprocess_text(text):
    text = lowercase_text(text)
    text = remove_url_html(text)
    text = remove_users(text)
    text = remove_punctuations(text)
    text = remove_whitespaces(text)
    text = tokenization(text)
    text = spelling_correction(text, 'textblob')
    return text
df['preprocessed_text'] = df['TEXT'].apply(preprocess_text)
preprocess_text = df['preprocessed_text'].to_list()

unigram_counts = {}
for sentence in preprocess_text:
    for word in sentence:
        if word in unigram_counts:
            unigram_counts[word] += 1
        else:
            unigram_counts[word] = 1

sid = SentimentIntensityAnalyzer()
ls_word_sentiment_vader = []
for word in tqdm(unigram_counts):
    ls_word_sentiment_vader.append((word, sid.polarity_scores(word)['compound']))
with open('ls_word_sentiment_vader.pickle', 'wb') as f:
    pickle.dump(ls_word_sentiment_vader, f)

hf_sentiment_model = pipeline('sentiment-analysis')
ls_word_sentiment_hf = []
for word in tqdm(unigram_counts):
    hf_res = hf_sentiment_model(word)
    score = hf_res[0]['score']
    pos_neg = hf_res[0]['label']
    if pos_neg == 'NEGATIVE':
        score = -score
    ls_word_sentiment_hf.append((word,score))
with open('ls_word_sentiment_hf.pickle', 'wb') as f:
    pickle.dump(ls_word_sentiment_hf, f)
