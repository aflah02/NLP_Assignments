from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
from preprocess_text import *

def train_and_evaluate(train_sentences, train_labels, test_sentences, test_labels):
    model = make_pipeline(TfidfVectorizer(), MultinomialNB())
    model.fit(train_sentences, train_labels)
    predicted_test_labels = model.predict(test_sentences)
    
    return accuracy_score(test_labels, predicted_test_labels)

def preprocess_text(text, remove_alnum = False):
    text = lowercase_text(text)
    text = remove_url_html(text)
    text = remove_users(text)
    text = remove_punctuations(text)
    text = remove_whitespaces(text)
    if remove_alnum:
        text = remove_alphanum(text)
    text = tokenization(text)
    text = spelling_correction(text, 'textblob')
    return ' '.join(text)

def eval(dataset = "A"):
    if dataset=="A":
        df_a_train = pd.read_csv("A2/A1_dataset.csv")
        df_a_test = pd.read_csv("A2/A2_test_dataset.csv")
        df_a_train['preprocessed_text'] = df_a_train['TEXT'].apply(preprocess_text)
        df_a_test['preprocessed_text'] = df_a_test['TEXT'].apply(preprocess_text)
        df_a_train['preprocessed_text_without_alnum'] = df_a_train['TEXT'].apply(lambda text: preprocess_text(text, True))
        df_a_test['preprocessed_text_without_alnum'] = df_a_test['TEXT'].apply(lambda text: preprocess_text(text, True))

        a_train_sentences, a_train_labels = df_a_train["preprocessed_text"].values, df_a_train["LABEL"].values
        a_test_sentences, a_test_labels = df_a_test["preprocessed_text"].values, df_a_test["LABEL"].values
        acc_A_with_alnum = train_and_evaluate(a_train_sentences, a_train_labels, a_test_sentences, a_test_labels)

        a_train_sentences_without_alnum, a_train_labels_without_alnum = df_a_train["preprocessed_text_without_alnum"].values, df_a_train["LABEL"].values
        a_test_sentences_without_alnum, a_test_labels_without_alnum = df_a_test["preprocessed_text_without_alnum"].values, df_a_test["LABEL"].values
        acc_A_without_alnum = train_and_evaluate(a_train_sentences_without_alnum, a_train_labels_without_alnum, a_test_sentences_without_alnum, a_test_labels_without_alnum)

        print("Accuracy on A1 dataset without removing alphanumeric characters: ", acc_A_with_alnum)
        print("Accuracy on A1 dataset after removing alphanumeric characters: ", acc_A_without_alnum)
    
    elif dataset=="B":
        pass
    
    else:
        return ValueError("Pass the appropriate dataset type from A or B.")
