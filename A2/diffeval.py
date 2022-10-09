from lib2to3.pgen2.pgen import generate_grammar
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
from LanguageModels import *
from preprocess_text import *

def train_and_evaluate(train_sentences, train_labels, test_sentences, test_labels):
    model = make_pipeline(TfidfVectorizer(), MultinomialNB())
    model.fit(train_sentences, train_labels)
    predicted_test_labels = model.predict(test_sentences)
    return accuracy_score(test_labels, predicted_test_labels)

def preprocess_text(text):
    text = lowercase_text(text)
    text = remove_url_html(text)
    text = remove_users(text)
    text = remove_punctuations(text)
    text = remove_whitespaces(text)
    text = tokenization(text)
    text = spelling_correction(text, 'textblob')
    text = remove_alphanum(text)
    return ' '.join(text)

class IntrinsicEvaluation:
    """
    Perform Intrinsic Evaluation on the given generated sentences
    """
    def __init__(self, generated_path):
        self.generated_path = generated_path
        
    def get_perplexities(self):
        with open(self.generated_path, "r") as f:
            lines = f.readlines()
        perplexities = []
        sentences = []
        for line in lines:
            ppl = float(line.split(" ")[-1].strip())
            sentence = line.split(" ")[:-1]
            sentence = ' '.join(sentence)
            perplexities.append(ppl)
            sentences.append(sentence)
        return perplexities, sentences

    def get_avg_perplexity(self, perplexities):
        return sum(perplexities)/len(perplexities)


class ExtrinsicEvaluation:
    """
    Perform Extrinsic Evaluation based on the Dataset type
    """
    def __init__(self, train_path, test_path, addGen, generated_path_pos, generated_path_neg):
        self.train_path = train_path
        self.test_path = test_path
        self.addGen = addGen
        self.generated_path_pos = generated_path_pos
        self.generated_path_neg = generated_path_neg
    
    def build_train(self):
        train_df = pd.read_csv(self.train_path)
        train_sentences = train_df['preprocessed_text']
        for i in range(len(train_sentences)):
            train_sentences[i] = eval(train_sentences[i])
        train_labels = train_df['LABEL']

        for i in range(len(train_sentences)):
            train_sentences[i] = ' '.join(train_sentences[i])

        train_sentences = train_sentences.values
        train_labels = train_labels.values

        if self.addGen == True:
            with open(self.generated_path_pos, "r") as f:
                lines1 = f.readlines()
            with open(self.generated_path_neg, "r") as f:
                lines2 = f.readlines()
            lines = lines1 + lines2
            generated_sentences = []
            for line in lines:
                sentence = line.split(" ")[:-1]
                sentence = ' '.join(sentence)
                generated_sentences.append(sentence)
            vader = SentimentIntensityAnalyzer()
            generated_sentiments = []
            generated_sentiments = [1]*250 + [0]*250
            # for sentence in generated_sentences:
            #     sentiment = vader.polarity_scores(sentence)
            #     if sentiment['compound'] >= 0:
            #         generated_sentiments.append(1)
            #     elif sentiment['compound'] <= 0:
            #         generated_sentiments.append(0)
            # Concatenate the generated sentences with the original training sentences
            train_sentences = np.concatenate((train_sentences, generated_sentences))
            train_labels = np.concatenate((train_labels, generated_sentiments))
        print("Train Sentences: ", len(train_sentences))
        print("Train Labels: ", len(train_labels))
        return train_sentences, train_labels
    
    def build_test(self):
        test_df = pd.read_csv(self.test_path)
        test_sentences = test_df['TEXT'].apply(preprocess_text)
        test_labels = test_df['LABEL']
        test_labels = test_labels.values
        return test_sentences.values, test_labels

    def evaluate(self):
        train_sentences, train_labels = self.build_train()
        test_sentences, test_labels = self.build_test()
        return train_and_evaluate(train_sentences, train_labels, test_sentences, test_labels)
