from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
from LanguageModels import *
from preprocess_text import *
import math

def train_and_evaluate(train_sentences, train_labels, test_sentences, test_labels):
    model = make_pipeline(TfidfVectorizer(), MultinomialNB())
    model.fit(train_sentences, train_labels)
    predicted_test_labels = model.predict(test_sentences)
    return accuracy_score(test_labels, predicted_test_labels)

def ceil(number, digits) -> float: return math.ceil((10.0 ** digits) * number) / (10.0 ** digits)

def preprocess_text(text):
    text = lowercase_text(text)
    text = remove_url_html(text)
    text = remove_users(text)
    text = remove_punctuations(text)
    text = remove_whitespaces(text)
    text = tokenization(text)
    # text = spelling_correction(text, 'autocorrect_full')
    text = remove_alphanum(text)
    return ' '.join(text)

class IntrinsicEvaluation:
    """
    Perform Intrinsic Evaluation on the given generated sentences
    """
    def __init__(self, generated_sentences, language_model = LanguageModel):
        self.lm = language_model
        self.generated_sentences = generated_sentences
    
    def getPerplexity(self, sentence):
        """
        Gets the Perplexity of current sentence
        """
        lm = self.lm
        return lm.computePerplexity(sentence = sentence)
    
    def eval(self):
        """
        Evaluate
        """
        ppl = []
        for each_sentence in self.generated_sentences:
            each_ppl = self.getPerplexity(each_sentence)
            ppl.append(each_ppl)
        ppl = np.array(ppl)
        print("Average Perplexity of all 500 generated sentences: ", np.mean(ppl))


class ExtrinsicEvaluation:
    """
    Perform Extrinsic Evaluation based on the Dataset type
    """
    def __init__(self, dataset_type = "A"):
        self.dataset_type = dataset_type
        self.df_test = pd.read_csv("Neemesh_A2\A2_test_dataset_preprocessed.csv")

    def eval(self, generated_sentences = None):
        """
        Evaluate on the given dataset type
        """
        if self.dataset_type=="A":
            df_a_train = pd.read_csv("A2/A1_dataset.csv")
            df_a_test = self.df_test
            df_a_test.fillna(" ", inplace = True)
            df_a_train['preprocessed_text'] = df_a_train['TEXT'].apply(preprocess_text)
            # df_a_test['preprocessed_text'] = df_a_test['TEXT'].apply(preprocess_text)

            a_train_sentences, a_train_labels = df_a_train["preprocessed_text"].values, df_a_train["LABEL"].values
            a_test_sentences, a_test_labels = df_a_test["pretext"].values, df_a_test["LABEL"].values
            acc_A = train_and_evaluate(a_train_sentences, a_train_labels, a_test_sentences, a_test_labels)

            print("Accuracy on A1 dataset: ", ceil(acc_A, 4)*100)
        
        elif self.dataset_type=="B":
            df_a_train = pd.read_csv("A2/A1_dataset.csv")
            df_a_train['preprocessed_text'] = df_a_train['TEXT'].apply(preprocess_text)
            df_b_test = self.df_test
            df_b_test.fillna(" ", inplace = True)
            df_b_test['preprocessed_text'] = df_b_test['TEXT'].apply(preprocess_text)

            a_train_sentences, a_train_labels = df_a_train["preprocessed_text"].values, df_a_train["LABEL"].values
            b_train_sentences, b_train_labels = generated_sentences[:, 0], np.array([int(i) for i in generated_sentences[:, 1]])
            b_test_sentences, b_test_labels = df_b_test["preprocessed_text"].values, df_b_test["LABEL"].values

            b_train_sentences = np.append(b_train_sentences, a_train_sentences)
            b_train_labels = np.append(b_train_labels, a_train_labels)
            acc_B = train_and_evaluate(b_train_sentences, b_train_labels, b_test_sentences, b_test_labels)

            print("Accuracy on our generated dataset: ", round(100*acc_B, 2))

        else:
            return ValueError("Pass the appropriate dataset type from A or B.")
