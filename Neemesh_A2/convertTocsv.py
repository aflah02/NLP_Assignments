import pandas as pd
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer

sid = SentimentIntensityAnalyzer()
columns = ["generatedText", "ourLabels", "vaderLabels", "perpelexity"]

def find_polarity_scores(txt):
    ls_word_sentiment_vader = []
    for sentence in txt:
        ls_word_sentiment_vader.append(0 if sid.polarity_scores(sentence)['compound']<0 else 1)
    ls_word_sentiment_vader = np.array(ls_word_sentiment_vader)
    return ls_word_sentiment_vader

def txtTocsv(negFilePath, posFilePath):
    filename = '_'.join(negFilePath.split("\\")[-1][:-4].split("_")[2:])
    neg_txt = []
    ppl = []
    with open(negFilePath, 'r') as f:
        neg_txt = f.readlines()
    for each in range(len(neg_txt)):
        neg_txt[each] = neg_txt[each][:-2]
        ppl.append(neg_txt[each].split()[-1])
        neg_txt[each] = ' '.join(neg_txt[each].split()[:-1])

    pos_txt = []
    with open(posFilePath, 'r') as f:
        pos_txt = f.readlines()
    for each in range(len(pos_txt)):
        pos_txt[each] = pos_txt[each][:-2]
        ppl.append(pos_txt[each].split()[-1])
        pos_txt[each] = ' '.join(pos_txt[each].split()[:-1])

    txt = neg_txt + pos_txt

    neg_vaderScores = find_polarity_scores(neg_txt)
    pos_vaderScores = find_polarity_scores(pos_txt)

    txt = neg_txt + pos_txt
    for i in range(500):
        txt[i] = [txt[i]]
        if i<250:
            txt[i].append(0)
            txt[i].append(neg_vaderScores[i])
            txt[i].append(ppl[i])
        else:
            txt[i].append(1)
            txt[i].append(pos_vaderScores[i-250])
            txt[i].append(ppl[i])
        txt[i] = np.array(txt[i])
    txt = np.array(txt)
    df = pd.DataFrame(txt, columns = columns)
    df.to_csv(f"A2/generated_CSVs/generated_{filename}.csv")

txtTocsv("A2\\generated_sentences\\neg_gen_add_numerator.txt", "A2\\generated_sentences\\pos_gen_add_numerator.txt")
txtTocsv("A2\\generated_sentences\\neg_gen_div_denominator.txt", "A2\\generated_sentences\\pos_gen_div_denominator.txt")
txtTocsv("A2\\generated_sentences\\neg_gen_mul_numerator.txt", "A2\\generated_sentences\\pos_gen_mul_numerator.txt")
txtTocsv("A2\\generated_sentences\\neg_gen_only_ext.txt", "A2\\generated_sentences\\neg_gen_only_ext.txt")
txtTocsv("A2\\generated_sentences\\neg_hf_prompts.txt", "A2\\generated_sentences\\neg_hf_prompts.txt")
txtTocsv("A2\\generated_sentences\\neg_ppl_normalized.txt", "A2\\generated_sentences\\neg_ppl_normalized.txt")