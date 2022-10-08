from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy as np
from evaluation import ExtrinsicEvaluation

def find_polarity_scores(txt):
    sid = SentimentIntensityAnalyzer()
    ls_word_sentiment_vader = []
    for sentence in txt:
        ls_word_sentiment_vader.append(0 if sid.polarity_scores(sentence)['compound']<0 else 1)
    ls_word_sentiment_vader = np.array(ls_word_sentiment_vader)
    return ls_word_sentiment_vader

neg_txt = []
with open("A2/neg_gen_only_ext.txt", 'r') as f:
    neg_txt = f.readlines()
for each in range(len(neg_txt)):
    neg_txt[each] = neg_txt[each][:-2]
    neg_txt[each] = ' '.join(neg_txt[each].split()[:-1])

pos_txt = []
with open("A2/pos_gen_only_ext.txt", 'r') as f:
    pos_txt = f.readlines()
for each in range(len(pos_txt)):
    pos_txt[each] = pos_txt[each][:-2]
    pos_txt[each] = ' '.join(pos_txt[each].split()[:-1])

print((find_polarity_scores(neg_txt)==np.array([0]*250)).sum())
print((find_polarity_scores(pos_txt)==np.array([1]*250)).sum())

txt = neg_txt + pos_txt
for i in range(500):
    txt[i] = [txt[i]]
    if i<250:
        txt[i].append(0)
    else:
        txt[i].append(1)
    txt[i] = np.array(txt[i])
txt = np.array(txt)

np.random.shuffle(txt)

ExtrinsicEvaluation("B").eval(txt)

print("--------------------------------------------------------------------------")

neg_txt = []
with open("A2/neg_gen_add_numerator.txt", 'r') as f:
    neg_txt = f.readlines()
for each in range(len(neg_txt)):
    neg_txt[each] = neg_txt[each][:-2]
    neg_txt[each] = ' '.join(neg_txt[each].split()[:-1])

pos_txt = []
with open("A2/pos_gen_only_ext.txt", 'r') as f:
    pos_txt = f.readlines()
for each in range(len(pos_txt)):
    pos_txt[each] = pos_txt[each][:-2]
    pos_txt[each] = ' '.join(pos_txt[each].split()[:-1])

print((find_polarity_scores(neg_txt)==np.array([0]*250)).sum())
print((find_polarity_scores(pos_txt)==np.array([1]*250)).sum())

txt = neg_txt + pos_txt
for i in range(500):
    txt[i] = [txt[i]]
    if i<250:
        txt[i].append(0)
    else:
        txt[i].append(1)
    txt[i] = np.array(txt[i])
txt = np.array(txt)

np.random.shuffle(txt)

ExtrinsicEvaluation("B").eval(txt)

print("-----")

neg_txt = []
with open("A2/neg_gen_mul_numerator.txt", 'r') as f:
    neg_txt = f.readlines()
for each in range(len(neg_txt)):
    neg_txt[each] = neg_txt[each][:-2]
    neg_txt[each] = ' '.join(neg_txt[each].split()[:-1])

pos_txt = []
with open("A2/pos_gen_only_ext.txt", 'r') as f:
    pos_txt = f.readlines()
for each in range(len(pos_txt)):
    pos_txt[each] = pos_txt[each][:-2]
    pos_txt[each] = ' '.join(pos_txt[each].split()[:-1])

print(find_polarity_scores(neg_txt), np.array([0]*250))
print((find_polarity_scores(neg_txt)==np.array([0]*250)).sum())
print((find_polarity_scores(pos_txt)==np.array([1]*250)).sum())

txt = neg_txt + pos_txt
for i in range(500):
    txt[i] = [txt[i]]
    if i<250:
        txt[i].append(0)
    else:
        txt[i].append(1)
    txt[i] = np.array(txt[i])
txt = np.array(txt)

np.random.shuffle(txt)

ExtrinsicEvaluation("B").eval(txt)