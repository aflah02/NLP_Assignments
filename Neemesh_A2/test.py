from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy as np
from evaluation import ExtrinsicEvaluation

sid = SentimentIntensityAnalyzer()

def find_polarity_scores(txt):
    ls_word_sentiment_vader = []
    for sentence in txt:
        ls_word_sentiment_vader.append(0 if sid.polarity_scores(sentence)['compound']<0 else 1)
    ls_word_sentiment_vader = np.array(ls_word_sentiment_vader)
    return ls_word_sentiment_vader

ExtrinsicEvaluation("A").eval()

print("--------------------------------------------------------------------------")

neg_txt = []
with open("A2/generated_sentences/neg_gen_only_ext.txt", 'r') as f:
    neg_txt = f.readlines()
for each in range(len(neg_txt)):
    neg_txt[each] = neg_txt[each][:-2]
    neg_txt[each] = ' '.join(neg_txt[each].split()[:-1])

pos_txt = []
with open("A2/generated_sentences/pos_gen_only_ext.txt", 'r') as f:
    pos_txt = f.readlines()
for each in range(len(pos_txt)):
    pos_txt[each] = pos_txt[each][:-2]
    pos_txt[each] = ' '.join(pos_txt[each].split()[:-1])

neg_vaderScores = find_polarity_scores(neg_txt)
pos_vaderScores = find_polarity_scores(pos_txt)

txt = neg_txt + pos_txt
for i in range(500):
    txt[i] = [txt[i]]
    if i<250:
        txt[i].append(neg_vaderScores[i])
    else:
        txt[i].append(pos_vaderScores[i-250])
    txt[i] = np.array(txt[i])
txt = np.array(txt)

np.random.shuffle(txt)

ExtrinsicEvaluation("B").eval(txt)

print("--------------------------------------------------------------------------")

neg_txt = []
with open("A2/generated_sentences/neg_gen_add_numerator.txt", 'r') as f:
    neg_txt = f.readlines()
for each in range(len(neg_txt)):
    neg_txt[each] = neg_txt[each][:-2]
    neg_txt[each] = ' '.join(neg_txt[each].split()[:-1])

pos_txt = []
with open("A2/generated_sentences/pos_gen_add_numerator.txt", 'r') as f:
    pos_txt = f.readlines()
for each in range(len(pos_txt)):
    pos_txt[each] = pos_txt[each][:-2]
    pos_txt[each] = ' '.join(pos_txt[each].split()[:-1])

neg_vaderScores = find_polarity_scores(neg_txt)
pos_vaderScores = find_polarity_scores(pos_txt)

txt = neg_txt + pos_txt
for i in range(500):
    txt[i] = [txt[i]]
    if i<250:
        txt[i].append(neg_vaderScores[i])
    else:
        txt[i].append(pos_vaderScores[i-250])
    txt[i] = np.array(txt[i])
txt = np.array(txt)

np.random.shuffle(txt)

ExtrinsicEvaluation("B").eval(txt)

print("--------------------------------------------------------------------------")

neg_txt = []
with open("A2/generated_sentences/neg_gen_mul_numerator.txt", 'r') as f:
    neg_txt = f.readlines()
for each in range(len(neg_txt)):
    neg_txt[each] = neg_txt[each][:-2]
    neg_txt[each] = ' '.join(neg_txt[each].split()[:-1])

pos_txt = []
with open("A2/generated_sentences/pos_gen_mul_numerator.txt", 'r') as f:
    pos_txt = f.readlines()
for each in range(len(pos_txt)):
    pos_txt[each] = pos_txt[each][:-2]
    pos_txt[each] = ' '.join(pos_txt[each].split()[:-1])

neg_vaderScores = find_polarity_scores(neg_txt)
pos_vaderScores = find_polarity_scores(pos_txt)

txt = neg_txt + pos_txt
for i in range(500):
    txt[i] = [txt[i]]
    if i<250:
        txt[i].append(neg_vaderScores[i])
    else:
        txt[i].append(pos_vaderScores[i-250])
    txt[i] = np.array(txt[i])
txt = np.array(txt)

np.random.shuffle(txt)

ExtrinsicEvaluation("B").eval(txt)

print("--------------------------------------------------------------------------")

neg_txt = []
with open("A2/generated_sentences/neg_gen_div_denominator.txt", 'r') as f:
    neg_txt = f.readlines()
for each in range(len(neg_txt)):
    neg_txt[each] = neg_txt[each][:-2]
    neg_txt[each] = ' '.join(neg_txt[each].split()[:-1])

pos_txt = []
with open("A2/generated_sentences/pos_gen_div_denominator.txt", 'r') as f:
    pos_txt = f.readlines()
for each in range(len(pos_txt)):
    pos_txt[each] = pos_txt[each][:-2]
    pos_txt[each] = ' '.join(pos_txt[each].split()[:-1])

neg_vaderScores = find_polarity_scores(neg_txt)
pos_vaderScores = find_polarity_scores(pos_txt)

txt = neg_txt + pos_txt
for i in range(500):
    txt[i] = [txt[i]]
    if i<250:
        txt[i].append(neg_vaderScores[i])
    else:
        txt[i].append(pos_vaderScores[i-250])
    txt[i] = np.array(txt[i])
txt = np.array(txt)

np.random.shuffle(txt)

ExtrinsicEvaluation("B").eval(txt)

print("--------------------------------------------------------------------------")

neg_txt = []
with open("A2/generated_sentences/neg_hf_prompts.txt", 'r') as f:
    neg_txt = f.readlines()
for each in range(len(neg_txt)):
    neg_txt[each] = neg_txt[each][:-2]
    neg_txt[each] = ' '.join(neg_txt[each].split()[:-1])

pos_txt = []
with open("A2/generated_sentences/pos_hf_prompts.txt", 'r') as f:
    pos_txt = f.readlines()
for each in range(len(pos_txt)):
    pos_txt[each] = pos_txt[each][:-2]
    pos_txt[each] = ' '.join(pos_txt[each].split()[:-1])

neg_vaderScores = find_polarity_scores(neg_txt)
pos_vaderScores = find_polarity_scores(pos_txt)

txt = neg_txt + pos_txt
for i in range(500):
    txt[i] = [txt[i]]
    if i<250:
        txt[i].append(neg_vaderScores[i])
    else:
        txt[i].append(pos_vaderScores[i-250])
    txt[i] = np.array(txt[i])
txt = np.array(txt)

np.random.shuffle(txt)

ExtrinsicEvaluation("B").eval(txt)

print("--------------------------------------------------------------------------")

neg_txt = []
with open("A2/generated_sentences/neg_ppl_normalized.txt", 'r') as f:
    neg_txt = f.readlines()
for each in range(len(neg_txt)):
    neg_txt[each] = neg_txt[each][:-2]
    neg_txt[each] = ' '.join(neg_txt[each].split()[:-1])

pos_txt = []
with open("A2/generated_sentences/pos_ppl_normalized.txt", 'r') as f:
    pos_txt = f.readlines()
for each in range(len(pos_txt)):
    pos_txt[each] = pos_txt[each][:-2]
    pos_txt[each] = ' '.join(pos_txt[each].split()[:-1])

neg_vaderScores = find_polarity_scores(neg_txt)
pos_vaderScores = find_polarity_scores(pos_txt)

txt = neg_txt + pos_txt
for i in range(500):
    txt[i] = [txt[i]]
    if i<250:
        txt[i].append(neg_vaderScores[i])
    else:
        txt[i].append(pos_vaderScores[i-250])
    txt[i] = np.array(txt[i])
txt = np.array(txt)

np.random.shuffle(txt)

ExtrinsicEvaluation("B").eval(txt)