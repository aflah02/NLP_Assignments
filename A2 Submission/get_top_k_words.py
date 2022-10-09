import pickle

# load the lexicon with scores
ls_word_sentiment_hf = pickle.load(open('Pos Neg Prompts\ls_word_sentiment_hf.pickle', 'rb'))
ls_word_sentiment_vader = pickle.load(open('Pos Neg Prompts\ls_word_sentiment_vader.pickle', 'rb'))

# Sort the lexicon by score
ls_word_sentiment_hf.sort(key=lambda x: x[1])
ls_word_sentiment_vader.sort(key=lambda x: x[1])

# Save the neg 250 words
with open('Pos Neg Prompts\neg_250_hf.txt', 'w') as f:
    for i in range(250):
        f.write(ls_word_sentiment_hf[i][0] + '\n')

with open('Pos Neg Prompts\neg_250_vader.txt', 'w') as f:
    for i in range(250):
        f.write(ls_word_sentiment_vader[i][0] + '\n')

# Save the pos 250 words
with open('Pos Neg Prompts\pos_250_hf.txt', 'w') as f:
    for i in range(1, 251):
        f.write(ls_word_sentiment_hf[-i][0] + '\n')

with open('Pos Neg Prompts\pos_250_vader.txt', 'w') as f:
    for i in range(1, 251):
        f.write(ls_word_sentiment_vader[-i][0] + '\n')