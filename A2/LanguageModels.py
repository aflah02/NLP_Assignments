import pickle


class LanguageModel:
    """
    Bigram Model for Text Generation Based on a Prompt
    """
    def __init__(self, bigrams, unigram_counts, sentiment_model, add_to_numerator=False, add_to_denominator=False, add_externally=True, sentiment_scale_factor=1e-6, repetition_penalty=0.2):
        self.bigrams = bigrams
        self.unigram_counts = unigram_counts
        self.sentiment_model = sentiment_model
        self.score_dict = {}
        self.add_to_numerator = add_to_numerator
        self.add_to_denominator = add_to_denominator
        self.add_externally = add_externally
        self.sentiment_scale_factor = sentiment_scale_factor
        self.repetiton_penalty = repetition_penalty
        self.load_lexicon()

    def load_lexicon(self):
        dump_scores = None
        if self.sentiment_model == 'vader':
            dump_scores = pickle.load(open('ls_word_sentiment_vader.pickle', 'rb'))
        elif self.sentiment_model == 'hf':
            dump_scores = pickle.load(open('ls_word_sentiment_hf.pickle', 'rb'))
        for line in dump_scores:
            self.score_dict[line[0]] = line[1]

    def generate_text(self, prompt, sentiment, length=20):
        """
        Generate Text Based on a Prompt
        """
        last_word = prompt[-1]
        length -= len(prompt)
        generated_so_far = {}

        for word in prompt:
            if word in generated_so_far:
                generated_so_far[word] += 1
            else:
                generated_so_far[word] = 1

        for i in range(length):

            next_word = self.get_next_word(last_word, sentiment, generated_so_far, sentence_so_far = prompt)

            if next_word in generated_so_far:
                generated_so_far[next_word] += 1
            else:
                generated_so_far[next_word] = 1
                
            prompt.append(next_word)
            last_word = next_word
        return prompt
    
    def get_next_word(self, word, sentiment, generated_so_far, sentence_so_far):
        """
        Get the next word based on the last word
        """
        highest_freq = 0
        next_word = ''
        for bigram, freq in self.bigrams.items():
            if bigram[0] == word:
                numerator = freq
                denominator = self.unigram_counts[bigram[0]]

                if self.add_to_numerator:
                    numerator *= self.score_dict[bigram[1]] * sentiment * self.sentiment_scale_factor

                if self.add_to_denominator:
                    denominator /= self.score_dict[bigram[1]] * sentiment * self.sentiment_scale_factor

                normalized_prob = numerator / denominator

                if self.add_externally:
                    normalized_prob += self.score_dict[bigram[1]] * sentiment * self.sentiment_scale_factor

                if bigram[1] in generated_so_far.keys():
                    normalized_prob -= self.repetiton_penalty * generated_so_far[bigram[1]]
                # sentence_after_choosing = " ".join(sentence_so_far + [bigram[1]])

                # if self.sentance_penalty:


                if normalized_prob > highest_freq:
                    highest_freq = normalized_prob
                    next_word = bigram[1]

        return next_word

    def computePerplexity(self, sentence):
        """
        Compute Perplexity of a Sentence
        """
        perplexity = 1
        vocab_size = len(self.unigram_counts)
        perplexity *= vocab_size
        for i in range(len(sentence) - 1):
            perplexity *= 1 / self.bigrams[(sentence[i], sentence[i + 1])]
        return perplexity ** (1 / len(sentence))