import pickle
import numpy as np

class LanguageModel:
    """
    Bigram Model for Text Generation Based on a Prompt
    """
    def __init__(self, bigrams, unigram_counts, sentiment_model, add_to_numerator=False, add_to_denominator=False, add_externally=True, sentiment_scale_factor=1, repetition_penalty=0.2, normalizebyPerplexity=False):
        self.bigrams = bigrams
        self.unigram_counts = unigram_counts
        self.sentiment_model = sentiment_model
        self.score_dict = {}
        self.add_to_numerator = add_to_numerator
        self.add_to_denominator = add_to_denominator
        self.add_externally = add_externally
        self.sentiment_scale_factor = sentiment_scale_factor
        self.repetiton_penalty = repetition_penalty
        self.normalizebyPerplexity = normalizebyPerplexity
        self.load_lexicon()

    def load_lexicon(self):
        dump_scores = None
        if self.sentiment_model == 'vader':
            dump_scores = pickle.load(open('Pos Neg Prompts\ls_word_sentiment_vader.pickle', 'rb'))
        elif self.sentiment_model == 'hf':
            dump_scores = pickle.load(open('Pos Neg Prompts\ls_word_sentiment_hf.pickle', 'rb'))
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

        # Store Top 5 Words
        top_words = []
        top_word_freqs = []

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
                
                if self.normalizebyPerplexity:
                    normalized_prob /= self.computePerplexity(sentence_so_far + [bigram[1]])

                if len(top_words) < 5:
                    top_words.append(bigram[1])
                    top_word_freqs.append(normalized_prob)
                else:
                    if normalized_prob > min(top_word_freqs):
                        min_index = top_word_freqs.index(min(top_word_freqs))
                        top_words[min_index] = bigram[1]
                        top_word_freqs[min_index] = normalized_prob

        # choose a word from the top 5 with probability proportional to its frequency and seed
        top_word_freqs = np.array(top_word_freqs)
        top_word_freqs = top_word_freqs / top_word_freqs.sum()
        next_word = np.random.choice(top_words, p=top_word_freqs)
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