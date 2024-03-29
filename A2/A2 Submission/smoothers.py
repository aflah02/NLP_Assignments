class LMSmoothers:
    """
    Bigram Smoothing
    """
    def __init__(self, bigrams, vocab_len, unigram_counts, unigrams_probs = None):
        self.bigrams = bigrams
        self.unigrams_probs = unigrams_probs
        self.unigram_counts = unigram_counts
        self.vocab_len = vocab_len
        self.laplace_bigrams = {}
        self.add_k_bigrams = {}
        self.add_k_with_unigram_prior_bigrams = {}

    def laplace(self):
        """
        Laplace Smoothing
        """
        for bigram, freq in self.bigrams.items():
            self.laplace_bigrams[bigram] = (freq + 1) / (self.unigram_counts[bigram[0]] + self.vocab_len)
        return self.laplace_bigrams
    
    def add_k(self, k):
        """
        Add-k Smoothing
        """
        for bigram, freq in self.bigrams.items():
            self.add_k_bigrams[bigram] = (freq + k) / (self.unigram_counts[bigram[0]] + k * len(self.vocab_len))
        return self.add_k_bigrams

    def add_k_with_unigram_prior(self, k):
        """
        Add-k Smoothing with Unigram Prior
        """
        m = k * self.vocab_len
        for bigram, freq in self.bigrams.items():
            self.add_k_with_unigram_prior_bigrams[bigram] = (freq + m * self.unigrams_probs[bigram[1]]) / (self.unigram_counts[bigram[0]] + m)
        return self.add_k_with_unigram_prior_bigrams
        
        
    
        

