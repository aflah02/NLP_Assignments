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
            self.laplace_bigrams[bigram] = (freq + 1) / (self.unigram_counts[bigram[0]] + len(self.bigrams))
        return self.laplace_bigrams
    
    def add_k(self, k):
        """
        Add-k Smoothing
        """
        for bigram, freq in self.bigrams.items():
            self.add_k_bigrams[bigram] = (freq + k) / (self.unigram_counts[bigram[0]] + k * len(self.bigrams))
        return self.add_k_bigrams

    def add_k_with_unigram_prior(self, k):
        """
        Add-k Smoothing with Unigram Prior
        """
        m = k * self.vocab_len
        for bigram, freq in self.bigrams.items():
            self.add_k_with_unigram_prior_bigrams[bigram] = (freq + m * self.unigrams_probs[bigram[1]]) / (self.unigram_counts[bigram[0]] + m)
        return self.add_k_with_unigram_prior_bigrams

    def witten_bell1(unigrams, bigrams):
        wb_prob=[]
        for w1, w2 in bigrams:
            ngram_count = bigrams[(w1, w2)]
            prior_count = unigrams[w1] 
            type_count = len([bigrams[i] for i in bigrams if i[0] == w1])
            vocab_size = sum(bigrams.values())
            z = vocab_size - type_count
            if ngram_count == 0:
                prob = float(type_count)/float(z*(prior_count + type_count))
            else:
                prob = float(ngram_count)/float(prior_count + type_count)
            wb_lambda=1-bigrams[(w1, w2)]/float(bigrams[(w1, w2)]+sum([bigrams[i] for i in bigrams if i[0] == w1]))
            prob=(wb_lambda)*prob+(1-wb_lambda)*unigrams[w2]/float(sum(unigrams.values()))
            wb_prob.append([w1,w2,prob])
        return wb_prob
        
        
    
        

