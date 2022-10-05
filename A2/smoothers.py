class LMSmoothers:
    """
    Bigram Smoothing
    """
    def __init__(self, bigrams, vocab_len):
        self.bigrams = bigrams
        self.vocab_len = vocab_len
        self.laplace_bigrams = {}

    def laplace(self):
        """
        Laplace Smoothing
        """
        for bigram, freq in self.bigrams.items():
            self.laplace_bigrams[bigram] = (freq + 1) / (self.vocab_len + len(self.bigrams))
        return self.laplace_bigrams
        
        
    
        

