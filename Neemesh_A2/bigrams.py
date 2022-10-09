class Bigrams:
    """
    Takes in a list of tokenized strings and creates bigrams pairs with frequencies
    Can optionally return a matrix too with all possible bigram pairs
    """
    def __init__(self, list_of_tokenized_strings):
        self.list_of_tokenized_strings = list_of_tokenized_strings
        self.bigrams = {}
        self.matrix = {}
        self.vocab_len = len(list(set([item for sublist in self.list_of_tokenized_strings for item in sublist])))
        self.build_bigrams()
        self.build_matrix()

    def build_bigrams(self):
        """
        Returns a dictionary of bigrams with frequencies
        """
        for s in self.list_of_tokenized_strings:
            for i in range(len(s) - 1):
                bigram = (s[i], s[i + 1])
                if bigram in self.bigrams:
                    self.bigrams[bigram] += 1
                else:
                    self.bigrams[bigram] = 1
        # Add Remaining Bigrams with 0 frequency
        vocab = list(set([item for sublist in self.list_of_tokenized_strings for item in sublist]))
        for i in range(len(vocab)):
            for j in range(len(vocab)):
                if (vocab[i], vocab[j]) not in self.bigrams:
                    self.bigrams[(vocab[i], vocab[j])] = 0

    def build_matrix(self):
        """
        Create a 2D matrix with cell values indicating frequency
        """
        vocab_len = len(list(set([item for sublist in self.list_of_tokenized_strings for item in sublist])))
        # 2D matrix of zeros
        self.matrix = [[0 for i in range(vocab_len)] for j in range(vocab_len)]
        # Assign index to each word
        vocab = list(set([item for sublist in self.list_of_tokenized_strings for item in sublist]))
        vocab_index = {word: i for i, word in enumerate(vocab)}
        # Fill in the matrix
        for bigram, freq in self.bigrams.items():
            self.matrix[vocab_index[bigram[0]]][vocab_index[bigram[1]]] = freq

    def get_bigrams(self):
        """
        Returns a dictionary of bigrams with frequencies
        """
        return self.bigrams

    def get_matrix(self):
        """
        Returns a 2D matrix with cell values indicating frequency
        """
        return self.matrix