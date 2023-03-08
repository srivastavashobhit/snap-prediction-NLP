import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from collections import defaultdict
import pickle

class NLTKBaseLine():

    def __init__(self, sequence_length):
        self.sequence_length = sequence_length
        self.tokens = None
        self.sequences = []
        self.model = None

    def tokenize(self, corpus):
        # Step 2: Tokenize your corpus
        self.tokens = [word_tokenize(line.lower()) for line in corpus.splitlines()]

    def generate_seq(self):
        for line in self.tokens:
            #print(line)
            line_sequences = list(ngrams(line, self.sequence_length, pad_left=True, pad_right=True, left_pad_symbol='<s>', right_pad_symbol='</s>'))
            self.sequences += line_sequences

    def build(self):
        self.model = defaultdict(lambda: defaultdict(lambda: 0))


    def train(self, corpus):

        self.tokenize(corpus)
        self.generate_seq()

        for sequence in self.sequences:
            current_words, next_word = sequence[:-1], sequence[-1]
            current_words = ' '.join(current_words)
            self.model[current_words][next_word] += 1

        for current_words in self.model:
            total_count = float(sum(self.model[current_words].values()))
            for next_word in self.model[current_words]:
                self.model[current_words][next_word] /= total_count

    # def save_model(self, dest_file):
    #     with open(dest_file, 'wb') as f:
    #         pickle.dump(self.model, f)

    # def load_model(self, src_file):
    #     with open(src_file, 'rb') as f:
    #         self.model = pickle.load(f)

    def predict(self, single_input_seq):
        assert len(single_input_seq) == self.sequence_length-1, len(single_input_seq) 
        single_input_seq = ' '.join(single_input_seq)
        if single_input_seq not in self.model:
            # print("** Unknown Words **")
            return "Input Unknown"
        next_word_probs = self.model[single_input_seq]
        return max(next_word_probs, key=next_word_probs.get)
    
    def predict_all(self, list_of_input_seq):
        outputs = []
        for single_input_seq in list_of_input_seq:
            outputs.append(self.predict(single_input_seq))
        return outputs
