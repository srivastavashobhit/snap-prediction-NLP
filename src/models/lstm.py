import tensorflow as tf
import numpy as np
from tqdm import tqdm
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences

class NextWordLSTM(tf.keras.Sequential):
    def __init__(self, max_seq_len, vocab_size, tokenizer):
        super(NextWordLSTM, self).__init__()
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.tokenizer = tokenizer
        self.model = None

    def build_model(self):
        self.model = Sequential()
        self.model.add(Embedding(input_dim=self.vocab_size, output_dim=100, input_length=self.max_seq_len))
        self.model.add(LSTM(units=150, return_sequences=True))
        self.model.add(LSTM(units=150))
        self.model.add(Dense(units=self.vocab_size, activation='softmax')) # Output1  #inpur for output2
        self.model.add(Dense(units=self.vocab_size, activation='softmax')) # output2
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


    def train(self, X_train, y_train, to_validatate = False, X_val=None, y_val=None, epochs=1, verbose=1):
        if not self.model:
            self.build_model()
        if to_validatate:
            print("Training with validation")
            return self.model.fit(X_train, y_train, epochs=epochs, verbose=verbose, validation_data=(X_val, y_val))
        else:
            print("Training without validation")
            return self.model.fit(X_train, y_train, epochs=epochs, verbose=verbose)

    def predict(self, X_test):
        if not self.model:
            raise ValueError("Model has not been trained or loaded.")
        return self.model.predict(X_test)
    

    def predict_all_top_n(self, input_sequences, top_n):
        predictions = []
        for input_sequence in tqdm(input_sequences):
            # Convert seed text to integer sequence
            sequence = self.tokenizer.texts_to_sequences([input_sequence])[0]
            # Pad sequence to same length as input sequences
            sequence = pad_sequences([sequence], maxlen=self.max_seq_len, padding='pre')
            # Predict next word
            predicted = self.model.predict(sequence, verbose=0)
            top_n_indices = np.argsort(predicted.reshape(-1))[-top_n:][::-1]
            # Convert integer to word
            prediction_top_n = []
            for predicted_class_top_n in top_n_indices:
                prediction = ""
                not_found_word_index = True
                for word, index in self.tokenizer.word_index.items():
                    if index == predicted_class_top_n:
                        not_found_word_index = False
                        prediction_top_n.append(word)
                        break
                if not_found_word_index:
                    prediction_top_n.append(prediction)
            predictions.append(prediction_top_n)
        return predictions
    
