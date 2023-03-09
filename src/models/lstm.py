import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.models import Sequential

class NextWordLSTM:
    def __init__(self, max_seq_len, vocab_size, embedding_dim, lstm_units):
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.model = None

    def build_model(self):
        self.model = Sequential()
        self.model.add(Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dim, input_length=self.max_seq_len))
        self.model.add(LSTM(units=self.lstm_units, return_sequences=True))
        self.model.add(LSTM(units=self.lstm_units))
        self.model.add(Dense(units=self.vocab_size, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def train(self, X_train, y_train, epochs, verbose=1):
        if not self.model:
            self.build_model()
        self.model.fit(X_train, y_train, epochs=epochs, verbose=verbose)

    def predict(self, X_test):
        if not self.model:
            raise ValueError("Model has not been trained or loaded.")
        return self.model.predict(X_test)
