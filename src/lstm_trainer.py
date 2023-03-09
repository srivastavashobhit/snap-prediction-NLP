import tensorflow as tf
import numpy as np
from tqdm import tqdm
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from utils.lstm_driver_utils import *
from models.lstm import NextWordLSTM

if __name__ == "__main__":

    train_file = "corpora/snaps_corpa_train.txt"
    test_file = "corpora/snaps_corpa_test.txt"

    print("Generating Train Data")
    predictors, label, max_sequence_len, total_words, tokenizer = generate_train_data(train_file)

    print("Train Data Create. Size:",label.shape)

    embedding_dim = 100
    lstm_units = 150
    epochs = 100
    model = NextWordLSTM(max_sequence_len-1, total_words, embedding_dim, lstm_units)
    model.build_model()

    print("LSTM Model Instantiated")
    print("Starting Training")
    model.train(predictors, label, epochs)

    dest_file = 'saved_models/lstm/'

    print("Saving model in  ",dest_file)
    tf.keras.models.save_model(model, dest_file)





