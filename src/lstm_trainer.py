import tensorflow as tf
import argparse
import datetime
from sklearn.model_selection import train_test_split
import dill
import os

from utils.lstm_utils import *
from models.lstm import NextWordLSTM

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str, help="Train File", default="corpora/snaps_corpa_train.txt")
    parser.add_argument('-d', '--dest', type=str, help="Model Destination", default="saved_trained_models/lstm/")

    args = parser.parse_args()

    train_file = args.file

    print("Generating Train Data")
    predictors, label, max_sequence_len, total_words, tokenizer = generate_train_data(train_file)

    X_train, X_val, y_train, y_val = train_test_split(predictors, label, test_size=0.2, random_state=42)

    print("Train Data Create. Size:",label.shape)

    epochs = 100
    model = NextWordLSTM(--1, total_words, tokenizer)
    # model.build_model()

    print("LSTM Model Instantiated")
    print("Starting Training")
    model.train(X_train, y_train, to_validatate = True, X_val=X_val, y_val=y_val, epochs=2, verbose=1)

    time_str = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    dest_dir = args.dest + time_str
    os.makedirs(dest_dir)
    dest_file_name = os.path.join(dest_dir, "lstm.pkl")

    print("Saving model in",dest_file_name)

    with open(dest_file_name, 'wb') as model_file:
        dill.dump(model, model_file)



