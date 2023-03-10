import tensorflow as tf
import argparse
from utils.lstm_utils import *
import os
import dill
import pickle 
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str, help="Input Snap Text File", default="corpora/snaps_corpa_test.txt")
    parser.add_argument('-m', '--model', type=str, help="Model Path")
    parser.add_argument('-n', '--topn', type=str, help="TOP N Accuracy", default = 5)

    args = parser.parse_args()

    model_path = args.model

    if not model_path:
        # Set the path of the directory containing the subdirectories
        dir_path = "saved_trained_models\lstm"

        # Get a list of all subdirectories in the directory
        subdirs = [os.path.join(dir_path, subdir) for subdir in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, subdir))]

        # Sort the subdirectories by name (timestamp) in reverse chronological order
        sorted_subdirs = sorted(subdirs, key=lambda x: os.path.basename(x), reverse=True)

        # Get the most recent subdirectory
        model_path = os.path.join(sorted_subdirs[0],"lstm.pkl")

    # Open the file for reading in binary mode
    with open(model_path, "rb") as model_file:
        # Load the model from the file
        model = pickle.load(model_file)

    test_file = args.file
    test_sentence_inputs, test_sentence_outputs = get_test_data(test_file)

    top_n = int(args.topn)
    test_sentence_predictions_top_n = model.predict_all_top_n(test_sentence_inputs,top_n)

    accuracy = top_5_accuracy(test_sentence_predictions_top_n, test_sentence_outputs)

    print("The top",top_n,"accuracy is {:0.2f}".format(accuracy))

    