import tensorflow as tf
from utils.lstm_utils import *
import argparse
import pickle
import os

if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--input', type=str, help="Input Snap Text")
    parser.add_argument('-f', '--file', type=str, help="Input Snap Text File", default="corpora/snaps_corpa_test.txt")
    parser.add_argument('-m', '--model', type=str, help="Model Path")

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

    input_sequences = None

    if args.input:
        input_sequences = [args.input]
        
    elif args.file:
        with open(args.file, 'r') as file:
            input_sequences = [line.strip() for line in file]

    if input_sequences:
        y_pred = model.predict_all_top_n(input_sequences,1)
        print(y_pred)
    
    