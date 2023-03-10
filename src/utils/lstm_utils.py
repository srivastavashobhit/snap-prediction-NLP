import tensorflow as tf
import numpy as np
import os
from tqdm import tqdm
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def generate_train_data(data_file):
    # Load data from text file
    with open(data_file, 'r') as file:
        sentences = file.read().splitlines()

    # Tokenize the text data
    #custom_regex = r'[^\w\-]+'
    filters='!"#$%&()*+,./:;<=>?@[\\]^_`{|}~\t\n' # - is removed 
    tokenizer = Tokenizer(filters=filters)
    tokenizer.fit_on_texts(sentences)
    word_index = tokenizer.word_index
    total_words = len(word_index) + 1
    input_sequences = []

    # Generate input sequences for training
    for sentence in sentences:
        token_list = tokenizer.texts_to_sequences([sentence])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)

    # Pad input sequences
    max_sequence_len = max([len(x) for x in input_sequences])
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

    # Create training data and labels
    predictors, label = input_sequences[:,:-1],input_sequences[:,-1]
    label = tf.keras.utils.to_categorical(label, num_classes=total_words)

    return predictors, label, max_sequence_len, total_words, tokenizer

def get_all_subsets(words):
    subsets = []
    line_len = len(words)

    for i in range(2, line_len+1):
        for j in range(line_len-i+1):
            subsets.append( words[j:j+i])

    return subsets

def get_continuous_subsets(words, min_input_len):
    subsets = []

    line_len = len(words)

    for i in range(line_len - min_input_len + 1):
        for j in range(i + min_input_len, line_len + 1):
            subsets.append(words[i:j])
    return subsets

def get_test_data(test_file, min_input_len=2):
    input_sequences = []
    outputs = []
    with open(test_file, 'r', encoding='utf8') as f:
        for line in f:
            words = line.strip().split(" ")
            words_combinations = get_continuous_subsets(words, min_input_len+1)
            for words_combination in words_combinations:
                input_sequence = " ".join(words_combination[:-1])
                output = words_combination[-1]
                input_sequences.append(input_sequence)
                outputs.append(output)
    
    return input_sequences, outputs
def calculate_accuracy(test_sentence_outputs, test_predictions):
    correct_cnt = 0
    for x,y in zip(test_sentence_outputs, test_predictions):
        if x == y:
            correct_cnt += 1
    return correct_cnt/len(test_sentence_outputs)

# def predict_all_top_n(model, input_sequences, tokenizer, max_sequence_len, top_n):
#     predictions = []
#     for input_sequence in tqdm(input_sequences):
#         # Convert seed text to integer sequence
#         sequence = tokenizer.texts_to_sequences([input_sequence])[0]
#         # Pad sequence to same length as input sequences
#         sequence = pad_sequences([sequence], maxlen=max_sequence_len-1, padding='pre')
#         # Predict next word
#         predicted = model.predict(sequence, verbose=0)
#         top_n_indices = np.argsort(predicted.reshape(149))[-top_n:][::-1]
#         # Convert integer to word
#         prediction_top_n = []
#         for predicted_class_top_n in top_n_indices:
#             prediction = ""
#             not_found_word_index = True
#             for word, index in tokenizer.word_index.items():
#                 if index == predicted_class_top_n:
#                     not_found_word_index = False
#                     prediction_top_n.append(word)
#                     break
#             if not_found_word_index:
#                 prediction_top_n.append(prediction)
#         predictions.append(prediction_top_n)
#     return predictions

def top_5_accuracy(predictions, ground_truth):
    correct = 0
    for i in range(len(ground_truth)):
        if ground_truth[i] in predictions[i]:
            correct += 1
    accuracy = correct / len(ground_truth)
    return accuracy

def get_most_recent_model_path(dir_path = "saved_trained_models\lstm", model_name="lstm.pkl"):
    # Get a list of all subdirectories in the directory
    subdirs = [os.path.join(dir_path, subdir) for subdir in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, subdir))]
    # Sort the subdirectories by name (timestamp) in reverse chronological order
    sorted_subdirs = sorted(subdirs, key=lambda x: os.path.basename(x), reverse=True)
    # Get the most recent subdirectory
    return os.path.join(sorted_subdirs[0],model_name)
