import tensorflow as tf
from utils.lstm_driver_utils import *

if __name__ == "__main__":
    model_path = 'saved_models/lstm_2/'
    model = tf.keras.models.load_model(model_path)

    train_file = "corpora/snaps_corpa_train.txt"
    _, _, max_sequence_len, total_words, tokenizer = generate_train_data(train_file)
    
    test_file = "corpora/snaps_corpa_test.txt"
    test_sentence_inputs, test_sentence_outputs = get_test_data(train_file) #test_file

    top_n = 5
    test_sentence_predictions_top_n = predict_all_top_n(model, test_sentence_inputs, tokenizer, max_sequence_len, top_n)

    accuracy = top_5_accuracy(test_sentence_predictions_top_n, test_sentence_outputs)

    print("The top",top_n,"accuracy is {:0.2f}".format(accuracy))

    