from models.baseline_model import NLTKBaseLine
from utils.baseline_utils import get_test_data, calculate_accuracy

if __name__ == "__main__":

    train_file = "corpora/snaps_corpa_train.txt"
    test_file = "corpora/snaps_corpa_test.txt"
    
    with open(train_file, 'r', encoding='utf8') as f:
        train_corpus = f.read()

    sequence_length = 3
    model = NLTKBaseLine(sequence_length)
    model.build()
    model.train(train_corpus)

    top_n = 5

    test_sentence_inputs, test_sentence_outputs = get_test_data(test_file, sequence_length)

    test_predictions = model.predict_all_top_n(test_sentence_inputs, top_n)

    accuracy = calculate_accuracy(test_sentence_outputs, test_predictions)

    print("Prediction Accuracy on test set ",len(test_predictions), "inputs:", "{:.2f}".format(accuracy*100), "%")


    test_sentence_inputs, test_sentence_outputs = get_test_data(train_file, sequence_length)

    test_predictions = model.predict_all_top_n(test_sentence_inputs, top_n)

    accuracy = calculate_accuracy(test_sentence_outputs, test_predictions)

    print("Prediction Accuracy on train set ",len(test_predictions), "inputs:", "{:.2f}".format(accuracy*100), "%")