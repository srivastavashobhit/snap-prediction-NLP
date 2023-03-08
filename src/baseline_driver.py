from models.baseline_model import NLTKBaseLine

def get_test_data(test_file):
    test_sentence_inputs = []
    test_sentence_outputs = []
    with open(test_file, 'r', encoding='utf8') as f:
        for line in f:
            words = line.strip().split(" ")
            if len(words) >= sequence_length:
                for i in range(len(words)-2):
                    test_sentence_input = []
                    test_sentence_output = None
                    for j in range(sequence_length-1):
                        test_sentence_input.append(words[i+j])
                    test_sentence_output = words[i+j+1]
                test_sentence_inputs.append(test_sentence_input)
                test_sentence_outputs.append(test_sentence_output)
    return test_sentence_inputs, test_sentence_outputs

def calculate_accuracy(test_sentence_outputs, test_predictions):
    correct_cnt = 0
    for x,y in zip(test_sentence_outputs, test_predictions):
        if x == y:
            correct_cnt += 1
    return correct_cnt/len(test_sentence_outputs)

if __name__ == "__main__":

    train_file = "corpora/snaps_corpa_train.txt"
    test_file = "corpora/snaps_corpa_test.txt"
    
    with open(train_file, 'r', encoding='utf8') as f:
        train_corpus = f.read()

    sequence_length = 3
    model = NLTKBaseLine(sequence_length)
    model.build()
    model.train(train_corpus)

    test_sentence_inputs, test_sentence_outputs = get_test_data(test_file)

    test_predictions = model.predict_all(test_sentence_inputs)

    accuracy = calculate_accuracy(test_sentence_outputs, test_predictions)

    print("Prdiction Accuracy on test set ",len(test_predictions), "inputs:", "{:.2f}".format(accuracy*100), "%")


    test_sentence_inputs, test_sentence_outputs = get_test_data(train_file)

    test_predictions = model.predict_all(test_sentence_inputs)

    accuracy = calculate_accuracy(test_sentence_outputs, test_predictions)

    print("Prdiction Accuracy on train set ",len(test_predictions), "inputs:", "{:.2f}".format(accuracy*100), "%")