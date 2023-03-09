def get_test_data(test_file, sequence_length):
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