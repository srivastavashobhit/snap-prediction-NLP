# Next SNAP prediction using LSTM

To train the model on default dataset
`python lstm_trainer.py`

To train the model on custom dataset, when data is saved in a txt file with each line as a list of Snaps.
`python lstm_trainer.py -f [path_to_text_file]`

To evaluate the model on the default test data.
`python lstm_evaluator.py`

To evaluate the model on custom dataset, when data is saved in a txt file with each line as a list of Snaps.
`python lstm_evaluator.py -f [path_to_text_file]`

To make inference using the model on a single or multiple comma seperate inputs.
`python lstm_predictor.py -t snap-a1 snap-a2 snap-a3, snap-b1 snap-b2`

To make inference using the model on a input file with each line as a list of Snaps.
`python lstm_predictor.py -f [path_to_text_file]`

