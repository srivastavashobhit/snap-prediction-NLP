# Next SNAP prediction using LSTM

## Training
1. To train the model on default dataset

    `python lstm_trainer.py`

2. To train the model on custom dataset, when data is saved in a txt file with each line as a list of Snaps.

    `python lstm_trainer.py -f [path_to_text_file]`
[path_to_text_file]

## Evaluate
### Metric - TOP 5 Accuracy. 
1. To evaluate the model on the default test data with latest model.

    `python lstm_evaluator.py`

2. To evaluate the model on custom dataset, when data is saved in a txt file with each line as a list of Snaps with latest model with latest model.

    `python lstm_evaluator.py -f [path_to_text_file]`

3. To evaluate the model on the specific model.

    `python lstm_evaluator.py -m [path_to_model]`

4. To evaluate the model on a specific top_n accuracy.

    `python lstm_evaluator.py -n [top_n_accuracy]`


## Inference
1. To make inference using the model on a single or multiple comma seperate inputs with latest model.

    `python lstm_predictor.py -t "snap-a1 snap-a2 snap-a3, snap-b1 snap-b2"`

2. To make inference using the model on a input file with each line as a list of Snaps with latest model.

    `python lstm_predictor.py -f [path_to_text_file]`

3. To make inference using the model on a input file with each line as a list of Snaps.

    `python lstm_predictor.py -f [path_to_text_file] -m [path_to_model]`

