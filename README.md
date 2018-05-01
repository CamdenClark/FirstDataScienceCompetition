Basically, my approach here is as follows:

* Train individual models for symbols and numbers
* Segment larger test image into fifths, predict symbols and numbers separately
* For each fifth, generate the set of symbols/numbers where that fifth is that symbol with >= .2 probability.
* Find the cartesian product of each set of symbols/numbers
* For each combination of predicted symbols/numbers (or each unique expression), predict 1 if any of the expressions evaluate to true.
* Predict 0 if none of the predicted symbol/number combinations (expressions) evaluate to true

### To run

* Make a folder called data and put `train.csv`, `train_labels.csv`, and `test.csv` in the folder.
* Run `python process_data.py` and `python score.py`

Thanks!
