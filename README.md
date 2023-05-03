**f1-fp2-quali-predictor** is an attempt to build a prediction model to predict Formula 1 qualifying results using data from Free Practice 2 sessions.

* The training and test data is extracted using [FastF1](https://github.com/theOehrly/Fast-F1) with official Formula 1 Live Timing as data source.
  * Data from 44 free practice and qualifying sessions from the 2021 and 2022 Formula 1 seasons is used to create and evaluate the model.
  * A total of 844 records are included after incomplete records are filtered out.
  * 80 % of the records are used as training data and 20 % as test data.
* [NumPy](https://github.com/numpy/numpy) and [Pandas](https://github.com/pandas-dev/pandas) are used to process extracted data.
* [scikit-learn](https://scikit-learn.org/stable/) is used to create and evaluate the prediction model.
* A [decision tree algorithm](https://en.wikipedia.org/wiki/Decision_tree_learning) is used to create a human readable model to predict a driver's qualifying position (1-20) using both numerical (e.g. lap time) and categorical (e.g. tyre compound) feature data.

## Results

The mean squared error (MSE) of the resulting prediction model is **25.5118**.

This is significantly better than random guesses which would give an MSE of 571.5. But more data, more optimization and most likely a different model than a decision tree is probably needed to make the prediction model more accurate.