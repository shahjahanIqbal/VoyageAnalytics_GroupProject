# Gender Classification Module

## Objective
Evaluate whether aggregated flight and hotel travel features
can distinguish the male and female labels in the dataset.

## Dataset
- Labelled users: 900
- Male labels: 452
- Female labels: 448
- Numerical features: 14

## Algorithms Compared
- Dummy Classifier
- Logistic Regression
- Random Forest
- Extra Trees

## Results
- Best cross-validation balanced accuracy: 52.22%
- Test accuracy: 47.78%
- Test balanced accuracy: 47.78%
- Test macro F1: 47.54%
- Test ROC-AUC: 0.4594

## Conclusion
The current features do not demonstrate reliable gender
classification on the held-out test set.

Further feature validation and experimentation are needed.
This model is an exploratory academic baseline and should not
be used to infer an individual's gender.