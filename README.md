# Machine Learning Project
by Sabrina Fonseca Pereira and Maria Sousa

## About the project
The purpose of this project is to implement and use different machine learning models to classify
glass fragments into 6 categories, based on 9 different forensic measurements. 

A neural network and a decision tree were implemented from scratch and compared to the corresponding `scikit-learn` implementation. Other models provided by the `scikit-learn` library were also used as a way to explore their efficacy in solving this classification problem.

## The repository
### code
- `implementations.py` contains the implementation of our machine learning models from scratch
- `project.ipynb` the notebook with model predictions and analysis.

### data
- `df_train.csv` training set
- `df_test.csv` test set

### dt-visualization
- `decision-tree.json` was made from the dictionary outputted by the decision tree training
- `tree.svg` is tree visualisation generated from the json file (generated with https://vanya.jp.net/vtree/)

`report.pdf` the report with project findings and conclusions.

The jupyter notebook contains the function calls, visualisations and tests with the sklearn library. The notebook is dependent on files in the code folder. 