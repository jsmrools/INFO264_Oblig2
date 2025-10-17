#trains a given model on data using cross validation and evaluates it

from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
import numpy as np

# cross validation and model selection functions for training and evaluating models
# just a under the hood of sklearn cross validation with added print statements for clarity
def train_eval_model_cv(model, X_train, y_train, X_val, y_val, seed = 7, cv = 3, scoring='accuracy'):
    """
    Function to train and evaluate A MODEL SINGULAR using k-fold cross-validation.

    """
    #85% of the data is used for training and validation
    combined_X = np.concatenate((X_train, X_val), axis=0)
    combined_y = np.concatenate((y_train, y_val), axis=0)


    kf = KFold(n_splits=cv, shuffle=True, random_state=seed)
    all_scores = []
    print(f'Starting {model} {cv}-fold cross-validation...')
    for (train_index, val_index) in kf.split(combined_X,combined_y):
        X_train, X_val = combined_X[train_index], combined_X[val_index]
        y_train, y_val = combined_y[train_index], combined_y[val_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        if scoring == 'accuracy':
            score = accuracy_score(y_val, y_pred)
        elif scoring == 'f1weighted':
            score = f1_score(y_val, y_pred, average='weighted')
        else:
            raise ValueError("Unsupported scoring method. Currently only 'accuracy', is supported. Because F1 is not relevant as we have preprocessed the data to be balanced.")
        all_scores.append(score)
    
    mean_score = np.mean(all_scores)
    print(f'Mean {scoring} over {cv} folds: {mean_score:.4f}')
    return mean_score
