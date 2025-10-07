
#defines models(RF, SVM, LR, KNN) and returns them with given hyperparameters

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

def buildmodel(model_name, randomseed = 7, **kwargs): #source for **kwargs: https://stackoverflow.com/questions/33948321/python-function-accepting-arbitrary-keyword-arguments-and-passing-them-to-another
    """
    Function to build and return a machine learning model based on the specified model name.
    input: 
        model_name - string specifying the type of model ('RF', 'SVM', 'LR', 'KNN')
        randomseed - integer for random state (default is 7)
        arguments - additional keyword arguments for model hyperparameters
    """
    if model_name == 'RF':
        model = RandomForestClassifier(random_state=randomseed, **kwargs)
    elif model_name == 'SVM':
        model = SVC(random_state=randomseed, **kwargs)
    elif model_name == 'LR':
        model = LogisticRegression(random_state=randomseed, max_iter=1000, **kwargs)
    elif model_name == 'KNN':
        model = KNeighborsClassifier(**kwargs)
    else:
        raise ValueError("Unsupported model type. Choose from 'RF', 'SVM', 'LR', or 'KNN'.")

    return model
