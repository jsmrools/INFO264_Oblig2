
#defines models(RF, SVM, LR, KNN) and returns them with given hyperparameters

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

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

# src/model_builder.py
# i want to account for randomseed in PCA pipelines as well
def buildmodel_pca(name, randomseed: int = 7, pca_solver: str = 'full'):
    """
    Build a Pipeline: StandardScaler -> PCA -> classifier, with deterministic seeds.
    pca_solver: 'full' (deterministic) or 'randomized' (faster for large k; keep randomseed fixed).
    input:
        name - string specifying the type of model ('SVM_PCA', 'KNN_PCA', 'LR_PCA', 'RF')
        randomseed - integer for random state (default is 7)
        pca_solver - string specifying PCA solver type ('full' or 'randomized')
    """

    if name == 'SVM_PCA':
        return Pipeline([
            ('scale', StandardScaler(with_mean=True)), #Pseudocode: 𝑋𝑠=(𝑋−𝜇)/𝜎Xs
            ('pca', PCA(random_state=randomseed, svd_solver=pca_solver)),
            ('clf', SVC())
        ])
    if name == 'KNN_PCA':
        return Pipeline([
            ('scale', StandardScaler()),
            ('pca', PCA(random_state=randomseed, svd_solver=pca_solver)),
            ('clf', KNeighborsClassifier())
        ])
    if name == 'LR_PCA':
        return Pipeline([
            ('scale', StandardScaler()),
            ('pca', PCA(random_state=randomseed, svd_solver=pca_solver)),
            ('clf', LogisticRegression(max_iter=2000))
        ])
    if name == 'RF':
        # no scaler/PCA for trees by default (as they are invariant to monotonic transformations)<-copilots suggestion (explained more in report)
        return RandomForestClassifier(random_state=randomseed)
    # keep your other originals as needed...
