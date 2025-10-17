import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

def explore_data(X, y):
    """
    Function to explore the dataset.
    input: d - X,y data as a numpy array
    output: None (prints visual implementation of the dataset and class distribution)
    """
    # Explore the dataset
    print("Shape of X:", X.shape)  
    print("Shape of y:", y.shape)
    print("Number of classes:", len(np.unique(y)))
    print("Unique labels:", np.unique(y))

    # Class distribution in numbers:
    unique, counts = np.unique(y, return_counts=True)
    class_counts = pd.DataFrame({"Class": unique, "Count": counts})
    print(class_counts)

def split_data_into_3sets(X, y, randomseed = 7): #failsafe of 7 even though already set globally
    """
    Function to split the dataset into training val and testing sets.
    input: 
        X - feature data as a numpy array
        y - labels as a numpy array
        train_ratio - ratio of training data (default is 0.7)
    output: X_train, y_train, X_val, y_val, X_test, y_test
    """
    X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, test_size=0.3, random_state=randomseed, shuffle=True, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, random_state=randomseed, shuffle=True, stratify=y_val_test)
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def showimage(X, index):
    """
    Function to display an image from the dataset.
    input: 
        X - feature data as a numpy array
        index - index of the image to display
    output: None (displays the image)
    """
    for k in range(index):
        plt.imshow(X[k].reshape(20,20), vmin=0, vmax=255, cmap="gray")
        plt.show()