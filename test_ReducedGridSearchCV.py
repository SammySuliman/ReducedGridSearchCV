from sklearn import svm  # Example model
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import time
from sklearn.datasets import make_moons, make_circles
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

from ReducedGridSearchCV import ReducedGridSearchCV

def test1():
    # Example dataset: Iris dataset
    data = load_iris()
    X = data.data
    y = data.target

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Example model
    model = svm.SVC()

    # Define the parameter grid
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],              # Regularization parameter
        'kernel': ['linear', 'rbf', 'poly'],                            # Kernel type
        'degree': [1, 2, 3, 4, 5, 6],                 # Degree for polynomial kernel
        'gamma': [0.001, 0.01, 0.1, 1, 10.0],       # Kernel coefficient
        'shrinking': [True, False],          # Use shrinking heuristic
        'class_weight': [None, 'balanced'] # Adjust weights
    }

    # Perform timed standard grid search
    start = time.time()
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
    grid_search.fit(X, y)
    end = time.time()

    print('Elapsed time: ', end - start)
    print('best params', grid_search.best_params_)
    print('best score (training)', grid_search.best_score_)

    # Evaluate the best model
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    print("Test set accuracy: ", accuracy_score(y_test, y_pred))

    # Perform timed modified grid search
    start = time.time()
    reduced_grid_search = ReducedGridSearchCV(estimator=model, param_grid=param_grid, cv=5)
    reduced_grid_search.fit(X_train, y_train)
    end = time.time()

    print('Elapsed time: ', end - start)
    print('best params', reduced_grid_search.best_params_)
    print('best score (training)', reduced_grid_search.best_score_)

    # Evaluate the best model
    best_model = reduced_grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    print("Test set accuracy: ", accuracy_score(y_test, y_pred))

def test2():

    X, y = make_moons(n_samples=1000, noise=0.3, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the neural network and parameter grid
    mlp = MLPClassifier(max_iter=2000, random_state=42)

    param_grid = {
        'hidden_layer_sizes': [(10,), (50,), (10, 10), (50, 50)],
        'activation': ['relu', 'tanh'],
        'learning_rate_init': [0.001, 0.01, 0.1],
        'solver': ['adam', 'sgd']
    }

    # Perform GridSearchCV
    start = time.time()
    grid_search = GridSearchCV(estimator=mlp, param_grid=param_grid, cv=3, verbose=2, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    end = time.time()

    # Best parameters and model performance
    best_model = grid_search.best_estimator_
    print('best params', grid_search.best_params_)
    print('best score (training)', grid_search.best_score_)

    # Evaluate the best model
    y_pred = best_model.predict(X_test)
    print("Test set accuracy: ", accuracy_score(y_test, y_pred))

    print('Elapsed time: ', end - start)

    # Perform ReducedGridSearchCV
    start = time.time()
    reduced_grid_search = ReducedGridSearchCV(estimator=mlp, param_grid=param_grid, cv=3)
    reduced_grid_search.fit(X_train, y_train)
    end = time.time()

    # Best parameters and model performance
    best_model = reduced_grid_search.best_estimator_
    print('best params', reduced_grid_search.best_params_)
    print('best score (training)', reduced_grid_search.best_score_)

    # Evaluate the best model
    y_pred = best_model.predict(X_test)
    print("Test set accuracy: ", accuracy_score(y_test, y_pred))

    print('Elapsed time: ', end - start)

test1()
test2()