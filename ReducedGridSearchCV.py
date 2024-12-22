import numpy as np
import itertools
from sklearn.base import clone

from gridsearch_helper_functions import *

class ReducedGridSearchCV:
    def __init__(self, estimator, param_grid, cv=5, scoring='accuracy'):
        """
        Parameters:
        - estimator: The model to tune (e.g., a scikit-learn classifier or regressor).
        - param_grid: Dictionary or list of dictionaries with hyperparameter ranges.
        - scoring: Function to evaluate model performance (default: estimator's score method).
        - cv: Number of cross-validation folds or a cross-validation generator.
        """
        self.estimator = estimator
        self.param_grid = param_grid
        self.scoring = scoring
        self.cv = cv
        self.best_params_ = None
        self.best_score_ = -np.inf
        
        #num_params = np.max([len(l) for l in self.param_grid])
        num_params = len(self.param_grid)
        self.downtrend_limit = num_params // 2

    def fit(self, X, y):
        """
        Fit the model using grid search to find the best parameters.

        Parameters:
        - X: Feature matrix.
        - y: Target vector.
        """
        best_score = -np.inf
        prev_score = -np.inf
        best_params = None
        trend = 0
        
        values, keys, nums = sort_params(self.param_grid)
        combinations = list(itertools.product(*values))
        new_param_grid = fracturing2(combinations, nums)
        best_param_vals, best_scores = search_best_params(new_param_grid, self.estimator, keys, self.downtrend_limit, self.cv, self.scoring)
        
        self.best_score_ = best_score
        self.best_params_ = dict(zip(keys, best_param_vals))

        # Train the final model with the best parameters
        self.best_estimator_ = clone(self.estimator).set_params(**self.best_params_)
        self.best_estimator_.fit(X, y)

        return self

    def predict(self, X, Y):
        """Make predictions with the best estimator."""
        return self.best_estimator_.predict(X)