import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.base import clone
import itertools
import time

def sort_params(dictionary):
    values = list(dictionary.values())
    keys = list(dictionary.keys())
    nums = 0
    sorted_values = []
    sorted_keys = []
    for i, l in enumerate(values):
        if type(l[0]) == float:
            k = keys[i]
            sorted_values.insert(0, l)
            sorted_keys.insert(0, k)
            nums += 1
        elif type(l[0]) == int:
            k = keys[i]
            sorted_values.insert(1, l)
            sorted_keys.insert(1, k)
            nums += 1
        else:
            k = keys[i]
            sorted_keys.append(k)
            sorted_values.append(l)
    return sorted_values, sorted_keys, nums

def fracturing(nested_list, index):
    nested_list2 = []
    tup_inner_list = []
    last_val = nested_list[0][index]
    for tup in nested_list:
        if tup[index] == last_val:
            tup_inner_list.append(tup)
        else:
            nested_list2.append(tup_inner_list)
            tup_inner_list = [tup]
        last_val = tup[index]
    nested_list2.append(tup_inner_list)
    return nested_list2

def find_first_param(nested_list, depth=0):
    unnested_list = nested_list[0]
    depth += 1
    if type(unnested_list) == tuple:
        return unnested_list
    else:
        return find_first_param(unnested_list, depth)
    
def fracturing2(params, nums):
    """
    Divides list of hyperparameter options based on hyperparamer type.
    """
    index = 0
    while index < nums:
        new_params = []
        if type(params[0]) == list:
            for l in params:
                l2 = fracturing(l, index)
                new_params.append(l2)
        else:
            new_params = fracturing(params, index)
        params = new_params
        index += 1
    return params

def search_best_params(X, y, new_param_grid, estimator, param_keys, downtrend_limit, cv, scoring,
                       best_score=-np.inf, best_params=None, downtrend=0, plateau_threshold=0.05):
    
    if type(new_param_grid[0]) == tuple:
        #print('last call?')
        final_scores = []
        for param_values in new_param_grid:
            model = clone(estimator)
            params = dict(zip(param_keys, param_values))
            model.set_params(**params)
            scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
            mean_score = np.mean(scores)
            final_scores.append(mean_score)
        best_scores = np.max(final_scores)
        best_score_index = np.argmax(final_scores)
        best_params = new_param_grid[best_score_index]
        return best_params, best_scores
        
    for index, param_list in enumerate(new_param_grid):
        
        param_values = find_first_param(param_list)
        model = clone(estimator)
        params = dict(zip(param_keys, param_values))
        model.set_params(**params)
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
        mean_score = np.mean(scores)
        if mean_score > best_score and np.abs(mean_score - best_score) > plateau_threshold:
            #print('successful iteration #', index)
            #print('new best score', mean_score)
            best_score = mean_score
            best_params = params
            downtrend = 0
        else:
            #print('failed iteration #', index)
            downtrend += 1
        #print('downtrend', downtrend)
        if downtrend > downtrend_limit:
            #print('is this the one?')
            optimized_param_grid = new_param_grid[index - downtrend_limit - 1]
            return search_best_params(X, y, optimized_param_grid, estimator, param_keys, downtrend_limit,
                                      cv, scoring, best_score, best_params)
    else:
        optimized_param_grid = new_param_grid[-1]
        return search_best_params(X, y, optimized_param_grid, estimator, param_keys, downtrend_limit,
                                  cv, scoring, best_score, best_params)