The algorithm works by creating a nested array of each possible choice of hyperparameters, organized by having each layer of the nested array change one of the numeric parameters while keeping the rest constant, with the most influential numeric parameter being changed first.

Ex:
[[[[(0.001, 0.001, 1, 'linear', True, None), (0.001, 0.001, 1, 'linear', True, 'balanced'), (0.001, 0.001, 1, 'linear', False, None), ...]...]...], 
[[[(0.01, 0.001, 1, 'linear', True, None), (0.01, 0.001, 1, 'linear', True, 'balanced'), (0.01, 0.001, 1, 'linear', False, None), ...]...]...]...]

And so on down the list:
[[[(0.001, 0.001, 1, 'linear', True, None), (0.001, 0.001, 1, 'linear', True, 'balanced'), (0.001, 0.001, 1, 'linear', False, None), ...],
[(0.001, 0.01, 1, 'linear', True, None), (0.001, 0.01, 1, 'linear', True, 'balanced'), (0.001, 0.01, 1, 'linear', False, None), ...]...]...]

The algorithm then begins by iterating through the first entry of the top layer of the array (where only the most influential numeric hyperparameter is changing and all others are being held fixed) and finding which of the possible choices for this hyperparameter from the parameter grid achieves the best performance. If the performace fails to improve over several iterations, then we are in a "downward trend" (iterating away from the optimal value), and the loop is automatically closed to save time, reverting to the previous best performing model. Then, we set this hyperparameter fixed and enter the next layer of the nested array and continue the same process. When all numeric hyperparameters have been fixed, we perform a normal grid search over the non-numeric hyperparameters.

In order to reduce time spent searching for a better performing model after the current model plateaues (achieving roughly the same performance for each value after a certain point), we add a plateau threshold of 0.05. Each subsequent model must achieve performance better than the current best performing model by greater than the plateau threshold.

VALIDATION: \
Example 1: \
with regular GridSearchCV: \
Accuracy (on training): 0.98 \
Accuracy (on testing): 1.0 \
Elapsed time: 264.8235182762146

with ReducedGridSearchCV: \
Accuracy (on training): 0.95 \
Accuracy (on testing): 1.0 \
Elapsed time: 1.8074696063995361

Example 2: \
with regular GridSearchCV: \
Accuracy (on training): 0.912 \
Accuracy (on testing): 0.905 \
Elapsed time: 20.62173557281494

with ReducedGridSearchCV: \
Accuracy (on training): 0.910 \
Accuracy (on testing): 0.92 \
Elapsed time: 10.864146947860718 
