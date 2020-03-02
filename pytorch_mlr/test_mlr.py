from mlr import MLR

from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model.tests import test_sag

import torch

#torch.manual_seed(0)


iris = load_iris()
X = iris.data
y = iris.target

#X = X[y!=0]
#y = y[y!=0]


#penalty = 'elasticnet'
penalty = 'l1'
alpha = 1/X.shape[0]
l1_ratio = 0.5
max_iter = 1000
learning_rate = test_sag.get_step_size(X, alpha, True, True)
#learning_rate = 0.001 

print("Learning_rate {}\n".format(learning_rate))


print("\nTorch MLR with {}".format(penalty))

my = MLR(max_iter=max_iter, penalty=penalty, verbose=1, alpha=alpha,
        batch_size=1, learning_rate=learning_rate, n_jobs=0, tol=0, 
        l1_ratio=l1_ratio)

my.fit(X,y)

print(my.coef_)


print("\nScikit MLR with {}".format(penalty))

mlr = LogisticRegression(multi_class="multinomial", max_iter=max_iter, 
        solver="saga", tol=0, penalty=penalty, C=1./alpha,
        l1_ratio=l1_ratio)

mlr.fit(X,y)

print(mlr.coef_)

print("\nScikit SGD LR with {}".format(penalty))

sgd = SGDClassifier(loss="log", max_iter=max_iter, tol=0, penalty=penalty,
        alpha=alpha, learning_rate='constant', eta0=learning_rate,
        l1_ratio=l1_ratio)

sgd.fit(X,y)

print(sgd.coef_)

