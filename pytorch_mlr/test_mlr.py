from mlr import MLR

from sklearn.datasets import load_iris
import torch

#torch.manual_seed(0)


iris = load_iris()
X = iris.data
y = iris.target

penalty = 'none'
print(penalty)

my = MLR(max_iter=1000, penalty=penalty, verbose=1, alpha=1, batch_size=10,
        learning_rate=0.001, n_jobs=0, tol=1e-4)
my.fit(X,y)

print(my.coef_)

