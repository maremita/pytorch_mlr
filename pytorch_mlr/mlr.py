import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import LabelEncoder

from torch.utils import data as utils_data


class linear_layer(nn.Module):

    def __init__(self, n_classes, n_features, bias):
        super(linear_layer, self).__init__()

        # to check dtype dependently to input data
        self.linear = nn.Linear(n_features, n_classes, bias=bias).double()

        with torch.no_grad():
            self.linear.weight.zero_()
            if bias:
                self.linear.bias.zero_()

    def forward(self, x):
        return self.linear(x)


class MLR(BaseEstimator, ClassifierMixin):

    def __init__(self, penalty='l2', tol=1e-4, alpha=1.0, l1_ratio=0., 
            learning_rate=0.001, fit_intercept=True, intercept_scaling=1, 
            class_weight=None, random_state=None, solver='sgd', max_iter=100,
            n_jobs=0, batch_size=1, device='cpu', verbose=0):

        self.penalty = penalty
        self.tol = tol
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.learning_rate=learning_rate
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.class_weight = class_weight
        self.random_state = random_state
        self.solver = solver
        self.max_iter = max_iter
        self.n_jobs = n_jobs
        self.batch_size = batch_size
        self.device = device
        self.verbose = verbose

    def fit(self, X, y):

        self.device_ = torch.device(self.device)

        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
 
        le = LabelEncoder()
        encoded_y = le.fit_transform(y).astype(np.long, copy=False)

        X = torch.from_numpy(X).double().to(self.device_)
        encoded_y = torch.from_numpy(encoded_y).long().to(self.device_)
        
        # Store the classes seen during fit
        self.classes_ = np.unique(y)

        n_classes = len(self.classes_)
        n_samples, n_features = X.shape

        if n_classes < 2:
            raise ValueError("This solver needs samples of at least 2 "
                    " classes in the data, but the data contains only "
                    " one class: {}".format(classes_[0]))

        # Check fit_intercept (True or False)
        _bias = self.fit_intercept 

        # Check self.class_weight
        # It has to be a Tensor
 
        # initialize a linear model
        self.model = linear_layer(n_classes, n_features, _bias)

        # Check penalty type
        self.check_penalty()

        # Define loss function
        self.cross_entropy_loss = nn.CrossEntropyLoss(
                weight=self.class_weight) 

        # Define optimizer
        if self.solver == "sgd":
            self.optimizer = optim.SGD(self.model.parameters(), 
                    lr=self.learning_rate, weight_decay=0)
        else:
            raise NotImplementedError("Only SGD solver is supported")

        # scale alpha by number of samples
        self.alpha /= n_samples

        # Define regularizer
        self.regularizer = self._regularizer()

        # train the model
        self._fit(X, encoded_y)
 
        self.coef_ = self.model.linear.weight.data.numpy()
        #self.intercept_ = self.model.linear.bias.data.numpy()

        # Return the classifier
        return self

    def _fit(self, X, y):
        n_iter = 0

        X_y = utils_data.TensorDataset(X, y)
        X_y_loader = utils_data.DataLoader(X_y, batch_size=self.batch_size,
                shuffle=True, num_workers=self.n_jobs)

        previous_w = torch.zeros(self.model.linear.weight.shape).detach()

        for epoch in range(self.max_iter):
            for batch_ind, (X_batch, y_batch) in enumerate(X_y_loader):
                # Clear gradients before each batch
                self.model.zero_grad()

                # Run forward pass
                logits = self.model(X_batch)

                # compute the loss
                loss = self.cross_entropy_loss(logits, y_batch)
 
                # regularization
                #loss += self.alpha * self.regularizer()

                # compute gradients
                loss.backward()

                # update parameters
                self.optimizer.step()

                # Regularization (in gradient space)
                self.regularizer()

            # Check if the stopping criteria is reached
            with torch.no_grad():
                #print(loss.item())
                current_w = self.model.linear.weight
 
                max_weight = torch.max(self.model.linear.weight.data)
                max_change = (current_w - previous_w).abs().max().item()

                previous_w.copy_(current_w)

                if ((max_weight != 0 and max_change/max_weight <= self.tol) 
                        or max_weight == 0 and max_change):
                    
                    if self.verbose:
                        print( "Convergence after {} epochs".format(epoch+1))

                    break
                elif self.verbose == 2:
                    print("Epoch {}, change {}".format(epoch+1,
                        max_change/max_weight))

            n_iter +=1

        if self.verbose and n_iter >= self.max_iter:
            print("max_iter reached")
        
        self.n_iter_ = n_iter

    def check_penalty(self):
        # check penalty type 
        penalty_types = ['none', 'l1', 'l2', 'elasticnet'] 
        
        if self.penalty not in penalty_types:
            raise ValueError("Regularization type should be one of these "
                    "values {}; and got {}".format(
                        ", ".join(penalty_types), reg_type))
        
        if self.penalty == 'none':
            self.penalty_type = 0
            self.alpha = 0.0

        elif self.penalty == 'l1':
            self.penalty_type = 1
            self.l1_ratio = 1.0

        elif self.penalty == 'l2':
            self.penalty_type = 2
            self.l1_ratio = 0.0

        elif self.penalty == 'elasticnet':
            self.penalty_type = 3
        
        if self.penalty in ["l1", "elasticnet"]:
            w = self.model.linear.weight.data
            self.q = torch.zeros_like(w).detach()
            self.wuq = torch.zeros_like(w).detach()
            self.u = 0.

    def _regularizer(self):

        if self.penalty_type == 0:
            return lambda : None

        elif self.penalty_type == 1:
            return self._l1

        elif self.penalty_type == 2:
            return self._l2

        elif self.penalty_type == 3:
            return self._elasticnet

    def _l1(self):
        """
        self.model.linear.weight.abs().sum()
        """
        w = self.model.linear.weight.data
        z = w.data.clone().detach()
        lr = self.optimizer.param_groups[0]['lr']
        decay_scale = lr * self.l1_ratio * self.alpha 
        self.u += decay_scale

        # w_i > 0
        self.wuq = w - (self.u + self.q)
        self.wuq.clamp_(0., np.inf)
        w.copy_(w.where(w.le(0.), self.wuq))

        # w_i < 0 
        self.wuq = w + (self.u - self.q)
        self.wuq.clamp_(-np.inf, 0.)
        w.copy_(w.where(w.ge(0.), self.wuq))

        # update q
        self.q += w - z

    def _l2(self):
        """
        self.model.linear.weight.pow(2).sum()
        """
        w = self.model.linear.weight.data
        lr = self.optimizer.param_groups[0]['lr']
        decay_scale = lr * (1 - self.l1_ratio) * self.alpha
        w.add_(-decay_scale, w)

    def _elasticnet(self):
        """
        self.model.linear.weight
        ((1 - self.l1_ratio) * _l2()) + (self.l1_ratio * _l1())
        """
        self._l2()
        self._l1()

    def decision_function(self, X):
        """
        Predict confidence scores for samples.
        The confidence score for a sample is the signed distance of that
        sample to the hyperplane.
        Parameters
        ----------
        X : array_like or sparse matrix, shape (n_samples, n_features)
            Samples.
        Returns
        -------
        array, shape=(n_samples, n_classes)
            Confidence scores per (sample, class) combination. 
        """

        # check_is_fitted(self)

        if not isinstance(X, torch.DoubleTensor):
            X = torch.as_tensor(X, dtype=torch.double)

        n_features = self.coef_.shape[1]
 
        if X.shape[1] != n_features:
            raise ValueError("X has %d features per sample; expecting %d" 
                    % (X.shape[1], n_features))

        with torch.no_grad():
            logits = self.model(X)

        return logits

    def predict(self, X):
        """
        Predict class labels for samples in X.
        
        Parameters
        ----------
        X : array_like or sparse matrix, shape (n_samples, n_features)
            Samples.
        
        Returns
        -------
        C : array, shape [n_samples]
            Predicted class label per sample.
        """

        logits = self.decision_function(X)
        _, indices = torch.max(logits.data, 1)

        return self.classes_[indices]

    def predict_proba(self, X):
        """
        Probability estimates.
        The returned estimates for all classes are ordered by the
        label of classes.
        The softmax function is used to find the predicted probability of
        each class.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Vector to be scored, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        Returns
        -------
        T : array-like of shape (n_samples, n_classes)
            Returns the probability of the sample for each class in the model,
            where classes are ordered as they are in ``self.classes_``.
        """

        logits = self.decision_function(X)

        return F.softmax(logits, dim=1).numpy()

    def predict_log_proba(self, X):
        """
        Predict logarithm of probability estimates.
        The returned estimates for all classes are ordered by the
        label of classes.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Vector to be scored, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        Returns
        -------
        T : array-like of shape (n_samples, n_classes)
            Returns the log-probability of the sample for each class in the
            model, where classes are ordered as they are in ``self.classes_``.
        """

        logits = self.decision_function(X)

        return F.log_softmax(logits, dim=1).numpy()
