#!/usr/bin/env python

from pytorch_mlr.mlr import MLR
from pytorch_mlr import seq_collections
from pytorch_mlr import kmer_collections as kmers

import sys
from pprint import pprint
import time

import numpy as np
from sklearn.model_selection import train_test_split
#from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model.tests import test_sag

#import torch


__author__ = "amine"


if __name__ == "__main__":

    seq_file = sys.argv[1]
    cls_file = sys.argv[2]
    reg_penalty = sys.argv[3] # none, l1, l2 or elasticnet
    device = sys.argv[4]   # cpu or cuda or cuda:0

    k=6

    ## Get data
    ###########
    seq_data = seq_collections.SeqCollection((seq_file, cls_file))

    seq_cv_kmers = kmers.SeenKmersCollection(seq_data, k=k, sparse="no")
    X = seq_cv_kmers.data
    y = np.asarray(seq_data.labels)

    print("X shape {}".format(X.shape))

    X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.33, shuffle=True, random_state=42)
   
    ## Hyper-parameters
    ###################

    #penalty = 'elasticnet'
    penalty = 'none'
    alpha = 10 * X_train.shape[0]
    l1_ratio = 0.5
    max_iter = 1000
    learning_rate = test_sag.get_step_size(X_train, alpha, True, True)

    print("\nTorch MLR with {}".format(penalty))

    pt_mlr = MLR(max_iter=max_iter, penalty=penalty, verbose=1, alpha=alpha,
            batch_size=1, learning_rate=learning_rate, n_jobs=4, tol=0, 
            l1_ratio=l1_ratio, device=device)

    start = time.time()
    pt_mlr.fit(X_train, y_train)
    end = time.time()
    print("Fit time: {}".format(end - start))

    y_pred = pt_mlr.predict(X_test)
    pt_score = classification_report(y_test, y_pred)

    print("\nPytorch_mlr scores:\n {}\n".format(pt_score))

    print("\nScikit MLR with {}".format(penalty))

    sk_mlr = LogisticRegression(multi_class="multinomial", max_iter=max_iter, 
            solver="saga", tol=0, penalty=penalty, C=1./alpha,
            l1_ratio=l1_ratio)

    start = time.time()
    sk_mlr.fit(X_train, y_train)
    end = time.time()
    print("Fit time: {}".format(end - start))

    y_pred = sk_mlr.predict(X_test)
    sk_score = classification_report(y_test, y_pred)

    print("\nScikit_mlr scores:\n {}".format(sk_score))

