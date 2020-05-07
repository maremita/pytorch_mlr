#!/usr/bin/env python

from pytorch_mlr.pytorch_mlr import MLR
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

from matplotlib import pyplot as plt

__author__ = "amine"


def plot_losses(losses, fig_file):

    plt.figure(figsize=(10,5))
    #plt.title("")

    for loss in losses:
        plt.plot(loss["loss"],label=loss["label"])

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    #plt.show()
    plt.savefig(fig_file, bbox_inches="tight")

if __name__ == "__main__":

    seq_file = sys.argv[1]
    cls_file = sys.argv[2]
    k = int(sys.argv[3])
    reg_penalty = sys.argv[4] # none, l1, l2 or elasticnet
    device = sys.argv[5]   # cpu or cuda or cuda:0

    print("K {}".format(k))

    ## Get data
    ###########
    seq_data = seq_collections.SeqCollection((seq_file, cls_file))

    seq_cv_kmers = kmers.SeenKmersCollection(seq_data, k=k, dtype=np.int64, sparse=None)
    X = seq_cv_kmers.data
    y = np.asarray(seq_data.labels)

    print("X shape {}".format(X.shape), flush=True)

    X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.33, shuffle=True, random_state=41, stratify=y)

    ## Hyper-parameters
    ###################

    #penalty = 'elasticnet'
    penalty = reg_penalty
    alpha = 1
    l1_ratio = 0.5
    max_iter = 100
    val_ratio = 0
    save_losses = True

    learning_rate = test_sag.get_step_size(X_train, alpha, True, True)
    print("Learning rate = {}".format(learning_rate), flush=True)

    print("\nTorch MLR with {}".format(penalty), flush=True)

    pt_mlr = MLR(max_iter=max_iter, penalty=penalty, alpha=alpha,
            learning_rate=learning_rate, tol=0, validation=val_ratio,
            n_iter_no_change=max_iter, l1_ratio=l1_ratio, device=device,
            random_state=None, keep_losses=save_losses, verbose=2)

    start = time.time()
    pt_mlr.fit(X_train, y_train)
    end = time.time()
    print("Fit time: {}".format(end - start), flush=True)

    y_pred = pt_mlr.predict(X_test)
    pt_score = classification_report(y_test, y_pred)
    print("\nPytorch_mlr scores:\n {}\n".format(pt_score), flush=True)

    if save_losses:
        losses = []
        losses.append({"loss":pt_mlr.train_losses_, "label":"train loss"})
        if val_ratio:
            losses.append({"loss":pt_mlr.val_losses_, "label":"validation loss"})   
 
        plot_losses(losses, "mlr_{}.png".format(penalty))

    print("\nScikit MLR with {}".format(penalty), flush=True)

    sk_mlr = LogisticRegression(multi_class="multinomial", 
            max_iter=max_iter, solver="saga", tol=1e-10, penalty=penalty,
            C=1./alpha, l1_ratio=l1_ratio)

    start = time.time()
    sk_mlr.fit(X_train, y_train)
    end = time.time()
    print("Fit time: {}".format(end - start), flush=True)

    y_pred = sk_mlr.predict(X_test)
    sk_score = classification_report(y_test, y_pred)

    print("\nScikit_mlr scores:\n {}".format(sk_score), flush=True)

