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

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model.tests import test_sag

from matplotlib import pyplot as plt

__author__ = "amine"


def plot_losses(losses, styles, fig_file):

    plt.figure(figsize=(10,5))
    #plt.title("")

    for loss in losses:
        plt.plot(loss["loss"], loss["style"], label=loss["label"], color=loss["color"])

    plt.yscale("log")
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


    # scaling data
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    ## Hyper-parameters
    ###################

    #penalty = 'elasticnet'
    penalty = reg_penalty
    alpha = 1
    l1_ratio = 0.5
    max_iter = 1000
    no_change = max_iter
    val_ratio = 0.2
    save_losses = True
    learning_rate = 1e-4
    verbose = 1

    #learning_rate = test_sag.get_step_size(X_train, alpha, True, True)
    #print("Learning rate = {}".format(learning_rate), flush=True)

    ## Without scaling
    print("\nPytorch_MLR with {} without scaling data".format(penalty), flush=True)

    pt_mlr_ns = MLR(max_iter=max_iter, penalty=penalty, alpha=alpha,
            learning_rate=learning_rate, tol=0, validation=val_ratio,
            n_iter_no_change=no_change, l1_ratio=l1_ratio, device=device,
            random_state=None, keep_losses=save_losses, verbose=verbose)

    pt_mlr_ns.fit(X_train, y_train)
    print("Epoch Fit time: {}".format(pt_mlr_ns.epoch_time_), flush=True)
    print("Loss: {}".format(pt_mlr_ns.train_loss_), flush=True)
    print("# iertations: {}".format(pt_mlr_ns.n_iter_), flush=True)

    y_pred = pt_mlr_ns.predict(X_test)
    pt_score = classification_report(y_test, y_pred)
    print("\nPytorch_mlr scores:\n {}\n".format(pt_score), flush=True)


    ## With scaling
    print("\nPytorch_MLR with {} with scaling data".format(penalty), flush=True)

    pt_mlr_ws = MLR(max_iter=max_iter, penalty=penalty, alpha=alpha,
            learning_rate=learning_rate, tol=0, validation=val_ratio,
            n_iter_no_change=no_change, l1_ratio=l1_ratio, device=device,
            random_state=None, keep_losses=save_losses, verbose=verbose)

    pt_mlr_ws.fit(X_train_scaled, y_train)
    print("Epoch Fit time: {}".format(pt_mlr_ws.epoch_time_), flush=True)
    print("Loss: {}".format(pt_mlr_ws.train_loss_), flush=True)
    print("# iertations: {}".format(pt_mlr_ws.n_iter_), flush=True)

    y_pred = pt_mlr_ws.predict(X_test_scaled)
    pt_score = classification_report(y_test, y_pred)
    print("\nPytorch_mlr scores:\n {}\n".format(pt_score), flush=True)


    ## Plot losses
    if save_losses:
        losses = []
        styles = []
        losses.append({"loss":pt_mlr_ns.train_losses_, "label":"train loss without data scaling",
            "style":"-", "color":"blue"})
        losses.append({"loss":pt_mlr_ws.train_losses_, "label":"train loss with data scaling",
            "style":"-", "color":"red"})
        if val_ratio:
            losses.append({"loss":pt_mlr_ns.val_losses_, "label":"validation loss without data scaling",
                "style":"--", "color":"green"})
            losses.append({"loss":pt_mlr_ws.val_losses_, "label":"validation loss with data scaling", 
                "style":"--", "color":"orange"}) 

        plot_losses(losses, styles, "mlr_data_scaling_k{}_{}.png".format(k, penalty))

