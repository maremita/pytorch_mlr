#!/usr/bin/env bash

VIRUS=HCV02

./seq_mlr.py ../data/viruses/$VIRUS/data.fa ../data/viruses/$VIRUS/class.csv 4 l2 cpu
