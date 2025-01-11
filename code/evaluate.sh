#!/bin/bash

seamlessresults=$1
semalessdropoutresults=$2
deltalmresults=$3

TMPDIR=$seamlessresults python evaluations.py
TMPDIR=$semalessdropoutresults python dropout_evaluations.py
TMPDIR=$deltalmresults python dlmeval.py
