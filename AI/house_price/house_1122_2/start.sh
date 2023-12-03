#!/bin/bash

python preprocess.py 
python eval.py "$@"
python train.py "$@"
python save_final_result.py "$@"