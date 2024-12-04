#!/bin/bash

git clone https://github.com/yandex-research/tabm
pip install torch
pip install 'rtdl_num_embeddings>=0.0.11'

python tabm_eval.py
