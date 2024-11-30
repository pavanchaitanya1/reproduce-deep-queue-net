from unicodedata import bidirectional
import pandas as pd
import numpy as np

class BaseConfig:
    test_size=0.2
    sub_rt=0.005
    TIME_STEPS=42
    BATCH_SIZE=32
    modelname='4-port switch/FIFO'
    no_of_port=4
    no_of_buffer=1
    ser_rate=2.5*1024**2
    sp_wgt=0.
    seed=0
    window=63
    no_process=15 
    epochs=100
    n_outputs=1 
    learning_rate=0.001
    l2=0.1
    lstm_params={'layer':2,   'cell_neurons':[200,100],     'keep_prob':1}  
    att=64
    mul_head=3
    mul_head_output_nodes=32
    lstm=1
    bidirectional=1
    use_transformer=1
    transformer_params={'n_heads':2, 'n_layers':2, 'dim_feedforward':100}
    dropout=0.1
    mask=0

class modelConfig:
    scaler='./trained/scaler'  
    model='./trained/model' 
    md=341
    train_sample= './trained/sample/train.h5'
    test1_sample= './trained/sample/test1.h5'
    test2_sample= './trained/sample/test2.h5'
    bins=100
    errorbins='./trained/error'
    error_correction=False
    