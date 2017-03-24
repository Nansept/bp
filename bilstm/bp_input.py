# coding: utf-8

import numpy as np
import pandas as pd
import tensorflow as tf

def bp_raw_data(test_size=0.1):
    # read data from file
    data = np.load("../Embedding3D.npy")
    sequence_length = data.any(axis=2).sum(axis=1)
    l = np.genfromtxt("../EmbeddingLabel.csv", delimiter=",", usecols=[1], dtype=np.int32)
    l += 1
    label = np.zeros(data.shape[0: 2], dtype=np.int32)
    y_begin = 0
    for i, m in enumerate(sequence_length):
        y_end = y_begin + m
        label[i][:m] = l[y_begin: y_end]
        y_begin = y_end
        
    # cut 1 / 10 data from tail as test data
    assert test_size > 0, "test_size shold be a positive number"
    
    size = data.shape[0]
    if test_size < 1:
        test_size = int(size * test_size)
    
    train_data = data[: -test_size]
    train_label = label[: -test_size]    
    test_data = data[-test_size: ]
    test_label = label[-test_size: ]   
     
    return (train_data, train_label), (test_data, test_label)

class BPInput(object):
    def __init__(self, batch_size, data, label):
        self.data_size, self.num_steps, self.hidden_size = data.shape
        self.batch_size = batch_size
        self.epoch_size = self.data_size // batch_size
        data = tf.convert_to_tensor(data, name="data", dtype=tf.float32)
        label = tf.convert_to_tensor(label, name="label", dtype=tf.int32)
        i = tf.train.range_input_producer(self.epoch_size, shuffle=False).dequeue()
        self.inputs = tf.strided_slice(data, [i * batch_size, 0, 0], [(i + 1) * batch_size, self.num_steps, self.hidden_size])
        self.inputs.set_shape([batch_size, self.num_steps, self.hidden_size])
        self.targets = tf.strided_slice(label, [i * batch_size, 0], [(i + 1) * batch_size, self.num_steps])
        self.targets.set_shape([batch_size, self.num_steps])
             


