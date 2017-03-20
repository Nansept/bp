# coding: utf-8

import numpy as np
import pandas as pd
import tensorflow as tf

def bp_raw_data():
    # read dat from file
    data = np.load("../labeledEmbedding.npy")
    label = np.genfromtxt("../EmbeddingLabel.csv", usecols=[1], 
            delimiter=",", dtype=np.int32)

    # cut 1 / 10 data from tail as test data
    size = data.shape[0]
    test_size = size // 10
    
    train_data = data[: -test_size]
    train_label = label[: -test_size]    
    
    test_data = data[-test_size: ]
    test_label = label[-test_size: ]   
     
    return (train_data, train_label), (test_data, test_label)


def input_producer(data, label, batch_size, num_steps):

    epoch_size = (data.shape[0] // batch_size - 1) // num_steps
    batch_len = data.shape[0] // batch_size
    vec_len = data.shape[1]
    data = np.reshape(data[0: batch_size*batch_len], 
                      [batch_size, batch_len, vec_len])
    label = label[0: batch_size*batch_len].reshape(batch_size, batch_len)
    epoch_size = (batch_len - 1) // num_steps

    data = tf.convert_to_tensor(data, name="data", dtype=tf.float32)
    label = tf.convert_to_tensor(label, name="label", dtype=tf.int32)
    assertion = tf.assert_positive(
        epoch_size,
        message="epoch_size == 0, decrease batch_size or num_steps")
    with tf.control_dependencies([assertion]):
        epoch_size = tf.identity(epoch_size, name="epoch_size") 
    i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
    x = tf.strided_slice(data, [0, i * num_steps, 0], 
                         [batch_size, (i + 1) * num_steps, vec_len])
    x.set_shape([batch_size, num_steps, vec_len])
    y = tf.strided_slice(label, [0, i * num_steps], 
                         [batch_size, (i + 1) * num_steps])
    y.set_shape([batch_size, num_steps])
    return x, y              


