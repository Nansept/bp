
# coding: utf-8

# In[1]:

import time
import numpy as np
import pandas as pd
import tensorflow as tf


# In[2]:

# batch size
batch_size = 10
# input vector size
hidden_size = 300
# unroll rnn with truncated bp,
# with a fixed number of number(num_step) cells
num_steps = 10
# forget bias
forget_bias = 1.0
# keep probability
keep_prob = 0.5
# num of layers in multiple layer lstm
num_layers = 2
# num of output classes 
num_classes = 5


# In[3]:

data = np.load("../labeledEmbedding.npy")
assert data.shape[1] == hidden_size, "hidden_size shoule be equal to data.shape[0]"
label = pd.read_csv("../EmbeddingLabel.csv", index_col=0, names=['label']).values
batch_len = data.shape[0] // batch_size
data = np.reshape(data[0: batch_size*batch_len], [batch_size, batch_len, -1])
label = label[0: batch_size*batch_len].reshape(batch_size, -1)
epoch_size = (batch_len - 1) // num_steps


# In[4]:

# input data
def input_producer(data, label, epoch_size, batch_size, num_steps):
    data = tf.convert_to_tensor(data, name="data", dtype=tf.float32)
    label = tf.convert_to_tensor(label, name="label", dtype=tf.int32)
    assertion = tf.assert_positive(
        epoch_size,
        message="epoch_size == 0, decrease batch_size or num_steps")
    with tf.control_dependencies([assertion]):
        epoch_size = tf.identity(epoch_size, name="epoch_size") 
    i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
    x = tf.strided_slice(data, [0, i * num_steps, 0], [batch_size, (i + 1) * num_steps, hidden_size])
    x.set_shape([batch_size, num_steps, hidden_size])
    y = tf.strided_slice(label, [0, i * num_steps], [batch_size, (i + 1) * num_steps])
    y.set_shape([batch_size, num_steps])
    return x, y  


# In[5]:

inputs, labels = input_producer(data, label, epoch_size, batch_size, num_steps)
inputs = tf.unstack(inputs, num=num_steps, axis=1)


# In[6]:

lstm_fw = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, forget_bias=forget_bias)

lstm_fw = tf.contrib.rnn.DropoutWrapper(lstm_fw, output_keep_prob=keep_prob)

multi_lstm_fw = tf.contrib.rnn.MultiRNNCell([lstm_fw] * num_layers)

initial_state_fw = multi_lstm_fw.zero_state(batch_size, tf.float32)

lstm_bw = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, forget_bias=forget_bias)

lstm_bw = tf.contrib.rnn.DropoutWrapper(lstm_bw, output_keep_prob=keep_prob)

multi_lstm_bw = tf.contrib.rnn.MultiRNNCell([lstm_bw] * num_layers)

initial_state_bw = multi_lstm_bw.zero_state(batch_size, tf.float32)


# In[7]:

outputs, output_state_fw, output_state_bw = tf.contrib.rnn.static_bidirectional_rnn(cell_fw=multi_lstm_fw, cell_bw=multi_lstm_bw, inputs=inputs, initial_state_fw = initial_state_fw, initial_state_bw = initial_state_bw,dtype=tf.float32)


# In[8]:

softmax_w = tf.get_variable("softmax_w", [2*hidden_size, num_classes], dtype=tf.float32)
softmax_b = tf.get_variable("softmax_b", [num_classes], dtype=tf.float32)


# In[9]:

output = tf.reshape(tf.concat(outputs, 1), [-1, 2 * hidden_size])
logits = tf.matmul(output, softmax_w) + softmax_b
loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
    logits=[logits],
    targets=[tf.reshape(labels, [-1])],
    weights=[tf.ones([batch_size * num_steps], dtype=tf.float32)])

cost = tf.reduce_sum(loss) / batch_size


# In[10]:

optimizer = tf.train.GradientDescentOptimizer(0.1)
train_op = optimizer.minimize(cost)


# In[11]:

sv = tf.train.Supervisor(logdir="log")


# In[12]:

with sv.managed_session() as sess:
    sess.run(initial_state_fw)

    start_time = time.time()
    costs = 0.0
    iters = 0.0
    
    state_fw = sess.run(initial_state_fw)
    state_bw = sess.run(initial_state_bw)

    for step in range(epoch_size):
        feed_dict = {}
        for i, (c, h) in enumerate(initial_state_fw):
            feed_dict[c] = state_fw[i].c
            feed_dict[h] = state_fw[i].h
        for i, (c, h) in enumerate(initial_state_bw):
            feed_dict[c] = state_bw[i].c
            feed_dict[h] = state_bw[i].h
            
        c, state_fw, state_bw, _ = sess.run(
            [cost, output_state_fw, output_state_bw, train_op],
            feed_dict)
    
        costs += c
        iters += num_steps
    
        if step % (epoch_size // 10) == 0:
            print("%.3f perplexity: %.3f speed: %.0f wps" %
                (step * 1.0 / epoch_size, np.exp(costs / iters),
                 iters * batch_size / (time.time() - start_time)))


# In[ ]:



