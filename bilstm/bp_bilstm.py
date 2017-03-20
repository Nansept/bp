# coding: utf-8

import time
import numpy as np
import pandas as pd
import tensorflow as tf

import bp_input

class BPInput(object):
    def __init__(self, batch_size, num_steps, data, label):
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.epoch_size = (data.shape[0] // batch_size - 1) // num_steps
        self.inputs, self.targets = bp_input.input_producer(
            data, label, batch_size, num_steps)


class BPModel(object):
    def __init__(self, is_training, hidden_size, keep_prob, 
                 num_classes, num_layers, input_):
        self.input_ = input_
        # forward lstm layers
        lstm_fw = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size,)
        if is_training and keep_prob < 1:
            lstm_fw = tf.contrib.rnn.DropoutWrapper(lstm_fw,
                output_keep_prob=keep_prob)
        multi_lstm_fw = tf.contrib.rnn.MultiRNNCell([lstm_fw] * num_layers)
        self.init_state_fw = multi_lstm_fw.zero_state(batch_size, tf.float32)
        # backward lstm layers
        lstm_bw = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size,)
        if is_training and keep_prob < 1:
            lstm_bw = tf.contrib.rnn.DropoutWrapper(lstm_bw,
                output_keep_prob=keep_prob)
        multi_lstm_bw = tf.contrib.rnn.MultiRNNCell([lstm_bw] * num_layers)
        self.init_state_bw = multi_lstm_bw.zero_state(batch_size, tf.float32)
        # bilstm
        outputs, state_fw, state_bw = tf.contrib.rnn.static_bidirectional_rnn(
            cell_fw=multi_lstm_fw, cell_bw=multi_lstm_bw, 
            inputs=tf.unstack(input_.inputs, num=input_.num_steps, axis=1), 
            initial_state_fw = self.init_state_fw, 
            initial_state_bw = self.init_state_bw, 
            dtype=tf.float32)
        self.final_state_fw, self.final_state_bw = state_fw, state_bw
        # softmax layer
        softmax_w = tf.get_variable("softmax_w", [2*hidden_size, num_classes],
            dtype=tf.float32)
        softmax_b = tf.get_variable("softmax_b", [num_classes], dtype=tf.float32)
        output = tf.reshape(tf.concat(outputs, 1), [-1, 2 * hidden_size])
        # logits layer
        self.logits = logits = tf.matmul(output, softmax_w) + softmax_b
        # prediction
        self.prediction = tf.argmax(logits, 1)
        # loss and cost
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            logits=[logits],
            targets=[tf.reshape(input_.targets, [-1])],
            weights=[tf.ones([batch_size * num_steps], dtype=tf.float32)])
        self.cost = cost = tf.reduce_sum(loss) / batch_size
        # train op
        if not is_training:
            return
        optimizer = tf.train.GradientDescentOptimizer(0.1)
        self.train_op = optimizer.minimize(cost)


def run_epoch(sess, model, eval_op=None, verbose=False):
    """Run an epoch"""
    start_time = time.time()
    costs = 0.0
    iters = 0.0
    preds = []
    state_fw = sess.run(model.init_state_fw)
    state_bw = sess.run(model.init_state_bw)
    fetches = {
        "cost": model.cost,
        "final_state_fw": model.final_state_fw,
        "final_state_bw": model.final_state_bw,
        "prediction": model.prediction,
    }
    if eval_op is not None:
        fetches["eval_op"] = eval_op
    # run a epoch
    for step in range(model.input_.epoch_size):
        feed_dict = {}
        for i, (c, h) in enumerate(model.init_state_fw):
            feed_dict[c] = state_fw[i].c
            feed_dict[h] = state_fw[i].h
        for i, (c, h) in enumerate(model.init_state_bw):
            feed_dict[c] = state_bw[i].c
            feed_dict[h] = state_bw[i].h
        # 
        vals = sess.run(fetches, feed_dict)
        costs += vals["cost"]
        state_fw = vals["final_state_fw"]
        state_bw = vals["final_state_bw"]
        iters += model.input_.num_steps
        preds.append(vals["prediction"])
        # print 
        if verbose and step % (model.input_.epoch_size // 10) == 0:
            print("%.3f perplexity: %.3f speed: %.0f wps" %
                (step * 1.0 / model.input_.epoch_size, np.exp(costs / iters),
                 iters * model.input_.batch_size / (time.time() - start_time)))
    return np.exp(costs / iters), preds
    
    
batch_size = 5
num_steps = 10
hidden_size = 300
num_classes = 5
max_epoch = 200
num_layers = 2
(train_data, train_label), (test_data, test_label) = bp_input.bp_raw_data()
assert hidden_size == train_data.shape[-1]
assert hidden_size == test_data.shape[-1]
    
    
with tf.Graph().as_default():
    initializer = tf.random_uniform_initializer(-0.1, 0.1)
    train_input = BPInput(batch_size, num_steps, train_data, train_label)
    with tf.variable_scope("Model", reuse=None, initializer=initializer):
        m = BPModel(is_training=True, hidden_size=hidden_size, keep_prob=0.5, 
                    num_classes=num_classes, num_layers=num_layers,
                    input_=train_input)
            
    test_input = BPInput(batch_size, num_steps, test_data, test_label)
    with tf.variable_scope("Model", reuse=True, initializer=initializer):
        mtest = BPModel(is_training=False, hidden_size=hidden_size,
                        keep_prob=1.0, num_classes=num_classes,
                        num_layers=num_layers, input_=test_input)
    ##
    sv = tf.train.Supervisor(logdir="log")
    with sv.managed_session() as sess:
        #for i in range(max_epoch):
        for i in range(30):
            print("epoch: %4d" % i)
            train_perplexity, train_p = run_epoch(sess, m, eval_op=m.train_op,
                                         verbose=False)
            print("Epoch: %4d Train Perplexity: %.3f" % (i, train_perplexity))
            #
            if (i % 10 == 0) or (i + 1 == max_epoch):
            #if (i % 1 == 0) or (i + 1 == max_epoch):
                test_perplexity, test_p = run_epoch(sess, mtest)
                print("Test Perplexity: %.3f" % test_perplexity)

# analyse result             
train_p = np.hstack(train_p)   
train_result = pd.DataFrame(data={"pred": train_p, 
    "label": train_label[:train_p.shape[0]]})          

train_result["acc"] = train_result["label"] == train_result["pred"]
train_acc = 1.0 * train_result["acc"].sum() / train_result.shape[0]

train_result_pos = train_result[train_result["label"] > 0]
train_pos_acc = 1.0 * train_result_pos["acc"].sum() / train_result_pos.shape[0]


test_p = np.hstack(test_p)
test_result = pd.DataFrame(data={"pred": test_p, 
    "label": test_label[:test_p.shape[0]]})  
test_result["acc"] = test_result["label"] == test_result["pred"]
test_acc = 1.0 * test_result["acc"].sum() / test_result.shape[0]

test_result_pos = test_result[test_result["label"] > 0]
test_pos_acc = 1.0 * test_result_pos["acc"].sum() / test_result_pos.shape[0]



if __name__ == "__main__":
    batch_size = 5
    num_steps = 10
    hidden_size = 300
    num_classes = 5
    max_epoch = 50
    num_layers = 2
    (train_data, train_label), (test_data, test_label) = bp_input.bp_raw_data()
    assert hidden_size == train_data.shape[-1]
    assert hidden_size == test_data.shape[-1]
    
    
    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-0.1, 0.1)
        train_input = BPInput(batch_size, num_steps, train_data, train_label)
        with tf.variable_scope("Model", reuse=None, initializer=initializer):
            m = BPModel(is_training=True, hidden_size=hidden_size, keep_prob=0.5, 
                        num_classes=num_classes, num_layers=num_layers,
                        input_=train_input)
                
        test_input = BPInput(batch_size, num_steps, test_data, test_label)
        with tf.variable_scope("Model", reuse=True, initializer=initializer):
            mtest = BPModel(is_training=False, hidden_size=hidden_size,
                            keep_prob=1.0, num_classes=num_classes,
                            num_layers=num_layers, input_=test_input)

        sv = tf.train.Supervisor(logdir="log")
        with sv.managed_session() as sess:
            for i in range(max_epoch):
                print("epoch: %4d" % i)
                train_perplexity, train_p = run_epoch(sess, m, eval_op=m.train_op,
                                             verbose=True)
                print("Epoch: %4d Train Perplexity: %.3f" % (i, train_perplexity))
            
                if (i % 10 == 0) or (i + 1 == max_epoch):
                    test_perplexity, test_p = run_epoch(sess, mtest)
                    print("Test Perplexity: %.3f" % test_perplexity)




