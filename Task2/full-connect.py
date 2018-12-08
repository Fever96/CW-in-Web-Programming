import tensorflow as tf
import os
import time
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

def full_connect():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    tf_config = tf.ConfigProto()

    tf_config.gpu_options.allow_growth = True
    mnist = input_data.read_data_sets("MNIST_data",one_hot=True)

    path=os.getcwd()

    f=open(path+"/results_full-connect.txt","a",encoding="utf-8")
    #适当的增加batch_size，会提高模型训练的时间
    batch_size = 128
    n_batch = mnist.train.num_examples // batch_size
    x = tf.placeholder(tf.float32,[None,784])
    y = tf.placeholder(tf.float32, [None,10])
    keep_prob = tf.placeholder(tf.float32)

    w1 = tf.Variable(tf.truncated_normal([784,1024],stddev=0.1))
    b1 = tf.Variable(tf.random_normal([1024]))
    L1 = tf.nn.tanh(tf.matmul(x,w1) + b1)
    L1_drop = tf.nn.dropout(L1,keep_prob)

    w2 = tf.Variable(tf.truncated_normal([1024,1024],stddev=0.1))
    b2 = tf.Variable(tf.random_normal([1024]))
    L2 = tf.nn.relu(tf.matmul(L1_drop,w2) + b2)
    L2_drop = tf.nn.dropout(L2,keep_prob)


    w3 = tf.Variable(tf.truncated_normal([1024,10],stddev=0.1))
    b3 = tf.Variable(tf.random_normal([10]))


    prediction = tf.matmul(L2_drop,w3) + b3

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))

    train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

    init = tf.global_variables_initializer()

    correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

    start = time.time()
    with tf.Session(config = tf_config) as sess:
        sess.run(init)
        for epoch in range(50):
            epoch_start = time.time()
            for batch in range(n_batch):
                batch_xs,batch_ys = mnist.train.next_batch(batch_size)
                sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys,keep_prob:0.75})
            epoch_end = time.time()
            epoch_time = epoch_end - epoch_start
            test_acc = sess.run(accuracy, feed_dict={x:mnist.test.images,y:mnist.test.labels,keep_prob:1.0})
            train_acc = sess.run(accuracy, feed_dict={x: mnist.train.images, y: mnist.train.labels,keep_prob:1.0})
            f.write("Iter:" + str(epoch) + ", Test Acc:" + str(test_acc) + ",Train Acc:" + str(train_acc)+"\n")
            print("Iter:" + str(epoch) + ", Test Acc:" + str(test_acc) + ",Train Acc:" + str(train_acc) + ",Time:"+ str(epoch_time))
    end = time.time()
    print("Total Time:"+str(end - start) + "s")
    print("success")

if __name__ == '__main__':
    full_connect()