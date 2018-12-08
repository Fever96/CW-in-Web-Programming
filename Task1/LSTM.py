import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import classification_report
import os
import tensorflow as tf
from tensorflow.python.ops import rnn,rnn_cell
import numpy as np
import sys
import shutil
from sklearn.metrics import accuracy_score

path = os.getcwd()
#data preprocessing
def data_preprocess():
    data = np.loadtxt(path + "/Housing Data Set/housing.data",dtype="float128")
    #a = 0.0001  # learn rate
    y = data[:, 13:]
    x = data[:, 0:13]
    s=[]
    #for i in x:
    #    print(i)
    y=np.argsort(y,axis=0,kind='quicksort',order=None)
    m=y.size
    m1=m/3
    m2=m*2/3
    #print(m)
    for i in y:
        if(i<m1):
            #print("0")
            s.append(0)
        elif(i>m1 and i<m2):
            #print("1")
            s.append(1)
        else:
            #print("2")
            s.append(2)

    f=open(path+"/Housing Data Set/housing_new.data","w",encoding="utf-8")
    #if(os.path.exists(path+"/Housing Data Set/housing_new.data")):
    #    f = open(path + "/Housing Data Set/housing_new.data", encoding="utf-8")
    for i in range(0,len(s)):
        f.write(str(s[i])+" "+str(x[i][0])+" "+str(x[i][1])+" "+str(x[i][2])+" "+str(x[i][3])+" "+str(x[i][4])
                + " "+str(x[i][5])+" "+str(x[i][6])+" "+str(x[i][7])+" "+str(x[i][8])+" "+str(x[i][9])
                + " "+str(x[i][10])+" "+str(x[i][11])+" "+str(x[i][12])+"\n")
    f.close()

def data_preprocess2():
    data = np.loadtxt(path + "/Housing Data Set/housing_new.data",dtype="float128")
    #a = 0.0001  # learn rate
    y = data[:, :1]
    x = data[:, 1:]
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    f1=open(path+"/LSTM/x_train_housing.txt","w",encoding="utf-8")
    f2=open(path+"/LSTM/x_test_housing.txt","w",encoding="utf-8")
    f3=open(path+"/LSTM/label_housing.txt","w",encoding="utf-8")
    f4=open(path+"/LSTM/y_test_housing.txt","w",encoding="utf-8")
    for i in range(0,len(X_train)):
        f1.write(str(X_train[i][0])+" "+str(X_train[i][1])+" "+str(X_train[i][2])+" "+str(X_train[i][3])+" "+
                 str(X_train[i][4]) + " " +str(X_train[i][5])+" "+str(X_train[i][6])+" "+str(X_train[i][7])+" "+
                 str(X_train[i][8]) + " " +str(X_train[i][9])+" "+str(X_train[i][10])+" "+str(X_train[i][11])+" "+
                 str(X_train[i][12])+"\n")
    f1.close()

    for i in range(0,len(X_test)):
        f2.write(str(X_train[i][0])+" "+str(X_train[i][1])+" "+str(X_train[i][2])+" "+str(X_train[i][3])+" "+
                 str(X_train[i][4]) + " " +str(X_train[i][5])+" "+str(X_train[i][6])+" "+str(X_train[i][7])+" "+
                 str(X_train[i][8]) + " " +str(X_train[i][9])+" "+str(X_train[i][10])+" "+str(X_train[i][11])+" "+
                 str(X_train[i][12])+"\n")
    f2.close()

    for i in range(0,len(y_train)):
        #print(len(y_train))
        #print(y_train)
        if(y_train[i][0]==1.0):
            f3.write("0 1 0\n")
        elif(y_train[i][0]==0.0):
            f3.write("1 0 0\n")
        else:
            f3.write("0 0 1\n")

    f3.close()

    for i in range(0,len(y_test)):
        f4.write(str(y_test[i][0])+"\n")

    f4.close()

def LSTM():

    if os.path.exists(path+'/LSTM/model'):
        shutil.rmtree(path+'/LSTM/model')

    os.makedirs(path+'/LSTM/model')


    file_traindata = path + '/LSTM/x_train_housing.txt'
    file_trainlabel = path + '/LSTM/label_housing.txt'
    file_testdata = path + '/LSTM/x_test_housing.txt'
    output_file_path = path + '/LSTM/result_housing.txt'
    output_file = open(output_file_path, 'w+')
    traindata = np.loadtxt(file_traindata)
    trainlabel = np.loadtxt(file_trainlabel)
    testdata = np.loadtxt(file_testdata)
    loss_file=open(path+'/LSTM/loss_housing.txt','w+')
    # model_path = path + "\\model\\base_0_sim_lstm.ckpt"
    model_path = path+"/LSTM/model/lstm.ckpt"

    learning_rate = 0.0001
    # training_iters = 10000
    #print(str(feature))

    n_input = 1  # MNIST data input (img shape: 28*28)
    n_steps = 13  # timesteps
    n_hidden = 32  # hidden layer num of features
    n_classes = 3

# tf Graph input
    x = tf.placeholder("float", [1, n_steps, n_input])
    y = tf.placeholder("float", [1, n_classes])

    # Define weights
    weights = {
        'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
    }
    biases = {
        'out': tf.Variable(tf.random_normal([n_classes]))
    }



    def RNN(x, weights, biases):
        x = tf.transpose(x, [1, 0, 2])
        # used to be (batch_size, n_stpe, n_input)
        # Reshaping to (n_steps*batch_size, n_input)
        x = tf.reshape(x, [-1, n_input])
        # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
        x = tf.split(0, n_steps, x)
        # Define a lstm cell with tensorflow

        lstm_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
        # lstm_cell = rnn_cell.MultiRNNCell(lstm_cell,state_is_tuple=True)

        # Get lstm cell output
        outputs, states = rnn.rnn(lstm_cell, x, dtype=tf.float32)

        # Linear activation, using rnn inner loop last output
        return tf.matmul(outputs[-1], weights['out']) + biases['out']


    pre = RNN(x, weights, biases)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pre, y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    init = tf.global_variables_initializer()
    #sess.run(init)
    #init = tf.initialize_local_variables()
    #init = tf.initialize_all_variables()
    saver = tf.train.Saver()


    #train
    istrain=1
    with tf.Session() as sess:
        sess.run(init)
        if istrain == 0:
            saver.restore(sess, model_path)
        print(traindata.shape[0])
        count = 1
        for i in range(traindata.shape[0]):
            data = traindata[i]
            label = trainlabel[i]
            # print(data.shape)
            data = data.reshape(1, n_steps, n_input)
            # print(label)
            label = label.reshape(1, n_classes)
            # print(data.shape)
            # print(label.shape)

            # print("sess run")
            if istrain == 1:
                sess.run(optimizer, feed_dict={x: data, y: label})
                loss1=sess.run(cost, feed_dict={x: data, y: label})
                loss_file.write(str(i)+' '+str(loss1)+'\n')
                if i % 50 == 0:
                    loss = sess.run(cost, feed_dict={x: data, y: label})
                    print(i, loss)
            else:
                data_test = testdata[i]
                data_test = data_test.reshape(1, n_steps, n_input)
                result = sess.run(pre, feed_dict={x: data_test})

                print(result)
                num = np.argmax(result)
                print(count, num)
                output_file.write(str(num))
                output_file.write('\n')
                count += 1


        if istrain == 1:
            save_path = saver.save(sess, model_path)

    #test
    istrain=0
    with tf.Session() as sess1:
        sess1.run(init)
        if istrain == 0:
            saver.restore(sess1, model_path)
        print(traindata.shape[0])
        count = 1
        for i in range(testdata.shape[0]):
            data = traindata[i]
            label = trainlabel[i]
            # print(data.shape)
            data = data.reshape(1, n_steps, n_input)
            # print(label)
            label = label.reshape(1, n_classes)
            # print(data.shape)
            # print(label.shape)

            # print("sess run")
            if istrain == 1:
                sess1.run(optimizer, feed_dict={x: data, y: label})
                if i % 50 == 0:
                    loss = sess1.run(cost, feed_dict={x: data, y: label})
                    print(i, loss)
            else:
                data_test = testdata[i]
                data_test = data_test.reshape(1, n_steps, n_input)
                result = sess1.run(pre, feed_dict={x: data_test})

                #print(result)
                num = np.argmax(result)
                print(count, num)
                output_file.write(str(num))
                output_file.write('\n')
                count += 1

        if istrain == 1:
            save_path = saver.save(sess, model_path)
        loss_file.close()


def report():
    y_true = np.loadtxt(path + '/LSTM/y_test_housing.txt')
    y_pre = np.loadtxt(path + '/LSTM/result_housing.txt')
    result = open(path + '/LSTM/report_housing.txt', 'w', encoding='utf-8')
    result.write(classification_report(y_true, y_pre))
    labels = [0.0, 1.0, 2.0]

    print(accuracy_score(y_true, y_pre))
    results=open(path+"/result.txt","a",encoding="utf-8")
    results.write("LSTM Accuary: "+str(accuracy_score(y_true, y_pre)))
    print(classification_report(y_true,y_pre,labels=labels))
    result.close()

if __name__ == '__main__':
    #data_preprocess()
    #data_preprocess2()
    #LSTM()
    report()