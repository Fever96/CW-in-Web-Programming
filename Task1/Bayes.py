import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

#data preprocessing
def data_preprocess():
    path = os.getcwd()
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
    print(m)
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

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    #for i in y:
    #    print(i)

def Bayse():
    path = os.getcwd()
    data = np.loadtxt(path + "/Housing Data Set/housing_new.data",dtype="float128")
    a = 0.0001  # learn rate
    y = data[:, :1]
    x = data[:, 1:]
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    clf = GaussianNB()
    def Bys_learn():
        clf.fit(X_train, y_train)
    Bys_learn()
    def Bys_predict():
        clf.predict(X_test)
        #print(pred)

    Bys_predict()
    # 准确度评估 评估正确/总数
    accuracy = clf.score(X_test, y_test)
    results=open(path+"/result.txt","a",encoding="utf-8")
    results.write("Bayes Accuary: "+str(accuracy)+"\n")
    print("Accuary: "+str(accuracy))

if __name__ == '__main__':
    #data_preprocess()
    Bayse()