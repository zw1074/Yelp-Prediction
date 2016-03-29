import numpy as np
import pandas as pda
import data_tools
import matplotlib.pyplot as plt
import itertools
from collections import Counter
from collections import defaultdict


from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import BayesianRidge
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn import metrics
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def rating_class(data):
    res = []
    for x in data:
        if x <=3:
            res.append(1)
        elif x>= 4:
            res.append(3)
        else:
            res.append(2)
    return np.array(res)

def rating_class_2(data):
    res = []
    for x in data:
        if x <=3.5:
            res.append(1)
        else:
            res.append(2)
    return np.array(res)


def c_v(th,predict,Y):
    a = predict
    temp = np.vstack((a < 2-th[0],np.logical_and(a>2-th[0], a<=2+th[1]),a>(2+th[1])))
    temp = np.transpose(temp)
    
    return np.sum([np.where(i==True)[0][0] for i in temp] == Y)/float(len(Y))
        
     
def pd(x):
    return (1/x) /(np.sum(1/x))

def visual(pred,Y,n= 3):
    if n==3:
        res = np.zeros([3,3])
        for i in list(xrange(1,4)):
            res[:,(i-1)] = np.array([np.sum(np.logical_and(Y==i,pred == 1 )),np.sum(np.logical_and(Y==i,pred ==2 )),np.sum(np.logical_and(Y==i,pred ==3 ))])
        return res
    else:
        res = np.zeros([5,5])
        for i in list(xrange(1,6)):
            res[:,(i-1)] = np.array([np.sum(np.logical_and(Y==i,pred == 1 )),np.sum(np.logical_and(Y==i,pred ==2 )),np.sum(np.logical_and(Y==i,pred ==3 )),np.sum(np.logical_and(Y==i,pred ==4 )),np.sum(np.logical_and(Y==i,pred ==5 ))])
        return res        

if __name__ == "__main__":
    d = np.load("attri3.npy")    
    h = np.load("attri3_header.npy")
    location = np.loadtxt('location.txt',delimiter=",",dtype = object)
    loc = location[:,0:2].astype(float)
    
    data = pda.DataFrame(d)
    data.columns = h
    
    Y = rating_class(data['stars'])
    X_header = np.delete(h,[1,2,4])
    X = np.hstack((np.array(data[X_header]),loc))
    X[:,1] = X[:,1]*1.0/np.max(X[:,1])
    
    
    
    X_train, X_test, Y_train, Y_test = np.array(train_test_split(X, Y, train_size=.75))
    
    
    
    #3: Logistic Regression on star
    accu_1 = pda.DataFrame(columns=['C','L1','L2'])
    for i in range(-6, 5):
        c = np.power(10.0, i)
        model_1 = LogisticRegression(penalty='L1'.lower(), C=c)
        model_2 = LogisticRegression(penalty='L2'.lower(), C=c)
        model_1.fit(X_train, Y_train)
        model_2.fit(X_train, Y_train)
        accu_1.loc[i] = [c] + [metrics.accuracy_score(model_1.predict(X_test), Y_test),metrics.accuracy_score(model_2.predict(X_test), Y_test)]     
    

    #3_2 weighted logistic regression 
    dist = Counter()
    for i in Y_train:
        dist[i] += 1
    dist = np.array(dist.values())/(len(Y_train)*1.0)
    
    accu_1_2 = pda.DataFrame(columns=['C','L1','L2'])
    for i in range(-6, 5):
        c = np.power(10.0, i)
        model_1 = LogisticRegression(penalty='L1'.lower(), C=c,class_weight = {1:dist[0],2:dist[1],3:dist[2]})
        model_2 = LogisticRegression(penalty='L2'.lower(), C=c,class_weight = {1:dist[0],2:dist[1],3:dist[2]})
        model_1.fit(X_train, Y_train)
        model_2.fit(X_train, Y_train)
        accu_1_2.loc[i] = [c] + [metrics.accuracy_score(model_1.predict(X_test), Y_test),metrics.accuracy_score(model_2.predict(X_test), Y_test)]     
        
    
    #baseline2 : svm
    
    accu_2 = []
    for i in range(-6, 5):
        c = np.power(10.0, i)
        m = SVC(kernel = 'poly',C=c,class_weight = {1:dist[0],2:dist[1],3:dist[2]})
        m.fit(X_train,Y_train)
        accu_2.append(metrics.accuracy_score(m.predict(X_test),Y_test))
    
  
    
    #baseline1: knn
        
    accu_3 = []
    for k in [1,5,200]:
        knn = KNeighborsClassifier(k)
        knn.fit(X_train,Y_train) 
        accu_3.append(metrics.accuracy_score(Y_test, knn.predict(X_test)))
       
        #visual
        column_labels = ['A', 'B', 'C', 'D']
        column_labels = ['predicted 1','predicted 2','predicted 3']
        row_labels = ['1','2','3']
        data = visual(knn.predict(X_test),Y_test) 
        fig, ax = plt.subplots()
        heatmap = ax.pcolor(data, cmap=plt.cm.Blues)    
        ax.set_xticks(np.arange(data.shape[0])+0.5, minor=False)
        ax.set_yticks(np.arange(data.shape[1])+0.5, minor=False)
        ax.invert_yaxis()
        ax.xaxis.tick_top()
        
        ax.set_xticklabels(row_labels, minor=False)
        ax.set_yticklabels(column_labels, minor=False)
        plt.show()          

    #4 : Naive bayesian 
    
    center = np.zeros(len(X_train[1,]))
    for rating in list(xrange(1,4)):
        center = np.vstack((center,np.mean(X_train[np.where(Y_train==rating)[0],],0)))
    center = center[1:,:-2]    
    
    predict = []
    for i in X_train[:,:-2]:
        predict.append(pd(np.mean(np.abs(i - center),1))*dist / np.sum(pd(np.mean(np.abs(i - center),1))*dist))
    predict = np.array(predict)    
    
    Y_pred = np.array([np.where(i == max(i))[0][0]+1 for i in predict])
    np.sum([np.where(i == max(i))[0][0]+1 for i in predict] ==Y_train)/float(len(Y_train))
    print(visual(Y_pred,Y_train))
    
    
    predict = []
    for i in X_train[:,:-2]:
        predict.append(pd(np.mean(np.abs(i - center),1)) / np.sum(pd(np.mean(np.abs(i - center),1))))
    predict = np.array(predict)    
    
    Y_pred = np.array([np.where(i == max(i))[0][0]+1 for i in predict])
    print(visual(Y_pred,Y_train))
    
    #grid = list(itertools.product(np.linspace(0,0.2,20),repeat = 2))
    #acc_4 = []
    #for i in grid:
        #acc_4.append(c_v(i,predict,Y_train))
    #max_index = np.where(acc_4 == max(acc_4))[0][0]
    #th = grid[max_index]
    
    
    #5 : weighted naive bayesian 
    weight  = np.var(center,0)/sum(np.var(center,0))
    weight  = np.var(center / np.max(center,0) ,0) / np.sum(np.var(center / np.max(center,0) ,0))
     
    predict_ = []
    for i in X_train[:,:-2]:
        predict_.append(pd(np.inner(np.abs(i - center),weight)) /np.sum(pd(np.inner(np.abs(i - center),weight) ))) 
    predict_ = np.array(predict_)
    
    Y_pred_ = np.array([np.where(i == np.max(i))[0][0]+1 for i in predict_])
    print(visual(Y_pred_,Y_train))
    np.mean(Y_pred_ ==Y_train)

    #model 6: w2v logistic
    
    prior = X_train[:,0]
    X_300 = X_train[:,1:]
    
     
    
    zero_index = []
    for i,j in enumerate(X_300):
        temp =np.linalg.norm(j)
        if temp !=0:
            X_300[i,] = j/temp
        else:
            zero_index.append(i)
        
    X_300 = np.delete(X_300,zero_index,0)    
    Y_train = np.delete(Y_train,zero_index,0)
    prior = np.delete(prior,zero_index,0)
    
    transit = np.transpose(np.vstack((prior,Y_train)))
    markov = np.zeros((5,5))
    
    for data in zip(transit):
        today = data[0][0]
        tmr = data[0][1]
        markov[today-1,tmr-1] += 1
    
    markov_prior = markov/transit.shape[0] 
    dist = np.sum(markov_prior,1)
    
    for i,j in enumerate(markov_prior):
        temp =np.linalg.norm(j)
        markov_prior[i,] = j/temp
          
    
    
    accu_6 = pda.DataFrame(columns=['C','L1','L2'])
    for i in range(-10, 5):
        c = np.power(10.0, i)
        model_1 = SGDClassifier(loss = 'log', penalty='L1'.lower(), alpha =c,n_jobs = -1)
        model_2 = SGDClassifier(loss = 'log',penalty='L2'.lower(), alpha=c,n_jobs = -1)
        model_1.fit(X_300, Y_train)
        model_2.fit(X_300, Y_train)
        accu_6.loc[i] = [c] + [metrics.accuracy_score(model_1.predict(X_300), Y_train),metrics.accuracy_score(model_2.predict(X_300), Y_train)]     

    
    #model 7 weight weighted logistic w2v
    weight = {1:dist[0],2:dist[1],3:dist[2],4:dist[3],5:dist[4]}
    
    accu_7 = pda.DataFrame(columns=['C','L1','L2'])
    for i in range(-8, 2):
        c = np.power(10.0, i)
        model_1 = SGDClassifier(loss = 'log', penalty='L1'.lower(), alpha =c,n_jobs = -1,class_weight = weight)
        model_2 = SGDClassifier(loss = 'log',penalty='L2'.lower(), alpha=c,n_jobs = -1,class_weight = weight)
        model_1.fit(X_300, Y_train)
        model_2.fit(X_300, Y_train)
        accu_7.loc[i] = [c] + [metrics.accuracy_score(model_1.predict(X_300), Y_train),metrics.accuracy_score(model_2.predict(X_300), Y_train)]     

    
    #model 8 mixed logistic model with markov prior
    center = np.zeros(len(X_300[1,]))
    for rating in list(xrange(1,6)):
        center = np.vstack((center,np.mean(X_300[np.where(prior==rating)[0],],0)))
   
    center = center[1:,]   
    weight  = np.var(center,0)/sum(np.var(center,0))
    weight  = np.var(center / np.max(center,0) ,0) / np.sum(np.var(center / np.max(center,0) ,0))    
    
    predict = []
    for i,j in zip(X_300,prior):
        predict.append(pd(np.inner(np.abs(i - center),weight))*dist*markov_prior[j-1] / np.sum(pd(np.mean(np.abs(i - center),1))*dist*markov_prior[j-1]))
    predict = np.array(predict)    
    
    Y_pred = np.array([np.where(i == max(i))[0][0]+1 for i in predict])
    #np.sum([np.where(i == max(i))[0][0]+1 for i in predict] ==Y_train)/float(len(Y_train))
    print(visual(Y_pred,Y_train,5))
    
    predict_ = []
    for i,j in zip(X_300,prior):
        predict_.append(pd(np.inner(np.abs(i - center),weight))*markov_prior[j-1] /np.sum(pd(np.inner(np.abs(i - center),weight) )*markov_prior[j-1])) 
    predict_ = np.array(predict_)    
    Y_pred_ = np.array([np.where(i == max(i))[0][0]+1 for i in predict_])
    print(visual(Y_pred_,Y_train,5))
    
    #model 9 sgd for transfored text 
    add = np.zeros((163455,5))
    for i in [1,2,3,4,5]:
        add[:,i-1] = np.array([(prior ==i).astype(int)])
    X_305 = np.hstack((add,X_300)) 
   
    accu_9 = pda.DataFrame(columns=['C','L1','L2'])
    for i in range(-10, 5):
        c = np.power(10.0, i)
        model_1 = SGDClassifier(loss = 'hinge',penalty='L1'.lower(), alpha =c,n_jobs = -1)
        model_2 = SGDClassifier(loss = 'hinge',penalty='L2'.lower(), alpha=c,n_jobs = -1)
        model_1.fit(X_305, Y_train)
        model_2.fit(X_305, Y_train)
        accu_9.loc[i] = [c] + [metrics.accuracy_score(model_1.predict(X_305), Y_train),metrics.accuracy_score(model_2.predict(X_305), Y_train)]     
    