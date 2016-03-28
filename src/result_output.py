from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import coo_matrix, hstack, vstack
from sklearn.preprocessing import normalize
import gensim, logging
import numpy as np

def veclist(l):
    a = []
    for i in xrange(len(l)):
        a.append(np.zeros(300))
        if len(l[i]) > 0:
            for k in l[i]:
                try:
                    a[i] += model_word2vec[k]
                except KeyError:
                    continue
        else:
            a[i] = np.ones(300)/300
    return a

def sumlist(l):
    a = []
    for i in xrange(len(l)):
        if len(l[i]) >0:
            a.append(l[i][0])
            for j in xrange(1,len(l[i])):
                a[i] = a[i] + ' ' + l[i][j]
        else:
            a.append(' ')
    return a

word_models = ['tf-idf', 'word2vec all', 'word2vec with tagging']
models = ['LogisticRegression', 'DecisionTreeClassifier', 'Perceptron', 'knn']

for word_model in word_models:
    if word_model == 'tf-idf':
        print "Begin doing tf-idf"
        X_train = np.load("X_review_train.npy")
        Y_train = np.load("Y_review_train.npy")
        X_test = np.load("X_review_test.npy")
        Y_test = np.load("Y_review_test.npy")
        print "Finish loading orginal data"
        X_text_train = [i[1] for i in X_train]
        X_text_test = [i[1] for i in X_test]
        X_text_train = sumlist(X_text_train)
        X_text_test = sumlist(X_text_test)
        tfidf_vectorozer = TfidfVectorizer()
        tfidf_vectorozer.fit(X_text_train)
        X_train_tfidf = tfidf_vectorozer.transform(X_text_train)
        X_test_tfidf = tfidf_vectorozer.transform(X_text_test)
        print "Finish transforming to tfidf"
        X_train = np.load("X_final_train.npy")
        X_test = np.load("X_final_test.npy")
        Y_train = np.load("Y_final_train.npy")
        Y_test = np.load("Y_final_test.npy")
        X_train_score = np.array([X_train[:,0]])
        X_test_score = np.array([X_test[:,0]])
        X_train_after = hstack([X_train_score.T,X_train_tfidf])
        X_test_after = hstack([X_test_score.T,X_test_tfidf])
        # Delete useless data
        del X_train, X_test, X_text_train, X_text_test, X_train_score, X_test_score, X_train_tfidf, X_test_tfidf
        print "finish pre-training"
    elif word_model == 'word2vec all':
        print "Begin doing word2vec all"
        X_train = np.load("X_review_train.npy")
        Y_train = np.load("Y_review_train.npy")
        X_test = np.load("X_review_test.npy")
        Y_test = np.load("Y_review_test.npy")
        print "Finish loading orginal data."
        X_text_train = [i[1] for i in X_train]
        X_text_test = [i[1] for i in X_test]
        print "Begin transform into vector."
        model_word2vec = gensim.models.Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)
        print "Finish loading Google_model"
        X_text_train = veclist(X_text_train)
        X_text_test = veclist(X_text_test)
        del model_word2vec
        print "Finish transformation and begin normalizing."
        X_text_train = normalize(X_text_train)
        X_text_test = normalize(X_text_test)
        print "Finish normalizing."
        X_train = np.load("X_final_train.npy")
        X_test = np.load("X_final_test.npy")
        Y_train = np.load("Y_final_train.npy")
        Y_test = np.load("Y_final_test.npy")
        X_train_score = np.array([X_train[:,0]])
        X_test_score = np.array([X_test[:,0]])
        X_test_after = np.hstack([X_test_score.T, X_text_test])
        X_train_after = np.hstack([X_train_score.T, X_text_train])
        # Delete useless data
        del X_train, X_test, X_train_score, X_test_score, X_text_train, X_text_test
        print "Finish pre-training"
    else:
        print "Begin doing word2vec with tagging"
        X_train_after = np.load("X_final_train.npy")
        X_test_after = np.load("X_final_test.npy")
        Y_train = np.load("Y_final_train.npy")
        Y_test = np.load("Y_final_test.npy")
        print "Finish pre-training"

    # Begin different models
    for model in models:
        if model == 'LogisticRegression':
            print "Begin LogisticRegression"
            if word_model in ['tf-idf','word2vec all']:
                continue
            X1,X2,Y1,Y2 = train_test_split(X_train_after,Y_train[0],train_size=0.80)
            score = 0
            cross_value = []
            for i in xrange(-6, 4):
                c = np.power(10.0, i)
                trainmodel = LogisticRegression(penalty = 'l2', C=c)
                trainmodel.fit(X1, Y1)
                crossvalue = np.mean(cross_validation.cross_val_score(trainmodel, X_train_after, Y_train[0]))
                cross_value.append(crossvalue)
                if crossvalue > score:
                    c_best = c
                    score = crossvalue
                print "%d is finished" % i
            trainmodel = LogisticRegression(penalty = 'l2', C=c_best)
            trainmodel.fit(X_train_after, Y_train[0])
            print 'Best accuracy for LogisticRegression is %.3f and the parameter is %.3f' % (metrics.accuracy_score(trainmodel.predict(X_test_after), Y_test[0]),c_best)
            np.save(word_model + '_' + "cross_val_Logistic.npy", cross_value)
        elif model == 'DecisionTreeClassifier':
            print "Begin DecisionTreeClassifier"
            if word_model in ['tf-idf','word2vec all']:
                continue
            X1,X2,Y1,Y2 = train_test_split(X_train_after,Y_train[0],train_size=0.80)
            score = 0
            cross_value = []
            for c in [5,10,15]:
                trainmodel = DecisionTreeClassifier(max_depth = c,criterion="entropy")
                trainmodel.fit(X1, Y1)
                crossvalue = np.mean(cross_validation.cross_val_score(trainmodel, X_train_after, Y_train[0]))
                cross_value.append(crossvalue)
                if crossvalue > score:
                    c_best = c
                    score = crossvalue
                print "%d is finished" % c
            trainmodel = DecisionTreeClassifier(max_depth = c_best)
            trainmodel.fit(X_train_after, Y_train[0])
            print 'Best accuracy for DecisionTreeClassifier is %.3f and the max_depth is %d' % (metrics.accuracy_score(trainmodel.predict(X_test_after), Y_test[0]),c_best)
            np.save(word_model + '_' + "cross_val_Decision.npy", cross_value)
        elif model == 'Perceptron':
            print "Begin Perceptron"
            X1,X2,Y1,Y2 = train_test_split(X_train_after,Y_train[0],train_size=0.80)
            trainmodel = Perceptron()
            trainmodel.fit(X1, Y1)
            score = np.mean(cross_validation.cross_val_score(trainmodel, X_train_after, Y_train[0]))
            print 'Perceptron cross validation value is %.3f' % (score)
            print 'Best accuracy for Perceptron is %.3f.' % metrics.accuracy_score(trainmodel.predict(X_test_after), Y_test[0])
            np.save(word_model + '_' + "cross_val_Perceptron.npy", score)
            trainmodel = Perceptron()
            trainmodel.fit(X_train_after, Y_train[0])
        # else:
        #     print "Begin knn"
        #     X1,X2,Y1,Y2 = train_test_split(X_train_after,Y_train[0],train_size=0.80)
        #     score = 0
        #     cross_value = []
        #     for c in [10,100,1000,1500]:
        #         trainmodel = KNeighborsClassifier(n_neighbors = c)
        #         trainmodel.fit(X1, Y1)
        #         crossvalue = np.mean(cross_validation.cross_val_score(trainmodel, X_train_after, Y_train[0]))
        #         cross_value.append(crossvalue)
        #         if crossvalue > score:
        #             c_best = c
        #             score = crossvalue
        #         print "%d is finished" % c
        #     trainmodel = KNeighborsClassifier(n_neighbors = c_best)
        #     trainmodel.fit(X_train_after, Y_train[0])
        #     print 'Best accuracy for KNeighborsClassifier is %.3f and the n_neighbors is %.3f' (metrics.accuracy_score(trainmodel.predict(X_test_after), Y_test[0]),c_best)
        #     np.save(word_model + '_' + "cross_val_Decision.npy", cross_value)

        if model not in ['Perceptron', 'knn']:
            print 1
            for i in [1,2,3]:
                name = 'pos_tag' + str(i) + '_' + model + '_' + word_model
                Y_test_probability_1 = trainmodel.predict_proba(X_test_after)[:, 1]
                fpr, tpr, thresholds = metrics.roc_curve(Y_test[0], Y_test_probability_1, pos_label=i)
                np.save(name + '_fpr.npy', fpr)
                np.save(name + '_tpr.npy', tpr)
                np.save(name + '_thresholds.npy', thresholds)
