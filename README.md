# Yelp-Prediction

This is a data challenge on Yelp.com. You can find the detail [here](https://www.yelp.com/dataset_challenge). The challenge does not give out a specific target for challenger, so what you need to do totally depends on yourself.

## Abstract

In this solution, we aim to predict the future rating of each restaurant in Phoenix. The reason we focus on Phoenix is because most data is located in Phoenix. And we divide the rating as three class with nearly equal amount based on their distribution. We mainly focus on two data file. One is [`yelp_academic_dataset_business.json`](https://drive.google.com/file/d/0BzIp01PoYYptaGhsTktpV3d5S3c/view?usp=sharing) and the other one is [`yelp_academic_dataset_review.json`](https://drive.google.com/file/d/0BzIp01PoYYptZmxHbnJrUVNIc0U/view?usp=sharing). Now, I would like to present some of the method we use in our solutions. 

## Preprocessing (business part)

The data file is `.json`. So we parsed the data into a dictionary by using the python package [json](https://docs.python.org/2/library/json.html). Then we transform the dictionary into a [dataframe](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html). 

### Missing value

Missing value ofter appears on the attribute field in business data. We first using dummy variable to generate additional feature for missing value. Then using pca to shrinkage the number of features. What we get finally is about 20 features.

### Normalizing

To apply some mathematical method (e.g. SVM, logistic regression), we normalize the data by using python package [sklearn](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.normalize.html). It is very simple and efficient. In fact, in this project, we use a lot of function in sklearn. The official website of it is [this](http://scikit-learn.org/stable/index.html).

## Preprocessing (review part)

To deal with the review part, we used different methods.

1. [Tf-idf](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)
2. [word2vec](https://radimrehurek.com/gensim/models/word2vec.html) without tagging.
3. word2vec with [tagging](http://www.nltk.org/book/ch05.html).

On the tagging part, we mainly focus on the adjective words which we think that it can represent the sentiment of the review.

## Combination
We finally combine the above features and the past rating. We use this as X input and apply different methods `['LogisticRegression', 'DecisionTreeClassifier', 'Perceptron', 'knn']`.

## Result
We use [micro-auc](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score) as the evaluation for this method. The picture is as following.

![alt text](https://github.com/zw1074/Yelp-Prediction/edit/master/figures/micro-average ROC curve of different word model when using DecisionTreeClassifier.png)
![alt text](https://github.com/zw1074/Yelp-Prediction/edit/master/figures/micro-average ROC curve of different word model when using LogisticRegression.png)
![alt text](https://github.com/zw1074/Yelp-Prediction/edit/master/figures/micro-average ROC curve of different word model when using Perceptron.png)
