from source import *
from sklearn import metrics
from sklearn.cross_validation import train_test_split



data = np.load("review_combined.npy")
Y = data[:,2]
X = data[:,[1,3,4,5]]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=.75)


transit = np.transpose(np.vstack((X_train[:,0],Y_train)))

markov = np.zeros((5,5))

for data in zip(transit):
    today = data[0][0]
    tmr = data[0][1]
    markov[today-1,tmr-1] += 1

markov_prior = markov/transit.shape[0]

np.save("markov_prior.npy", markov_prior)
np.save("X_train.npy",X_train)
np.save("X_test.npy",X_test)
np.save("Y_train.npy",Y_train)
np.save("Y_test.npy",Y_test)