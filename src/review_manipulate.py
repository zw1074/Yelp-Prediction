import pickle

with open('review.txt','rb') as f:
	review = pickle.load(f)

print review.head()