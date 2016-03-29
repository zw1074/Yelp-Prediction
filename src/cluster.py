from source import *
from sklearn import mixture
from sklearn.cluster import KMeans
from sklearn import tree
import matplotlib.pyplot as plt



if __name__ == '__main__':   
    raw = data_saver()
    data, b_id = categories_selector(raw,["Restaurants"])
    
    #review = data_saver(b_id,'rev')
    #loc_saver('location.txt',data)
    
    f2 = open('review.txt','rb')
    review = pickle.load(f2)
    f2.close()
    
    
    
    location = np.loadtxt('location.txt',delimiter=",")
    
    
    
    
    
    
    
    
    ######The clustering model
    g = mixture.GMM(n_components=4)
    g.fit(location) 
    pred = g.predict(location)
    #plt.scatter(location[:,0],location[:,1],c = pred)
    
    km = KMeans(n_clusters=5).fit_predict(location)
    #plt.scatter(location[:, 0], location[:, 1], c=km)
    
    #construct data
    temp =construct_matrix(data,attribute_generator(data),categories_generator(data))
    temp['loc'] = np.transpose(np.matrix(km))
    #data_new=  np.hstack((np.array(temp),np.transpose(np.matrix(km))))